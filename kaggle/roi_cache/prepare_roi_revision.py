"""Prepare, but never activate, a controlled final ROI-cohort revision.

This utility exists for the documented patient-39 sequence-direction exception.
It writes a *provisional* revised manifest, a fresh 400-row template, an
auditable replacement decision, and a small human visual-QC manifest.  It does
not alter an existing SQLite annotation database, replace a cohort in place,
or claim that a radiologist approved the candidate.

Use only after a metadata/visual conflict has been documented.  A radiologist
must review the candidate and the QC list before a new revision cohort is
initialized from the generated files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from build_roi_cache import (
    LANDMARKS,
    SEED,
    _annotation_template,
    _collect_annotation_candidates,
    find_competition_root,
)
from roi_geometry import pick_series


PLANE_TOKEN_PATTERNS = {
    "Sagittal": re.compile(r"(^|[_ -])sag([_ -]|$)", re.IGNORECASE),
    "Coronal": re.compile(r"(^|[_ -])cor([_ -]|$)", re.IGNORECASE),
    "Axial": re.compile(r"(^|[_ -])(tra|ax)([_ -]|$)", re.IGNORECASE),
}

REFERENCE_REQUIREMENTS = {
    "medial_joint_line": ("Coronal", 0),
    "patellofemoral_center": ("Axial", 1),
    "suprapatellar_recess": ("Sagittal", 1),
}


def description_plane(description: str) -> str:
    """Return one unambiguous direction token or ``Unknown``."""
    matches = [plane for plane, pattern in PLANE_TOKEN_PATTERNS.items() if pattern.search(description)]
    return matches[0] if len(matches) == 1 else "Unknown"


def _seeded_key(study_uid: str, salt: str) -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{study_uid}".encode()).hexdigest()


def _series_description(
    competition: Path,
    study_uid: str,
    series_uid: str,
) -> str:
    directory = competition / "train_series" / study_uid / series_uid
    paths = sorted(directory.glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"no DICOMs in {directory}")
    header = pydicom.dcmread(paths[0], stop_before_pixels=True)
    return str(getattr(header, "SeriesDescription", ""))


def selected_series_descriptions(
    competition: Path,
    series_frame: pd.DataFrame,
    study_uid: str,
) -> dict[str, dict[str, str]]:
    """Resolve the production selector and return only human-auditable facts."""
    rows = series_frame.loc[
        series_frame.StudyInstanceUID.astype(str).eq(str(study_uid))
    ].to_dict("records")
    result: dict[str, dict[str, str]] = {}
    for landmark, (plane, fluid) in REFERENCE_REQUIREMENTS.items():
        selected = pick_series(rows, plane, fluid)
        if selected is None:
            raise ValueError(f"{study_uid}/{landmark}: no selected {plane} series")
        series_uid = str(selected["SeriesInstanceUID"])
        description = _series_description(competition, str(study_uid), series_uid)
        result[landmark] = {
            "declared_plane": plane,
            "series_uid": series_uid,
            "series_description": description,
            "description_plane": description_plane(description),
        }
    return result


def is_description_consistent(details: dict[str, dict[str, str]]) -> bool:
    """Reject only explicit directional contradictions; unknown descriptions need human QC."""
    return all(
        record["description_plane"] in {"Unknown", record["declared_plane"]}
        for record in details.values()
    )


def choose_replacement(
    competition: Path,
    old_record: dict[str, object],
    selected_ids: set[str],
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    """Choose a label-free, plane-consistent candidate matching the key stratum.

    Pixel spacing/depth bins are intentionally allowed to vary when no exact
    unused candidate exists; the decision JSON records that relaxation rather
    than pretending the replacement is an exact clone.
    """
    series_frame = pd.read_csv(
        competition / "train_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    candidates = _collect_annotation_candidates(competition, series_frame)
    keys = ("laterality", "complete_preferred_planes", "manufacturer_bin", "field_strength_bin")
    pool = [
        item for item in candidates
        if str(item["StudyInstanceUID"]) not in selected_ids
        and all(item.get(key) == old_record.get(key) for key in keys)
    ]
    if not pool:
        raise RuntimeError("no unused candidate matches laterality/completeness/vendor/field-strength stratum")
    for candidate in sorted(pool, key=lambda item: _seeded_key(str(item["StudyInstanceUID"]), "revision-patient-39")):
        details = selected_series_descriptions(competition, series_frame, str(candidate["StudyInstanceUID"]))
        if is_description_consistent(details):
            return dict(candidate), details
    raise RuntimeError("no plane-consistent replacement candidate found in matching stratum")


def stratified_qc_rows(
    manifest: pd.DataFrame,
    replaced_uid: str,
    replacement_uid: str,
    count: int,
) -> pd.DataFrame:
    """Choose a compact retained-study visual review without pathology labels."""
    retained = manifest.copy()
    retained["StudyInstanceUID"] = retained.StudyInstanceUID.astype(str)
    retained = retained.loc[retained.StudyInstanceUID.ne(replaced_uid)].copy()
    # Patient 1 is always included because its ACL-oblique PF description was
    # the one non-patient-39 explicit conflict found in the source audit.
    chosen = [retained.iloc[0]]
    features = ("manufacturer_bin", "laterality", "prior_partition", "complete_preferred_planes")
    covered: dict[str, set[object]] = {feature: {chosen[0].get(feature)} for feature in features}
    available = retained.iloc[1:].copy()
    while len(chosen) < count - 1 and not available.empty:
        def score(row: pd.Series) -> tuple[int, str]:
            novelty = sum(row.get(feature) not in covered[feature] for feature in features)
            return (-novelty, _seeded_key(str(row.StudyInstanceUID), "retained-visual-qc"))
        winner_index = min(available.index, key=lambda index: score(available.loc[index]))
        winner = available.loc[winner_index]
        chosen.append(winner)
        for feature in features:
            covered[feature].add(winner.get(feature))
        available = available.drop(index=winner_index)
    records = []
    for row in chosen:
        records.append({
            "review_kind": "retained_saved_annotation",
            "StudyInstanceUID": str(row.StudyInstanceUID),
            "original_patient_number": int(row.name) + 1,
            "review_instruction": "Review all four saved landmark views and green markers; do not alter a point unless clinically wrong.",
        })
    records.append({
        "review_kind": "replacement_candidate",
        "StudyInstanceUID": replacement_uid,
        "original_patient_number": "replacement",
        "review_instruction": "Confirm all three reference views are clinically suitable before annotating four landmarks.",
    })
    return pd.DataFrame(records)


def prepare_revision(arguments: argparse.Namespace) -> None:
    competition = find_competition_root(arguments.competition_root)
    manifest_path = Path(arguments.manifest)
    manifest = pd.read_csv(manifest_path, dtype={"StudyInstanceUID": str})
    if len(manifest) != 100 or manifest.StudyInstanceUID.nunique() != 100:
        raise ValueError("final manifest must contain exactly 100 unique studies")
    patient_number = int(arguments.patient_number)
    if not 1 <= patient_number <= len(manifest):
        raise ValueError("patient number is outside the manifest")
    old_index = patient_number - 1
    old_record = manifest.iloc[old_index].to_dict()
    selected_ids = set(manifest.StudyInstanceUID.astype(str))
    candidate, details = choose_replacement(competition, old_record, selected_ids)

    revised = manifest.copy()
    replacement = dict(candidate)
    replacement["selection_role"] = "cohort_complete_replacement_r1"
    replacement["prior_partition"] = old_record["prior_partition"]
    replacement["cohort_revision"] = "final-r1-provisional"
    replacement["replaces_study_uid"] = str(old_record["StudyInstanceUID"])
    for column in revised.columns:
        if column not in replacement:
            replacement[column] = old_record.get(column)
    for column in replacement:
        if column not in revised.columns:
            revised[column] = pd.NA
    revised.loc[old_index] = pd.Series(replacement)[revised.columns]
    if revised.StudyInstanceUID.nunique() != 100:
        raise RuntimeError("replacement created duplicate study IDs")
    if revised.prior_partition.value_counts().to_dict() != manifest.prior_partition.value_counts().to_dict():
        raise RuntimeError("replacement changed the frozen fit/audit count")
    if revised.laterality.value_counts().to_dict() != manifest.laterality.value_counts().to_dict():
        raise RuntimeError("replacement changed laterality balance")
    if revised.complete_preferred_planes.value_counts().to_dict() != manifest.complete_preferred_planes.value_counts().to_dict():
        raise RuntimeError("replacement changed complete/imperfect balance")

    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    revised.to_csv(output / "annotation_manifest.csv", index=False)
    _annotation_template(revised.to_dict("records")).to_csv(output / "roi_annotations.csv", index=False)
    # Never write an object array: NumPy rejects it under the safe default
    # ``allow_pickle=False`` and cache loaders must not require pickle just to
    # read study identifiers.
    np.save(
        output / "annotation_ids.npy",
        revised.StudyInstanceUID.astype(str).to_numpy(dtype="U"),
    )
    qc = stratified_qc_rows(
        manifest,
        str(old_record["StudyInstanceUID"]),
        str(candidate["StudyInstanceUID"]),
        arguments.qc_count,
    )
    qc.to_csv(output / "visual_qc_manifest.csv", index=False)
    decision = {
        "status": "provisional_pending_radiologist_qc",
        "revision": "final-r1",
        "replaced_patient_number": patient_number,
        "replaced_study_uid": str(old_record["StudyInstanceUID"]),
        "replacement_study_uid": str(candidate["StudyInstanceUID"]),
        "preserved_constraints": {
            "laterality": old_record.get("laterality"),
            "complete_preferred_planes": bool(old_record.get("complete_preferred_planes")),
            "manufacturer_bin": old_record.get("manufacturer_bin"),
            "field_strength_bin": old_record.get("field_strength_bin"),
            "prior_partition": old_record.get("prior_partition"),
            "final_counts": {
                "studies": 100,
                "fit": int((revised.prior_partition == "fit").sum()),
                "audit": int((revised.prior_partition == "audit").sum()),
            },
        },
        "replacement_metadata": candidate,
        "selected_series": details,
        "required_human_actions": [
            "Confirm all replacement reference views are clinically suitable.",
            "Review every row in visual_qc_manifest.csv.",
            "Do not initialize or train from this revision until those reviews pass.",
        ],
    }
    (output / "replacement_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(f"prepared provisional revision at {output}")
    print(f"replacement study: {candidate['StudyInstanceUID']}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patient-number", type=int, default=39)
    parser.add_argument("--qc-count", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_revision(parse_arguments())
