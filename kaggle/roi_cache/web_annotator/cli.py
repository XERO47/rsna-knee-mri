"""Administrator command-line operations for the remote ROI annotator.

Every mutating command is explicit and validated; nothing here silently
reshuffles a cohort, regenerates assignments, or guesses a decision the
specification leaves for human sign-off (see ``references/
REMOTE_ROI_ANNOTATION_APP_SPEC.md`` section 17 and this package's README).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROI_CACHE_DIR = Path(__file__).resolve().parents[1]
if str(_ROI_CACHE_DIR) not in sys.path:
    sys.path.insert(0, str(_ROI_CACHE_DIR))

import build_roi_cache  # noqa: E402  (sha256_file/find_competition_root reuse)

from . import services
from .config import AppConfig, FINAL_STUDY_COUNT, PILOT_STUDY_COUNT
from .database import Database, ManifestHashMismatch

APP_VERSION = "1.0.0"
EXPORT_COLUMNS = [
    "StudyInstanceUID", "landmark", "SeriesInstanceUID", "SOPInstanceUID",
    "row_px", "column_px", "patient_x_mm", "patient_y_mm", "patient_z_mm",
    "laterality", "annotator", "qc_status",
]


def _config(arguments: argparse.Namespace) -> AppConfig:
    config = AppConfig.from_work_dir(arguments.competition_root, arguments.work_dir)
    config.ensure_directories()
    return config


def _open_db(config: AppConfig) -> Database:
    return Database(config.db_path)


def _print(message: str) -> None:
    print(message, flush=True)


# -- init / preflight -------------------------------------------------------

def command_init_cohort(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    manifest_path = Path(arguments.manifest).resolve()
    template_path = Path(arguments.template).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not template_path.is_file():
        raise FileNotFoundError(f"annotation template not found: {template_path}")
    manifest_sha256 = build_roi_cache.sha256_file(manifest_path)
    template_sha256 = build_roi_cache.sha256_file(template_path)
    db = _open_db(config)
    try:
        db.backup(config.backup_dir, "pre-init", config.max_backups)
    except Exception:
        pass  # a fresh database has nothing to back up yet
    try:
        cohort = db.create_or_resume_cohort(
            arguments.cohort, arguments.phase, str(manifest_path), manifest_sha256,
            str(template_path), template_sha256,
        )
    except ManifestHashMismatch as error:
        raise SystemExit(str(error))
    manifest = pd.read_csv(manifest_path, dtype={"StudyInstanceUID": str})
    if manifest.StudyInstanceUID.duplicated().any():
        raise ValueError("manifest contains duplicate StudyInstanceUID values")
    expected = PILOT_STUDY_COUNT if arguments.phase == "pilot" else FINAL_STUDY_COUNT
    if len(manifest) != expected:
        raise ValueError(f"{arguments.phase} manifest must contain exactly {expected} studies")
    records = manifest.to_dict("records")
    db.replace_studies(cohort.id, records)
    _print(f"cohort {arguments.cohort!r} ({arguments.phase}) initialized with {len(records)} studies")
    _print(f"manifest sha256={manifest_sha256}")
    _print(f"template sha256={template_sha256}")


def command_preflight(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    failures: list[str] = []
    cohort = db.get_cohort(arguments.cohort)
    if cohort is None:
        raise SystemExit(f"cohort {arguments.cohort!r} is not initialized; run init-cohort first")

    manifest_path = Path(cohort.manifest_path)
    template_path = Path(cohort.template_path)
    if not manifest_path.is_file() or build_roi_cache.sha256_file(manifest_path) != cohort.manifest_sha256:
        failures.append("manifest is missing or its hash no longer matches the initialized cohort")
    if not template_path.is_file() or build_roi_cache.sha256_file(template_path) != cohort.template_sha256:
        failures.append("annotation template is missing or its hash no longer matches the initialized cohort")

    competition_root = Path(arguments.competition_root).resolve()
    if not (competition_root / "train_series.csv").is_file() or not (competition_root / "train_series").is_dir():
        failures.append(f"competition root missing train_series.csv/train_series: {competition_root}")

    expected_studies = PILOT_STUDY_COUNT if cohort.phase == "pilot" else FINAL_STUDY_COUNT
    expected_rows = expected_studies * len(services.LANDMARKS)
    studies = db.list_studies(cohort.id)
    if len(studies) != expected_studies:
        failures.append(f"{cohort.phase} cohort has {len(studies)} studies; expected {expected_studies}")
    study_uids = [row["study_uid"] for row in studies]
    if len(set(study_uids)) != len(study_uids):
        failures.append("studies table contains duplicate StudyInstanceUID values")

    if template_path.is_file():
        template = pd.read_csv(template_path, dtype={"StudyInstanceUID": str})
        if len(template) != expected_rows:
            failures.append(f"template has {len(template)} rows; expected {expected_rows}")
        duplicate_keys = template.duplicated(["StudyInstanceUID", "landmark"])
        if duplicate_keys.any():
            failures.append("template contains duplicate StudyInstanceUID/landmark rows")
        counts = template.groupby("StudyInstanceUID").landmark.nunique()
        if not (counts == len(services.LANDMARKS)).all():
            failures.append("template does not have exactly four unique landmarks per study")
        unknown = set(template.landmark) - set(services.LANDMARKS)
        if unknown:
            failures.append(f"template contains unknown landmark names: {sorted(unknown)}")

    science = services.ScientificService(competition_root)
    try:
        science.check_data_root_available()
        train_series = pd.read_csv(competition_root / "train_series.csv", dtype={"StudyInstanceUID": str})
        known_studies = set(train_series.StudyInstanceUID)
        decoded_series: set[str] = set()
        for study_uid in study_uids:
            if study_uid not in known_studies:
                failures.append(f"study {study_uid} is absent from train_series.csv")
                continue
            if not (competition_root / "train_series" / study_uid).is_dir():
                failures.append(f"study {study_uid} has no on-disk directory")
                continue
            for landmark in services.LANDMARKS:
                try:
                    task = science.resolve_task(study_uid, landmark)
                except services.ReferenceSeriesUnavailable as error:
                    failures.append(str(error))
                    continue
                if task.series.series_uid not in decoded_series:
                    try:
                        build_roi_cache._read_pixels(task.series.middle.path)
                    except Exception as error:  # noqa: BLE001
                        failures.append(
                            f"pixel decode failed for {study_uid}/{task.series.series_uid}: {error}"
                        )
                    decoded_series.add(task.series.series_uid)
    except services.DataRootUnavailable as error:
        failures.append(str(error))

    all_annotations = db.list_annotations_for_cohort(cohort.id)
    assigned_study_uids = {row["study_uid"] for row in all_annotations}
    if cohort.state == "open" and not assigned_study_uids.issuperset(set(study_uids)):
        missing = sorted(set(study_uids) - assigned_study_uids)
        if missing:
            failures.append(f"studies without any assignment: {missing[:5]}")
    outside_cohort = assigned_study_uids - set(study_uids)
    if outside_cohort:
        failures.append(f"assignments reference studies outside the cohort: {sorted(outside_cohort)[:5]}")

    for directory in (config.work_dir, config.backup_dir, config.log_dir):
        if not os.access(directory, os.W_OK):
            failures.append(f"directory is not writable: {directory}")

    free_bytes = shutil.disk_usage(config.work_dir).free
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < config.min_free_disk_gb:
        failures.append(f"free disk space {free_gb:.1f} GiB is below the configured floor {config.min_free_disk_gb} GiB")

    if failures:
        _print("PREFLIGHT FAILED:")
        for failure in failures:
            _print(f"  - {failure}")
        raise SystemExit(1)
    _print(f"PREFLIGHT PASSED for cohort {arguments.cohort!r} ({cohort.phase}, {len(studies)} studies)")


# -- readers ------------------------------------------------------------

def command_create_reader(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    annotator_id, token = db.create_annotator(arguments.name)
    url = f"{arguments.base_url.rstrip('/')}/login?token={token}"
    _print(f"created reader {arguments.name!r} (id={annotator_id})")
    _print("Private bootstrap link (share once, then it is exchanged for a session):")
    _print(url)
    _print("This token is shown only once and is not stored in plaintext.")


def command_revoke_reader(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    db.revoke_annotator(arguments.name)
    _print(f"revoked reader {arguments.name!r}; existing sessions were deleted")


# -- assignment -----------------------------------------------------------

def command_assign(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    cohort = db.get_cohort(arguments.cohort)
    if cohort is None:
        raise SystemExit(f"cohort {arguments.cohort!r} is not initialized")
    annotator = db.get_annotator_by_name(arguments.reader)
    if annotator is None:
        raise SystemExit(f"reader {arguments.reader!r} does not exist; run create-reader first")
    all_studies = [row["study_uid"] for row in db.list_studies(cohort.id)]
    if arguments.study_list == "all":
        study_uids = all_studies
    else:
        study_list_frame = pd.read_csv(arguments.study_list, dtype={"StudyInstanceUID": str})
        study_uids = study_list_frame.StudyInstanceUID.tolist()
    created = db.assign_studies(
        cohort.id, int(annotator["id"]), study_uids, arguments.group, services.LANDMARKS
    )
    _print(f"assigned {len(study_uids)} studies to {arguments.reader!r} ({created} new assignment rows)")


# -- status / backup -------------------------------------------------------

def command_status(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    cohort = db.get_cohort(arguments.cohort)
    if cohort is None:
        raise SystemExit(f"cohort {arguments.cohort!r} is not initialized")
    _print(f"cohort={cohort.name} phase={cohort.phase} state={cohort.state}")
    annotators = db._connection.execute("SELECT * FROM annotators").fetchall()
    for annotator in annotators:
        assignments = db.list_assignments(cohort.id, annotator["id"])
        if not assignments:
            continue
        annotations = db.list_annotations_for_reader(cohort.id, annotator["id"])
        by_key = {(row["study_uid"], row["landmark"]): row["state"] for row in annotations}
        complete_patients = 0
        flagged = 0
        for assignment in assignments:
            states = [by_key.get((assignment["study_uid"], landmark), "incomplete") for landmark in services.LANDMARKS]
            if all(state == "pass" for state in states):
                complete_patients += 1
            flagged += states.count("unable")
        status = "active" if annotator["active"] else "revoked"
        _print(
            f"  reader={annotator['display_name']} ({status}) "
            f"patients={complete_patients}/{len(assignments)} flagged_tasks={flagged}"
        )
    db_size_mb = config.db_path.stat().st_size / (1024 * 1024) if config.db_path.exists() else 0.0
    _print(f"database size: {db_size_mb:.2f} MiB at {config.db_path}")
    backups = sorted(config.backup_dir.glob("annotator-*.sqlite3"))
    _print(f"backups: {len(backups)} in {config.backup_dir}")


def command_backup(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    target = db.backup(config.backup_dir, arguments.reason, config.max_backups)
    _print(f"backup written to {target}")


# -- export -----------------------------------------------------------------

def _annotation_row_to_export(row: dict[str, Any], annotator_name: str) -> dict[str, Any]:
    if row["state"] == "pass":
        return {
            "StudyInstanceUID": row["study_uid"],
            "landmark": row["landmark"],
            "SeriesInstanceUID": row["series_uid"],
            "SOPInstanceUID": row["sop_uid"],
            "row_px": row["row_px"],
            "column_px": row["column_px"],
            "patient_x_mm": row["patient_x_mm"],
            "patient_y_mm": row["patient_y_mm"],
            "patient_z_mm": row["patient_z_mm"],
            "laterality": row["laterality"],
            "annotator": annotator_name,
            "qc_status": "pass",
        }
    return {
        "StudyInstanceUID": row["study_uid"],
        "landmark": row["landmark"],
        "SeriesInstanceUID": "",
        "SOPInstanceUID": "",
        "row_px": "",
        "column_px": "",
        "patient_x_mm": "",
        "patient_y_mm": "",
        "patient_z_mm": "",
        "laterality": row["laterality"] or "",
        "annotator": annotator_name,
        "qc_status": row["state"],
    }


def _atomic_write_csv(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False, columns=EXPORT_COLUMNS)
    os.replace(temporary, output)


def _verify_pass_row_round_trip(science: services.ScientificService, row: dict[str, Any]) -> None:
    task = science.resolve_task(row["study_uid"], row["landmark"])
    point, resolved_sop_uid, _ = science.compute_and_validate_point(
        task, row["sop_uid"], row["row_px"], row["column_px"]
    )
    if resolved_sop_uid != row["sop_uid"]:
        raise ValueError(f"stored SOP UID does not resolve for {row['study_uid']}/{row['landmark']}")
    stored = np.asarray([row["patient_x_mm"], row["patient_y_mm"], row["patient_z_mm"]], dtype=np.float64)
    if float(np.max(np.abs(stored - point))) > 1e-3:
        raise ValueError(
            f"server recomputed patient XYZ does not match stored value for "
            f"{row['study_uid']}/{row['landmark']}"
        )


def command_export_reader(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    cohort = db.get_cohort(arguments.cohort)
    if cohort is None:
        raise SystemExit(f"cohort {arguments.cohort!r} is not initialized")
    annotator = db.get_annotator_by_name(arguments.reader)
    if annotator is None:
        raise SystemExit(f"reader {arguments.reader!r} does not exist")
    rows = [dict(row) for row in db.list_annotations_for_reader(cohort.id, int(annotator["id"]))]
    science = services.ScientificService(Path(arguments.competition_root).resolve())
    for row in rows:
        if row["state"] == "pass":
            _verify_pass_row_round_trip(science, row)
    exported = [_annotation_row_to_export(row, arguments.reader) for row in rows]
    frame = pd.DataFrame(exported, columns=EXPORT_COLUMNS)
    if frame.duplicated(["StudyInstanceUID", "landmark"]).any():
        raise ValueError("export would contain duplicate StudyInstanceUID/landmark rows")
    output = Path(arguments.output)
    db.backup(config.backup_dir, f"pre-export-{arguments.reader}", config.max_backups)
    _atomic_write_csv(frame, output)
    metadata = {
        "cohort": cohort.name,
        "cohort_phase": cohort.phase,
        "manifest_sha256": cohort.manifest_sha256,
        "reader": arguments.reader,
        "exported_at_utc": pd.Timestamp.utcnow().isoformat(),
        "row_count": len(frame),
        "pass_count": int((frame.qc_status == "pass").sum()),
        "incomplete_count": int((frame.qc_status == "incomplete").sum()),
        "unable_count": int((frame.qc_status == "unable").sum()),
        "application_version": APP_VERSION,
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    db.record_export_event(
        [row["id"] for row in db.list_annotations_for_reader(cohort.id, int(annotator["id"]))],
        actor=f"cli:export-reader:{arguments.reader}",
        request_id=f"export-{pd.Timestamp.utcnow().isoformat()}",
    )
    _print(f"exported {len(frame)} rows to {output}")
    _print(f"metadata written to {metadata_path}")


# -- agreement report --------------------------------------------------

def command_agreement_report(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    cohort = db.get_cohort(arguments.cohort)
    if cohort is None:
        raise SystemExit(f"cohort {arguments.cohort!r} is not initialized")
    all_annotations = [dict(row) for row in db.list_annotations_for_cohort(cohort.id)]
    by_study_landmark: dict[tuple[str, str], list[dict[str, Any]]] = {}
    annotator_names = {
        row["id"]: row["display_name"] for row in db._connection.execute("SELECT * FROM annotators")
    }
    for row in all_annotations:
        by_study_landmark.setdefault((row["study_uid"], row["landmark"]), []).append(row)
    records: list[dict[str, Any]] = []
    for (study_uid, landmark), rows in by_study_landmark.items():
        passing = [row for row in rows if row["state"] == "pass"]
        if len(passing) < 2:
            continue
        for i in range(len(passing)):
            for j in range(i + 1, len(passing)):
                a, b = passing[i], passing[j]
                distance = float(np.sqrt(
                    (a["patient_x_mm"] - b["patient_x_mm"]) ** 2
                    + (a["patient_y_mm"] - b["patient_y_mm"]) ** 2
                    + (a["patient_z_mm"] - b["patient_z_mm"]) ** 2
                ))
                records.append({
                    "StudyInstanceUID": study_uid,
                    "landmark": landmark,
                    "reader_a": annotator_names.get(a["annotator_id"], str(a["annotator_id"])),
                    "reader_b": annotator_names.get(b["annotator_id"], str(b["annotator_id"])),
                    "reader_a_x_mm": a["patient_x_mm"], "reader_a_y_mm": a["patient_y_mm"], "reader_a_z_mm": a["patient_z_mm"],
                    "reader_b_x_mm": b["patient_x_mm"], "reader_b_y_mm": b["patient_y_mm"], "reader_b_z_mm": b["patient_z_mm"],
                    "distance_mm": distance,
                    "series_differs": a["series_uid"] != b["series_uid"],
                    "sop_differs": a["sop_uid"] != b["sop_uid"],
                    "revision_a": a["revision"], "revision_b": b["revision"],
                })
    frame = pd.DataFrame(records).sort_values("distance_mm", ascending=False)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary: dict[str, Any] = {}
    for landmark in services.LANDMARKS:
        values = frame.loc[frame.landmark == landmark, "distance_mm"].tolist()
        if not values:
            continue
        sorted_values = sorted(values)
        summary[landmark] = {
            "n": len(values),
            "median_mm": statistics.median(values),
            "iqr_mm": [
                float(np.percentile(sorted_values, 25)),
                float(np.percentile(sorted_values, 75)),
            ],
            "p95_mm": float(np.percentile(sorted_values, 95)),
            "max_mm": max(values),
        }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _print(f"wrote {len(frame)} overlap comparisons to {output}")
    _print(f"wrote per-landmark summary to {summary_path}")
    _print(
        "This report does not set a pass/fail threshold. The project owner and "
        "both radiologists must review this distribution and record an explicit "
        "adjudication rule before proceeding (spec section 12.2)."
    )


# -- adjudication -----------------------------------------------------------

def command_adjudicate(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    cohort = db.get_cohort(arguments.cohort)
    if cohort is None:
        raise SystemExit(f"cohort {arguments.cohort!r} is not initialized")
    if arguments.landmark not in services.LANDMARKS:
        raise SystemExit(f"unknown landmark: {arguments.landmark}")
    record: dict[str, Any] = {
        "study_uid": arguments.study,
        "landmark": arguments.landmark,
        "decision": arguments.decision,
        "reviewer": arguments.reviewer,
        "reason": arguments.reason,
    }
    if arguments.decision == "choose_reader":
        annotator = db.get_annotator_by_name(arguments.chosen_reader)
        if annotator is None:
            raise SystemExit(f"reader {arguments.chosen_reader!r} does not exist")
        source = db.get_annotation(cohort.id, int(annotator["id"]), arguments.study, arguments.landmark)
        if source is None or source["state"] != "pass":
            raise SystemExit(
                f"{arguments.chosen_reader!r} has no passing annotation for "
                f"{arguments.study}/{arguments.landmark}"
            )
        record["chosen_annotator_id"] = int(annotator["id"])
        record["source_revision_a"] = source["revision"]
    elif arguments.decision == "replacement":
        science = services.ScientificService(Path(arguments.competition_root).resolve())
        task = science.resolve_task(arguments.study, arguments.landmark)
        point, resolved_sop_uid, _ = science.compute_and_validate_point(
            task, arguments.replacement_sop_uid, arguments.replacement_row, arguments.replacement_column
        )
        record.update({
            "replacement_series_uid": task.series.series_uid,
            "replacement_sop_uid": resolved_sop_uid,
            "replacement_row_px": arguments.replacement_row,
            "replacement_column_px": arguments.replacement_column,
            "replacement_patient_x_mm": float(point[0]),
            "replacement_patient_y_mm": float(point[1]),
            "replacement_patient_z_mm": float(point[2]),
        })
    else:
        raise SystemExit(f"unknown decision: {arguments.decision}")
    db.upsert_adjudication(cohort.id, record)
    _print(f"recorded adjudication for {arguments.study}/{arguments.landmark}: {arguments.decision}")


# -- canonical export -----------------------------------------------------

def _canonical_rows(db: Database, cohort) -> list[dict[str, Any]]:
    studies = db.list_studies(cohort.id)
    annotators = {
        row["id"]: row["display_name"] for row in db._connection.execute("SELECT * FROM annotators")
    }
    assignments_by_study: dict[str, list[int]] = {}
    for study in studies:
        assignments_by_study[study["study_uid"]] = []
    for row in db._connection.execute("SELECT * FROM assignments WHERE cohort_id = ?", (cohort.id,)):
        assignments_by_study.setdefault(row["study_uid"], []).append(row["annotator_id"])
    adjudications = {
        (row["study_uid"], row["landmark"]): row for row in db.list_adjudications(cohort.id)
    }
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for study in studies:
        study_uid = study["study_uid"]
        reader_ids = assignments_by_study.get(study_uid, [])
        for landmark in services.LANDMARKS:
            if len(reader_ids) >= 2:
                adjudication = adjudications.get((study_uid, landmark))
                if adjudication is None:
                    problems.append(f"missing adjudication for overlap study {study_uid}/{landmark}")
                    continue
                if adjudication["decision"] == "choose_reader":
                    annotation = db.get_annotation(
                        cohort.id, adjudication["chosen_annotator_id"], study_uid, landmark
                    )
                    if annotation is None or annotation["state"] != "pass":
                        problems.append(f"adjudicated reader annotation is missing/not pass for {study_uid}/{landmark}")
                        continue
                    rows.append({
                        "StudyInstanceUID": study_uid, "landmark": landmark,
                        "SeriesInstanceUID": annotation["series_uid"], "SOPInstanceUID": annotation["sop_uid"],
                        "row_px": annotation["row_px"], "column_px": annotation["column_px"],
                        "patient_x_mm": annotation["patient_x_mm"], "patient_y_mm": annotation["patient_y_mm"],
                        "patient_z_mm": annotation["patient_z_mm"], "laterality": annotation["laterality"],
                        "annotator": annotators.get(adjudication["chosen_annotator_id"], "unknown"),
                        "qc_status": "pass",
                    })
                else:
                    rows.append({
                        "StudyInstanceUID": study_uid, "landmark": landmark,
                        "SeriesInstanceUID": adjudication["replacement_series_uid"],
                        "SOPInstanceUID": adjudication["replacement_sop_uid"],
                        "row_px": adjudication["replacement_row_px"], "column_px": adjudication["replacement_column_px"],
                        "patient_x_mm": adjudication["replacement_patient_x_mm"],
                        "patient_y_mm": adjudication["replacement_patient_y_mm"],
                        "patient_z_mm": adjudication["replacement_patient_z_mm"],
                        "laterality": study["laterality"],
                        "annotator": f"adjudicated:{adjudication['reviewer']}",
                        "qc_status": "pass",
                    })
            elif len(reader_ids) == 1:
                annotation = db.get_annotation(cohort.id, reader_ids[0], study_uid, landmark)
                if annotation is None or annotation["state"] != "pass":
                    problems.append(f"single-reader annotation is missing/not pass for {study_uid}/{landmark}")
                    continue
                rows.append({
                    "StudyInstanceUID": study_uid, "landmark": landmark,
                    "SeriesInstanceUID": annotation["series_uid"], "SOPInstanceUID": annotation["sop_uid"],
                    "row_px": annotation["row_px"], "column_px": annotation["column_px"],
                    "patient_x_mm": annotation["patient_x_mm"], "patient_y_mm": annotation["patient_y_mm"],
                    "patient_z_mm": annotation["patient_z_mm"], "laterality": annotation["laterality"],
                    "annotator": annotators.get(reader_ids[0], "unknown"),
                    "qc_status": "pass",
                })
            else:
                problems.append(f"study {study_uid} has no assignment for {landmark}")
    if problems:
        raise ValueError("canonical export is incomplete:\n" + "\n".join(problems[:20]))
    return rows


def _validate_canonical_structure(frame: pd.DataFrame, expected_rows: int) -> None:
    """Mirror the structural checks at the top of ``build_roi_cache.command_fit_priors``.

    This intentionally duplicates only the cheap dataframe assertions (no
    DICOM geometry) so ``build_roi_cache.py`` stays untouched; the actual
    geometric prior-fitting gate must still be run separately.
    """
    required = {
        "StudyInstanceUID", "landmark", "patient_x_mm", "patient_y_mm", "patient_z_mm",
        "laterality", "qc_status",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"canonical export is missing columns: {missing}")
    if frame.duplicated(["StudyInstanceUID", "landmark"]).any():
        raise ValueError("canonical export contains duplicate StudyInstanceUID/landmark rows")
    counts = frame.groupby("StudyInstanceUID").landmark.nunique()
    if not (counts == len(services.LANDMARKS)).all():
        raise ValueError("canonical export does not have exactly four unique landmarks per study")
    if len(frame) != expected_rows:
        raise ValueError(f"canonical export has {len(frame)} rows; expected exactly {expected_rows}")
    if not (frame.qc_status == "pass").all():
        raise ValueError("canonical export contains a row that is not qc_status=pass")
    coordinates = frame[["patient_x_mm", "patient_y_mm", "patient_z_mm"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(coordinates.to_numpy()).all():
        raise ValueError("canonical export contains a non-finite patient coordinate")
    unknown = set(frame.landmark) - set(services.LANDMARKS)
    if unknown:
        raise ValueError(f"canonical export contains unknown landmark names: {sorted(unknown)}")


def command_canonical_export(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    cohort = db.get_cohort(arguments.cohort)
    if cohort is None:
        raise SystemExit(f"cohort {arguments.cohort!r} is not initialized")
    expected_rows = (PILOT_STUDY_COUNT if cohort.phase == "pilot" else FINAL_STUDY_COUNT) * len(services.LANDMARKS)
    rows = _canonical_rows(db, cohort)
    frame = pd.DataFrame(rows, columns=EXPORT_COLUMNS)
    _validate_canonical_structure(frame, expected_rows)
    science = services.ScientificService(Path(arguments.competition_root).resolve())
    for row in rows:
        _verify_pass_row_round_trip(science, {
            "study_uid": row["StudyInstanceUID"], "landmark": row["landmark"],
            "sop_uid": row["SOPInstanceUID"], "row_px": row["row_px"], "column_px": row["column_px"],
            "patient_x_mm": row["patient_x_mm"], "patient_y_mm": row["patient_y_mm"],
            "patient_z_mm": row["patient_z_mm"],
        })
    output = Path(arguments.output)
    db.backup(config.backup_dir, "pre-canonical-export", config.max_backups)
    _atomic_write_csv(frame, output)
    metadata = {
        "cohort": cohort.name,
        "cohort_phase": cohort.phase,
        "manifest_sha256": cohort.manifest_sha256,
        "row_count": len(frame),
        "exported_at_utc": pd.Timestamp.utcnow().isoformat(),
        "application_version": APP_VERSION,
        "note": (
            "Structural checks passed here. Run build_roi_cache.py fit-priors "
            "separately; this application does not itself mark the prior gate "
            "as passed."
        ),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    _print(f"wrote canonical export with {len(frame)} rows to {output}")
    _print(
        "Next step (not run automatically): "
        "python kaggle/roi_cache/build_roi_cache.py fit-priors --phase "
        f"{cohort.phase} --annotations {output} "
        + ("" if cohort.phase == "pilot" else f"--manifest {cohort.manifest_path} ")
        + f"--competition-root {arguments.competition_root} --output <priors.json>"
    )


# -- lifecycle --------------------------------------------------------------

def command_close_cohort(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    db.backup(config.backup_dir, "pre-close", config.max_backups)
    db.close_cohort(arguments.cohort)
    _print(f"cohort {arguments.cohort!r} is now closed (read-only)")


def command_reopen(arguments: argparse.Namespace) -> None:
    config = _config(arguments)
    db = _open_db(config)
    cohort = db.get_cohort(arguments.cohort)
    if cohort is None:
        raise SystemExit(f"cohort {arguments.cohort!r} is not initialized")
    annotator = db.get_annotator_by_name(arguments.reader)
    if annotator is None:
        raise SystemExit(f"reader {arguments.reader!r} does not exist")
    record = db.reopen(
        cohort.id, int(annotator["id"]), arguments.study, arguments.landmark,
        actor=f"cli:reopen:{arguments.reviewer}", reason=arguments.reason,
    )
    _print(f"reopened {arguments.study}/{arguments.landmark} for {arguments.reader!r}; new revision={record['revision']}")


def command_serve(arguments: argparse.Namespace) -> None:
    import uvicorn

    from .app import create_app
    from . import services as services_module

    config = _config(arguments)
    db = _open_db(config)
    cohort = db.get_cohort(arguments.cohort)
    if cohort is None:
        raise SystemExit(f"cohort {arguments.cohort!r} is not initialized; run init-cohort first")
    if config.db_path.exists():
        target = db.backup(config.backup_dir, "pre-start", config.max_backups)
        _print(f"pre-start backup written to {target}")
    science = services_module.ScientificService(
        config.competition_root,
        slice_cache_items=config.slice_cache_items,
        slice_cache_bytes=config.slice_cache_bytes,
        max_concurrent_decodes=config.max_concurrent_decodes,
    )
    app = create_app(config, db, science, arguments.cohort)
    _print(f"serving cohort {arguments.cohort!r} on http://{arguments.host}:{arguments.port}")
    _print(f"database: {config.db_path}")
    # The clean-shutdown backup runs inside the app's ASGI lifespan (see
    # app.create_app), not here: uvicorn re-raises the captured OS signal
    # with the default handler restored immediately after uvicorn.run()
    # returns, which would terminate the process before any code placed
    # after this call could run.
    uvicorn.run(app, host=arguments.host, port=arguments.port, workers=1, log_level="info")


# -- argument parsing --------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="roi_work/web_annotator")
    parser.add_argument("--competition-root", default="../data-root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_cohort = subparsers.add_parser("init-cohort", help="Register a frozen manifest/template as a cohort")
    init_cohort.add_argument("--cohort", required=True)
    init_cohort.add_argument("--phase", required=True, choices=("pilot", "final"))
    init_cohort.add_argument("--manifest", required=True)
    init_cohort.add_argument("--template", required=True)
    init_cohort.set_defaults(handler=command_init_cohort)

    preflight = subparsers.add_parser("preflight", help="Run all startup invariant checks")
    preflight.add_argument("--cohort", required=True)
    preflight.set_defaults(handler=command_preflight)

    create_reader = subparsers.add_parser("create-reader", help="Create a radiologist token")
    create_reader.add_argument("--name", required=True)
    create_reader.add_argument("--base-url", default="http://127.0.0.1:8000")
    create_reader.set_defaults(handler=command_create_reader)

    revoke_reader = subparsers.add_parser("revoke-reader", help="Deactivate a radiologist token")
    revoke_reader.add_argument("--name", required=True)
    revoke_reader.set_defaults(handler=command_revoke_reader)

    assign = subparsers.add_parser("assign", help="Assign explicit studies to a reader")
    assign.add_argument("--cohort", required=True)
    assign.add_argument("--reader", required=True)
    assign.add_argument("--study-list", required=True, help="'all' or a CSV with a StudyInstanceUID column")
    assign.add_argument("--group", required=True, choices=("overlap", "single_reader"))
    assign.set_defaults(handler=command_assign)

    status = subparsers.add_parser("status", help="Print per-reader progress and storage size")
    status.add_argument("--cohort", required=True)
    status.set_defaults(handler=command_status)

    backup = subparsers.add_parser("backup", help="Create a timestamped database backup")
    backup.add_argument("--reason", default="manual")
    backup.set_defaults(handler=command_backup)

    export_reader = subparsers.add_parser("export-reader", help="Atomic per-reader CSV export")
    export_reader.add_argument("--cohort", required=True)
    export_reader.add_argument("--reader", required=True)
    export_reader.add_argument("--output", required=True)
    export_reader.set_defaults(handler=command_export_reader)

    agreement = subparsers.add_parser("agreement-report", help="Blinded inter-reader agreement report")
    agreement.add_argument("--cohort", required=True)
    agreement.add_argument("--output", required=True)
    agreement.set_defaults(handler=command_agreement_report)

    adjudicate = subparsers.add_parser("adjudicate", help="Record one explicit adjudication decision")
    adjudicate.add_argument("--cohort", required=True)
    adjudicate.add_argument("--study", required=True)
    adjudicate.add_argument("--landmark", required=True)
    adjudicate.add_argument("--decision", required=True, choices=("choose_reader", "replacement"))
    adjudicate.add_argument("--reviewer", required=True)
    adjudicate.add_argument("--reason", default=None)
    adjudicate.add_argument("--chosen-reader", default=None)
    adjudicate.add_argument("--replacement-sop-uid", default=None)
    adjudicate.add_argument("--replacement-row", type=float, default=None)
    adjudicate.add_argument("--replacement-column", type=float, default=None)
    adjudicate.set_defaults(handler=command_adjudicate)

    canonical_export = subparsers.add_parser("canonical-export", help="Combine adjudicated + single-reader rows")
    canonical_export.add_argument("--cohort", required=True)
    canonical_export.add_argument("--output", required=True)
    canonical_export.set_defaults(handler=command_canonical_export)

    close_cohort = subparsers.add_parser("close-cohort", help="Make a cohort read-only")
    close_cohort.add_argument("--cohort", required=True)
    close_cohort.set_defaults(handler=command_close_cohort)

    reopen = subparsers.add_parser("reopen", help="Administratively reset one task to incomplete")
    reopen.add_argument("--cohort", required=True)
    reopen.add_argument("--reader", required=True)
    reopen.add_argument("--study", required=True)
    reopen.add_argument("--landmark", required=True)
    reopen.add_argument("--reviewer", required=True)
    reopen.add_argument("--reason", required=True)
    reopen.set_defaults(handler=command_reopen)

    serve = subparsers.add_parser("serve", help="Run the local Uvicorn server")
    serve.add_argument("--cohort", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=command_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
