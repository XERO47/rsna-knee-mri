"""Run the complete local Arm C pilot workflow without a notebook.

The script creates (or resumes) the deterministic 20-study selection, opens
the local Matplotlib annotation window when clicks are missing, fits pilot
priors, renders a smoke image, and builds one complete pilot shard.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from annotate_roi_pilot import ROIAnnotator
from build_roi_cache import (
    command_build_pilot,
    command_fit_priors,
    command_select_pilot,
    command_smoke,
    log,
)
from roi_geometry import LANDMARKS


def annotations_complete(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "annotation CSV does not exist"
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    expected_rows = 20 * len(LANDMARKS)
    if len(frame) != expected_rows:
        return False, f"annotation CSV has {len(frame)} rows; expected {expected_rows}"
    approved = frame.qc_status.astype(str).str.lower().isin(
        {"pass", "passed", "approved"}
    )
    coordinates = frame[["patient_x_mm", "patient_y_mm", "patient_z_mm"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete = approved & np.isfinite(coordinates.to_numpy()).all(axis=1)
    missing = int((~complete).sum())
    return missing == 0, f"{expected_rows - missing}/{expected_rows} landmarks complete"


def first_dicom_for_manifest(competition_root: Path, manifest_path: Path) -> Path:
    manifest = pd.read_csv(manifest_path, dtype={"StudyInstanceUID": str})
    study_uid = str(manifest.StudyInstanceUID.iloc[0])
    candidates = sorted((competition_root / "train_series" / study_uid).glob("*/*.dcm"))
    if not candidates:
        raise FileNotFoundError(f"no DICOM found for pilot study {study_uid}")
    return candidates[0]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", default="../data-root")
    parser.add_argument("--work-dir", default="roi_work/pilot")
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open the annotation window; fail if annotations are incomplete.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    competition_root = Path(arguments.competition_root).resolve()
    work_dir = Path(arguments.work_dir).resolve()
    manifest_path = work_dir / "pilot_manifest.csv"
    annotations_path = work_dir / "pilot_annotations.csv"
    priors_path = work_dir / "pilot_priors.json"
    smoke_path = work_dir / "roi_smoke.png"
    cache_dir = work_dir / "cache"

    if not manifest_path.exists() or not annotations_path.exists():
        log("creating deterministic 20-study pilot selection")
        command_select_pilot(SimpleNamespace(
            competition_root=str(competition_root),
            output_dir=str(work_dir),
        ))
    else:
        log(f"resuming existing pilot in {work_dir}")

    complete, status = annotations_complete(annotations_path)
    log(status)
    if not complete and not arguments.no_gui:
        ROIAnnotator(
            competition_root=str(competition_root),
            manifest_csv=str(manifest_path),
            annotations_csv=str(annotations_path),
            annotator=arguments.annotator,
        ).show()
        complete, status = annotations_complete(annotations_path)
        log(status)
    if not complete:
        raise RuntimeError(
            "pilot annotations are incomplete; rerun this command to resume"
        )

    log("fitting leave-one-study-out pilot priors")
    command_fit_priors(SimpleNamespace(
        phase="pilot",
        annotations=str(annotations_path),
        manifest=None,
        output=str(priors_path),
        competition_root=str(competition_root),
    ))

    log("running real-pixel deterministic smoke render")
    command_smoke(SimpleNamespace(
        dicom_path=str(first_dicom_for_manifest(competition_root, manifest_path)),
        priors=str(priors_path),
        competition_root=str(competition_root),
        output=str(smoke_path),
    ))

    log("building the complete pilot as one local shard")
    command_build_pilot(SimpleNamespace(
        manifest=str(manifest_path),
        priors=str(priors_path),
        competition_root=str(competition_root),
        output_dir=str(cache_dir),
        workers=arguments.workers,
    ))
    log(f"local pilot complete: cache={cache_dir} smoke={smoke_path}")


if __name__ == "__main__":
    main()
