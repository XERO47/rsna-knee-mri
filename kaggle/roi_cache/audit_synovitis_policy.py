"""Audit alternative synovitis ROI policies using DICOM geometry only.

This command never reads pixel data and never writes production cache arrays.
It measures whether proposed slice triplets and physical crops are actually
available in the source acquisitions, then emits a deterministic JSON report
and a CSV of difficult studies for human image review.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from build_roi_cache import load_study_geometry
from roi_geometry import (
    centerline_offset_to_patient,
    crop_bounds,
    padding_fraction,
    pick_series,
    study_origin,
    triplet_indices,
)


PADDING_LIMIT = 0.20
POLICIES = {
    "current_100x100": {
        "triplets": "fixed_overlapping",
        "axial_crop_mm": (100.0, 100.0),
        "sagittal_crop_mm": (100.0, 100.0),
    },
    "boundary_aware_100x100": {
        "triplets": "nearest_two",
        "axial_crop_mm": (100.0, 100.0),
        "sagittal_crop_mm": (100.0, 100.0),
    },
    "boundary_aware_sagittal_shift10": {
        "triplets": "nearest_two",
        "axial_crop_mm": (100.0, 100.0),
        "sagittal_crop_mm": (100.0, 100.0),
        "sagittal_inferior_shift_mm": 10.0,
    },
    "boundary_aware_sagittal_shift20": {
        "triplets": "nearest_two",
        "axial_crop_mm": (100.0, 100.0),
        "sagittal_crop_mm": (100.0, 100.0),
        "sagittal_inferior_shift_mm": 20.0,
    },
    "boundary_aware_sagittal_shift25": {
        "triplets": "nearest_two",
        "axial_crop_mm": (100.0, 100.0),
        "sagittal_crop_mm": (100.0, 100.0),
        "sagittal_inferior_shift_mm": 25.0,
    },
    "boundary_aware_sagittal_shift30": {
        "triplets": "nearest_two",
        "axial_crop_mm": (100.0, 100.0),
        "sagittal_crop_mm": (100.0, 100.0),
        "sagittal_inferior_shift_mm": 30.0,
    },
    "boundary_aware_sagittal_100x80": {
        "triplets": "nearest_two",
        "axial_crop_mm": (100.0, 100.0),
        "sagittal_crop_mm": (100.0, 80.0),
    },
    "boundary_aware_sagittal_90x80": {
        "triplets": "nearest_two",
        "axial_crop_mm": (100.0, 100.0),
        "sagittal_crop_mm": (90.0, 80.0),
    },
    "boundary_aware_sagittal_90x70": {
        "triplets": "nearest_two",
        "axial_crop_mm": (100.0, 100.0),
        "sagittal_crop_mm": (90.0, 70.0),
    },
    "boundary_aware_sagittal_80x70": {
        "triplets": "nearest_two",
        "axial_crop_mm": (100.0, 100.0),
        "sagittal_crop_mm": (80.0, 70.0),
    },
}


def nearest_two_triplets(series: Any, center: np.ndarray) -> tuple[tuple[int, int, int] | None, ...]:
    """Return two distinct in-bounds triplets nearest the physical landmark."""
    center_position = float(np.dot(center, series.slices[0].normal))
    candidates = [
        (index - 1, index, index + 1)
        for index in range(1, len(series.slices) - 1)
    ]
    ranked = sorted(
        candidates,
        key=lambda item: (
            abs(series.slices[item[1]].position - center_position),
            item[1],
        ),
    )
    selected: list[tuple[int, int, int] | None] = ranked[:2]
    while len(selected) < 2:
        selected.append(None)
    return tuple(selected)


def _planned_triplets(series: Any, center: np.ndarray, strategy: str):
    if strategy == "nearest_two":
        return nearest_two_triplets(series, center)
    if strategy == "fixed_overlapping":
        return triplet_indices(series.nearest_slice_index(center), len(series.slices))
    raise ValueError(f"unsupported triplet strategy: {strategy}")


def _window_validity(
    series: Any,
    center: np.ndarray,
    plane: str,
    crop_mm: Sequence[float],
    strategy: str,
    inferior_shift_mm: float = 0.0,
) -> tuple[list[bool], list[float | None]]:
    validity: list[bool] = []
    padding: list[float | None] = []
    for indices in _planned_triplets(series, center, strategy):
        if indices is None:
            validity.append(False)
            padding.append(None)
            continue
        values = []
        crop_center = center + np.array([0.0, 0.0, -float(inferior_shift_mm)])
        for index in indices:
            geometry = series.slices[index]
            bounds = crop_bounds(
                geometry,
                crop_center,
                plane,
                (float(crop_mm[0]), float(crop_mm[1])),
                None,
            )
            values.append(padding_fraction(bounds, geometry.rows, geometry.columns))
        maximum = max(values)
        validity.append(maximum <= PADDING_LIMIT + 1e-12)
        padding.append(maximum)
    return validity, padding


_CONTEXT: dict[str, Any] = {}


def _initialize(rows: dict[str, list[dict]], root: str, offset: list[float]) -> None:
    _CONTEXT.update(rows=rows, root=Path(root), offset=np.asarray(offset, dtype=np.float64))


def _audit_study(study_uid: str) -> dict[str, Any]:
    rows = _CONTEXT["rows"].get(study_uid, [])
    geometries, failures = load_study_geometry(study_uid, rows, _CONTEXT["root"])
    result: dict[str, Any] = {
        "StudyInstanceUID": study_uid,
        "geometry_failures": len(failures),
        "policies": {},
    }
    if not geometries:
        result["fatal_error"] = "no_valid_series_geometry"
        return result
    center = centerline_offset_to_patient(_CONTEXT["offset"], study_origin(geometries.values()))
    selected = {}
    for plane in ("Axial", "Sagittal"):
        row = pick_series(rows, plane, 1, allow_fluid_fallback=False)
        selected[plane] = None if row is None else geometries.get(str(row["SeriesInstanceUID"]))
    for name, policy in POLICIES.items():
        valid: list[bool] = []
        padding: list[float | None] = []
        per_plane: dict[str, int] = {}
        for plane in ("Axial", "Sagittal"):
            series = selected[plane]
            if series is None:
                plane_valid = [False, False]
                plane_padding: list[float | None] = [None, None]
            else:
                plane_valid, plane_padding = _window_validity(
                    series,
                    center,
                    plane,
                    policy[f"{plane.lower()}_crop_mm"],
                    policy["triplets"],
                    float(policy.get(f"{plane.lower()}_inferior_shift_mm", 0.0)),
                )
            valid.extend(plane_valid)
            padding.extend(plane_padding)
            per_plane[plane.lower()] = sum(plane_valid)
        result["policies"][name] = {
            "valid_windows": sum(valid),
            "axial_valid_windows": per_plane["axial"],
            "sagittal_valid_windows": per_plane["sagittal"],
            "both_planes": all(value >= 1 for value in per_plane.values()),
            "max_padding": max((value for value in padding if value is not None), default=None),
        }
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    frame = pd.read_csv(arguments.competition_root / "train_series.csv")
    frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
    frame["SeriesInstanceUID"] = frame.SeriesInstanceUID.astype(str)
    rows = {
        key: group.to_dict("records")
        for key, group in frame.groupby("StudyInstanceUID", sort=False)
    }
    ids = np.load(arguments.ids, allow_pickle=True).astype(str).tolist()
    priors = json.loads(arguments.priors.read_text())
    offset = priors["landmarks"]["suprapatellar_recess"]["median_offset_mm"]
    with ProcessPoolExecutor(
        max_workers=arguments.workers,
        initializer=_initialize,
        initargs=(rows, str(arguments.competition_root / "train_series"), offset),
    ) as pool:
        studies = list(pool.map(_audit_study, ids, chunksize=4))

    summary: dict[str, Any] = {}
    for name in POLICIES:
        distribution = Counter(item.get("policies", {}).get(name, {}).get("valid_windows", 0) for item in studies)
        summary[name] = {
            "policy": POLICIES[name],
            "valid_window_distribution": dict(sorted(distribution.items())),
            "at_least_two_valid_windows_rate": float(np.mean([
                item.get("policies", {}).get(name, {}).get("valid_windows", 0) >= 2
                for item in studies
            ])),
            "both_planes_rate": float(np.mean([
                item.get("policies", {}).get(name, {}).get("both_planes", False)
                for item in studies
            ])),
            "four_valid_windows_rate": float(np.mean([
                item.get("policies", {}).get(name, {}).get("valid_windows", 0) == 4
                for item in studies
            ])),
        }
    report = {
        "study_count": len(studies),
        "padding_limit": PADDING_LIMIT,
        "summary": summary,
        "studies": studies,
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    preferred = max(
        POLICIES,
        key=lambda name: (
            summary[name]["both_planes_rate"],
            summary[name]["four_valid_windows_rate"],
            summary[name]["at_least_two_valid_windows_rate"],
        ),
    )
    review = sorted(
        studies,
        key=lambda item: (
            item.get("policies", {}).get(preferred, {}).get("valid_windows", 0),
            item["StudyInstanceUID"],
        ),
    )
    arguments.review_csv.parent.mkdir(parents=True, exist_ok=True)
    with arguments.review_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "StudyInstanceUID",
                "policy",
                "valid_windows",
                "axial_valid_windows",
                "sagittal_valid_windows",
                "both_planes",
                "max_padding",
            ],
        )
        writer.writeheader()
        for item in review[:20]:
            values = item.get("policies", {}).get(preferred, {})
            writer.writerow({
                "StudyInstanceUID": item["StudyInstanceUID"],
                "policy": preferred,
                **values,
            })
    print(json.dumps({key: report[key] for key in ("study_count", "padding_limit", "summary")}, indent=2, sort_keys=True))
    print(f"recommended geometry candidate: {preferred}")
    print(f"review list: {arguments.review_csv}")


if __name__ == "__main__":
    main()
