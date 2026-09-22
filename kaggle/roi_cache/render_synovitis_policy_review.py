"""Render sagittal synovitis crop candidates for radiologist review.

The sheets compare the original symmetric crop with 20, 25, and 30 mm
inferior shifts while keeping the physical crop at 100 x 100 mm.  They are
diagnostic artifacts only and do not modify the production cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from audit_synovitis_policy import nearest_two_triplets
from build_roi_cache import (
    _read_pixels,
    _scale_uint8,
    intensity_limits,
    load_study_geometry,
)
from roi_geometry import (
    centerline_offset_to_patient,
    crop_bounds,
    crop_with_padding,
    padding_fraction,
    pick_series,
    study_origin,
)


SHIFTS = (0.0, 20.0, 25.0, 30.0)
COLORS = ("#ef4444", "#f59e0b", "#38bdf8", "#22c55e")


def _crosshair(axis, row: float, column: float, color: str) -> None:
    axis.plot(column, row, marker="+", color=color, markersize=14, markeredgewidth=2)


def _select_examples(report: dict, count: int) -> dict[str, list[str]]:
    categories = {"routine": [], "rescued_by_shift": [], "shift30_invalid": []}
    for item in report["studies"]:
        uid = item["StudyInstanceUID"]
        shift20 = item["policies"]["boundary_aware_sagittal_shift20"]["sagittal_valid_windows"]
        shift30 = item["policies"]["boundary_aware_sagittal_shift30"]["sagittal_valid_windows"]
        if shift20 == 2:
            categories["routine"].append(uid)
        elif shift30 == 2:
            categories["rescued_by_shift"].append(uid)
        else:
            categories["shift30_invalid"].append(uid)
    return {key: value[:count] for key, value in categories.items()}


def _render_sheet(
    category: str,
    study_ids: list[str],
    rows_by_study: dict[str, list[dict]],
    series_root: Path,
    offset: np.ndarray,
    output: Path,
) -> None:
    figure, axes = plt.subplots(len(study_ids), 5, figsize=(18, 4.4 * len(study_ids)), squeeze=False)
    for row_index, study_uid in enumerate(study_ids):
        rows = rows_by_study.get(study_uid, [])
        geometries, _ = load_study_geometry(study_uid, rows, series_root)
        center = centerline_offset_to_patient(offset, study_origin(geometries.values()))
        selected = pick_series(rows, "Sagittal", 1, allow_fluid_fallback=False)
        if selected is None or str(selected["SeriesInstanceUID"]) not in geometries:
            for axis in axes[row_index]:
                axis.text(0.5, 0.5, "No sagittal fluid-sensitive series", ha="center", va="center")
                axis.axis("off")
            continue
        series = geometries[str(selected["SeriesInstanceUID"])]
        indices = nearest_two_triplets(series, center)[0]
        if indices is None:
            for axis in axes[row_index]:
                axis.text(0.5, 0.5, "Fewer than three physical slices", ha="center", va="center")
                axis.axis("off")
            continue
        geometry = series.slices[indices[1]]
        low, high = intensity_limits(series)
        image = _scale_uint8(_read_pixels(geometry.path), low, high)
        landmark_row, landmark_column = geometry.patient_to_pixel(center)

        full_axis = axes[row_index, 0]
        full_axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        _crosshair(full_axis, landmark_row, landmark_column, "white")
        for shift, color in zip(SHIFTS, COLORS):
            crop_center = center + np.array([0.0, 0.0, -shift])
            bounds = crop_bounds(geometry, crop_center, "Sagittal", (100.0, 100.0), None)
            row0, row1, column0, column1 = bounds
            full_axis.add_patch(patches.Rectangle(
                (column0, row0), column1 - column0, row1 - row0,
                linewidth=1.5, edgecolor=color, facecolor="none",
            ))
        full_axis.set_title(f"{category}\n{study_uid[-12:]}\nwhite += predicted landmark")
        full_axis.axis("off")

        for column_index, (shift, color) in enumerate(zip(SHIFTS, COLORS), start=1):
            crop_center = center + np.array([0.0, 0.0, -shift])
            bounds = crop_bounds(geometry, crop_center, "Sagittal", (100.0, 100.0), None)
            pad = padding_fraction(bounds, geometry.rows, geometry.columns)
            crop = crop_with_padding(image, bounds)
            resized = cv2.resize(crop, (336, 336), interpolation=cv2.INTER_LINEAR)
            axis = axes[row_index, column_index]
            axis.imshow(resized, cmap="gray", vmin=0, vmax=255)
            row0, row1, column0, column1 = bounds
            marker_row = (landmark_row - row0) * 336.0 / (row1 - row0)
            marker_column = (landmark_column - column0) * 336.0 / (column1 - column0)
            _crosshair(axis, marker_row, marker_column, color)
            axis.set_title(f"inferior shift {int(shift)} mm\npadding {pad:.1%}")
            axis.axis("off")
    figure.suptitle(
        "Synovitis sagittal policy review — Is the suprapatellar recess and useful surrounding anatomy retained?",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-category", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    frame = pd.read_csv(arguments.competition_root / "train_series.csv")
    frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
    frame["SeriesInstanceUID"] = frame.SeriesInstanceUID.astype(str)
    rows = {key: group.to_dict("records") for key, group in frame.groupby("StudyInstanceUID", sort=False)}
    priors = json.loads(arguments.priors.read_text())
    offset = np.asarray(priors["landmarks"]["suprapatellar_recess"]["median_offset_mm"])
    report = json.loads(arguments.audit_report.read_text())
    selected = _select_examples(report, arguments.examples_per_category)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for category, study_ids in selected.items():
        output = arguments.output_dir / f"synovitis_{category}.png"
        _render_sheet(
            category,
            study_ids,
            rows,
            arguments.competition_root / "train_series",
            offset,
            output,
        )
        print(output)


if __name__ == "__main__":
    main()
