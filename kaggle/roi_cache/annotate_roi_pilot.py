"""Resumable local desktop annotator for Arm C landmark cohorts.

Run from the repository root::

    python kaggle/roi_cache/annotate_roi_pilot.py \
        --competition-root ../data-root \
        --manifest roi_work/pilot/pilot_manifest.csv \
        --annotations roi_work/pilot/pilot_annotations.csv \
        --annotator your-name

The Matplotlib window displays one landmark task at a time. Select a slice,
click the landmark, inspect the red round-trip marker, and press ``Save + next``.
Every save atomically replaces the CSV, so closing and rerunning resumes at the
first incomplete task. No notebook, Plotly, or ipywidgets runtime is required.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_roi_cache import _read_pixels, read_series_geometry  # noqa: E402
from roi_geometry import LANDMARKS, SeriesGeometry, pick_series  # noqa: E402


REFERENCE_SLOT = {
    "medial_joint_line": (("Coronal", 0), ("Coronal", 1)),
    "lateral_joint_line": (("Coronal", 0), ("Coronal", 1)),
    "patellofemoral_center": (("Axial", 1), ("Axial", 0)),
    "suprapatellar_recess": (("Sagittal", 1),),
}


def _display_pixels(path: Path) -> np.ndarray:
    image = _read_pixels(path)
    low, high = np.percentile(image, [2.0, 98.0])
    return np.clip((image - low) / max(float(high - low), 1e-6), 0.0, 1.0)


class ROIAnnotator:
    """Interactive local landmark collector backed by an atomic CSV."""

    def __init__(
        self,
        competition_root: str,
        manifest_csv: str,
        annotations_csv: str,
        annotator: str,
    ):
        self.competition_root = Path(competition_root)
        self.manifest = pd.read_csv(manifest_csv, dtype={"StudyInstanceUID": str})
        self.annotations_path = Path(annotations_csv)
        self.annotations = pd.read_csv(
            self.annotations_path,
            dtype={"StudyInstanceUID": str},
        )
        self.series_frame = pd.read_csv(
            self.competition_root / "train_series.csv",
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        self.by_study = {
            key: group.to_dict("records")
            for key, group in self.series_frame.groupby("StudyInstanceUID", sort=False)
        }
        self.annotator = annotator
        self.study_uids = [str(study) for study in self.manifest.StudyInstanceUID]
        self.tasks = [
            (str(study), landmark)
            for study in self.study_uids
            for landmark in LANDMARKS
        ]
        if not self.tasks:
            raise ValueError("annotation manifest contains no studies")
        approved = self.annotations.qc_status.astype(str).str.lower().isin(
            {"pass", "passed", "approved"}
        )
        completed = set(zip(
            self.annotations.loc[approved, "StudyInstanceUID"].astype(str),
            self.annotations.loc[approved, "landmark"].astype(str),
        ))
        self.task_index = next(
            (index for index, task in enumerate(self.tasks) if task not in completed),
            len(self.tasks) - 1,
        )
        self.series: SeriesGeometry | None = None
        self.clicked: tuple[float, float, np.ndarray] | None = None
        self._changing_slider = False

        self.figure, self.image_axis = plt.subplots(figsize=(10, 9))
        self.figure.subplots_adjust(bottom=0.22, top=0.88)
        self.image_artist = self.image_axis.imshow(
            np.zeros((2, 2)),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        self.marker, = self.image_axis.plot(
            [],
            [],
            "r+",
            markersize=16,
            markeredgewidth=2.5,
        )
        self.image_axis.axis("off")
        self.status_artist = self.figure.text(0.02, 0.02, "", fontsize=9)

        slider_axis = self.figure.add_axes((0.16, 0.145, 0.70, 0.035))
        self.slider = Slider(slider_axis, "Slice", 0, 1, valinit=0, valstep=1)
        previous_axis = self.figure.add_axes((0.08, 0.075, 0.18, 0.045))
        save_axis = self.figure.add_axes((0.31, 0.075, 0.25, 0.045))
        next_axis = self.figure.add_axes((0.61, 0.075, 0.18, 0.045))
        self.previous_button = Button(previous_axis, "Previous task")
        self.save_button = Button(save_axis, "Save + next")
        self.next_button = Button(next_axis, "Next task")

        self.slider.on_changed(self._on_slice_change)
        self.previous_button.on_clicked(lambda _: self._move(-1))
        self.save_button.on_clicked(self._save_and_next)
        self.next_button.on_clicked(lambda _: self._move(1))
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self._load_task()

    def _task_rows(self, study_uid: str) -> list[dict]:
        return self.by_study.get(study_uid, [])

    def _reference_row(self, rows: list[dict], landmark: str) -> dict:
        for plane, fluid in REFERENCE_SLOT[landmark]:
            row = pick_series(rows, plane, fluid, allow_fluid_fallback=False)
            if row is not None:
                return row
        raise ValueError(f"no valid reference series for {landmark}")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_artist.set_text(text)
        self.status_artist.set_color("crimson" if error else "black")
        self.figure.canvas.draw_idle()

    def _progress_title(self, study_uid: str, landmark: str) -> str:
        patient_number = self.study_uids.index(study_uid) + 1
        landmark_number = LANDMARKS.index(landmark) + 1
        patients_remaining = len(self.study_uids) - patient_number
        return (
            f"Patient {patient_number}/{len(self.study_uids)} "
            f"({patients_remaining} remaining)  |  "
            f"Landmark {landmark_number}/{len(LANDMARKS)}: {landmark}"
        )

    def _load_task(self) -> None:
        study_uid, landmark = self.tasks[self.task_index]
        self.clicked = None
        self.marker.set_data([], [])
        try:
            row = self._reference_row(self._task_rows(study_uid), landmark)
            self.series = read_series_geometry(
                study_uid,
                row,
                self.competition_root / "train_series",
            )
            middle = len(self.series.slices) // 2
            self._changing_slider = True
            self.slider.valmin = 0
            self.slider.valmax = len(self.series.slices) - 1
            self.slider.ax.set_xlim(0, max(len(self.series.slices) - 1, 1))
            self.slider.set_val(middle)
            self._changing_slider = False
            self.figure.suptitle(
                f"{self._progress_title(study_uid, landmark)}\n"
                f"Study {study_uid}\n"
                f"{self.series.plane}, fluid={self.series.fluid_sensitive}, "
                f"series {self.series.series_uid}",
                fontsize=10,
            )
            self._set_status(
                "Choose a slice and click the landmark. Keys: ←/→ slice, "
                "A/D task, Enter save."
            )
            self._render()
        except Exception as error:
            self.series = None
            self.figure.suptitle(
                f"{self._progress_title(study_uid, landmark)}\n"
                f"Study {study_uid}"
            )
            self._set_status(
                f"Reference view failed: {type(error).__name__}: {error}",
                error=True,
            )

    def _render(self) -> None:
        if self.series is None:
            return
        index = int(round(self.slider.val))
        item = self.series.slices[index]
        image = _display_pixels(item.path)
        self.image_artist.set_data(image)
        self.image_artist.set_extent(
            (-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5)
        )
        self.image_axis.set_xlim(-0.5, image.shape[1] - 0.5)
        self.image_axis.set_ylim(image.shape[0] - 0.5, -0.5)
        self.image_axis.set_aspect("equal")
        self.image_axis.set_title(f"Slice {index + 1}/{len(self.series.slices)} | SOP {item.sop_uid}")
        self.marker.set_data([], [])
        self.figure.canvas.draw_idle()

    def _on_slice_change(self, _: float) -> None:
        if self._changing_slider:
            return
        self.clicked = None
        self._render()

    def _on_click(self, event) -> None:
        if (
            self.series is None
            or event.inaxes is not self.image_axis
            or event.xdata is None
            or event.ydata is None
        ):
            return
        column, row = float(event.xdata), float(event.ydata)
        item = self.series.slices[int(round(self.slider.val))]
        if not (0 <= row < item.rows and 0 <= column < item.columns):
            self._set_status("Click inside the image.", error=True)
            return
        point = item.pixel_to_patient(row, column)
        roundtrip_row, roundtrip_column = item.patient_to_pixel(point)
        error = float(np.hypot(roundtrip_row - row, roundtrip_column - column))
        if error > 1e-5:
            self.clicked = None
            self._set_status(
                f"Round-trip error {error:.6g} px; selection was rejected.",
                error=True,
            )
            return
        self.clicked = (row, column, point)
        self.marker.set_data([roundtrip_column], [roundtrip_row])
        self._set_status(
            f"Selected row={row:.2f}, col={column:.2f}; patient="
            f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) mm; "
            f"round-trip={error:.3g} px. Press Enter or Save + next."
        )

    def _save_and_next(self, _) -> None:
        if self.series is None or self.clicked is None:
            self._set_status("Click a landmark before saving.", error=True)
            return
        study_uid, landmark = self.tasks[self.task_index]
        row, column, point = self.clicked
        item = self.series.slices[int(round(self.slider.val))]
        match = (
            (self.annotations.StudyInstanceUID.astype(str) == study_uid)
            & (self.annotations.landmark.astype(str) == landmark)
        )
        if int(match.sum()) != 1:
            raise ValueError(
                f"annotation template must contain one row for {study_uid}/{landmark}"
            )
        index = self.annotations.index[match][0]
        side = str(self.manifest.loc[
            self.manifest.StudyInstanceUID.astype(str) == study_uid,
            "laterality",
        ].iloc[0])
        values = {
            "SeriesInstanceUID": self.series.series_uid,
            "SOPInstanceUID": item.sop_uid,
            "row_px": row,
            "column_px": column,
            "patient_x_mm": point[0],
            "patient_y_mm": point[1],
            "patient_z_mm": point[2],
            "laterality": side,
            "annotator": self.annotator,
            "qc_status": "pass",
        }
        for key, value in values.items():
            self.annotations.loc[index, key] = value
        temporary = self.annotations_path.with_suffix(
            self.annotations_path.suffix + ".tmp"
        )
        self.annotations.to_csv(temporary, index=False)
        os.replace(temporary, self.annotations_path)
        if self.task_index == len(self.tasks) - 1:
            self._set_status(
                f"All {len(self.tasks)} tasks are saved. You may close the window."
            )
            return
        self._move(1)

    def _move(self, amount: int) -> None:
        self.task_index = max(
            0,
            min(len(self.tasks) - 1, self.task_index + amount),
        )
        self._load_task()

    def _on_key(self, event) -> None:
        if event.key in {"left", "right"} and self.series is not None:
            delta = -1 if event.key == "left" else 1
            value = max(0, min(len(self.series.slices) - 1, int(self.slider.val) + delta))
            self.slider.set_val(value)
        elif event.key in {"a", "A"}:
            self._move(-1)
        elif event.key in {"d", "D"}:
            self._move(1)
        elif event.key in {"enter", "return"}:
            self._save_and_next(None)

    def show(self) -> None:
        plt.show()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--annotator", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    annotator = ROIAnnotator(
        competition_root=arguments.competition_root,
        manifest_csv=arguments.manifest,
        annotations_csv=arguments.annotations,
        annotator=arguments.annotator,
    )
    annotator.show()


if __name__ == "__main__":
    main()
