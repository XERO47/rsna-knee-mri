"""Kaggle notebook UI for the Arm C landmark pilot.

Upload this file together with ``roi_geometry.py`` and run in a notebook::

    from annotate_roi_pilot import ROIAnnotator
    ui = ROIAnnotator(
        competition_root="/kaggle/input/competitions/rsna-knee-abnormality-detection",
        manifest_csv="/kaggle/working/roi_pilot/pilot_manifest.csv",
        annotations_csv="/kaggle/working/roi_pilot/pilot_annotations.csv",
        annotator="your-name",
    )
    ui.show()

Click the landmark on the displayed slice, confirm the reprojected marker is
correct, then press **Save + next**. Every save atomically rewrites the CSV.
The interface uses Plotly FigureWidget click events and ordinary ipywidgets,
both included in Kaggle notebook images.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydicom
from IPython.display import display
from pydicom.pixel_data_handlers.util import apply_modality_lut

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_geometry import LANDMARKS, SeriesGeometry, pick_series  # noqa: E402
from build_roi_cache import read_series_geometry  # noqa: E402


REFERENCE_SLOT = {
    "medial_joint_line": (("Coronal", 0), ("Coronal", 1)),
    "lateral_joint_line": (("Coronal", 0), ("Coronal", 1)),
    "patellofemoral_center": (("Axial", 1), ("Axial", 0)),
    "suprapatellar_recess": (("Sagittal", 1),),
}


def _display_pixels(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(path)
    image = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = image.max() - image
    low, high = np.percentile(image, [2.0, 98.0])
    return np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)


class ROIAnnotator:
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
        self.annotations = pd.read_csv(self.annotations_path, dtype={"StudyInstanceUID": str})
        self.series_frame = pd.read_csv(
            self.competition_root / "train_series.csv",
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        self.by_study = {
            key: group.to_dict("records")
            for key, group in self.series_frame.groupby("StudyInstanceUID", sort=False)
        }
        self.annotator = annotator
        self.tasks = [
            (str(study), landmark)
            for study in self.manifest.StudyInstanceUID
            for landmark in LANDMARKS
        ]
        self.task_index = 0
        self.series: SeriesGeometry | None = None
        self.clicked: tuple[float, float, np.ndarray] | None = None

        self.title = widgets.HTML()
        self.status = widgets.HTML()
        self.slider = widgets.IntSlider(description="Slice", continuous_update=False)
        self.previous = widgets.Button(description="Previous")
        self.save_next = widgets.Button(description="Save + next", button_style="success")
        self.skip = widgets.Button(description="Next without saving")
        self.figure = go.FigureWidget(
            data=[
                go.Heatmap(z=np.zeros((2, 2)), colorscale="Gray", showscale=False),
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    marker={"color": "red", "size": 12, "symbol": "cross"},
                ),
            ],
            layout={
                "height": 650,
                "margin": {"l": 20, "r": 20, "t": 35, "b": 20},
                "xaxis": {"scaleanchor": "y"},
                "yaxis": {"autorange": "reversed"},
                "clickmode": "event+select",
            },
        )
        self.figure.data[0].on_click(self._on_click)
        self.slider.observe(self._on_slice_change, names="value")
        self.previous.on_click(lambda _: self._move(-1))
        self.skip.on_click(lambda _: self._move(1))
        self.save_next.on_click(self._save_and_next)
        self._load_task()

    def _task_rows(self, study_uid: str) -> list[dict]:
        return self.by_study.get(study_uid, [])

    def _reference_row(self, rows: list[dict], landmark: str) -> dict:
        for plane, fluid in REFERENCE_SLOT[landmark]:
            row = pick_series(rows, plane, fluid, allow_fluid_fallback=False)
            if row is not None:
                return row
        raise ValueError(f"no valid reference series for {landmark}")

    def _load_task(self) -> None:
        study_uid, landmark = self.tasks[self.task_index]
        self.clicked = None
        try:
            row = self._reference_row(self._task_rows(study_uid), landmark)
            self.series = read_series_geometry(
                study_uid,
                row,
                self.competition_root / "train_series",
            )
            self.slider.max = len(self.series.slices) - 1
            self.slider.value = len(self.series.slices) // 2
            self.title.value = (
                f"<b>{self.task_index + 1}/{len(self.tasks)}</b> &nbsp; "
                f"Study: <code>{study_uid}</code> &nbsp; Landmark: <b>{landmark}</b><br>"
                f"Reference: {self.series.plane}, fluid={self.series.fluid_sensitive}, "
                f"series <code>{self.series.series_uid}</code>"
            )
            self.status.value = (
                "Click one anatomical landmark; the red cross is the physical "
                "round-trip projection."
            )
            self._render()
        except Exception as error:
            self.series = None
            self.title.value = f"<b>{study_uid} / {landmark}</b>"
            self.status.value = (
                "<span style='color:red'>Reference view failed: "
                f"{type(error).__name__}: {error}</span>"
            )

    def _render(self) -> None:
        if self.series is None:
            return
        item = self.series.slices[self.slider.value]
        image = _display_pixels(item.path)
        with self.figure.batch_update():
            self.figure.data[0].z = image
            self.figure.data[1].x = []
            self.figure.data[1].y = []
            self.figure.layout.title = f"SOP {item.sop_uid}"

    def _on_slice_change(self, change: dict) -> None:
        if change["name"] == "value":
            self.clicked = None
            self._render()

    def _on_click(self, trace, points, selector) -> None:
        if self.series is None or not points.xs or not points.ys:
            return
        column, row = float(points.xs[0]), float(points.ys[0])
        item = self.series.slices[self.slider.value]
        point = item.pixel_to_patient(row, column)
        roundtrip_row, roundtrip_column = item.patient_to_pixel(point)
        error = float(np.hypot(roundtrip_row - row, roundtrip_column - column))
        if error > 1e-5:
            self.clicked = None
            self.status.value = (
                f"<span style='color:red'>Round-trip error {error:.6g} px; "
                "do not save.</span>"
            )
            return
        self.clicked = (row, column, point)
        self.figure.data[1].x = [roundtrip_column]
        self.figure.data[1].y = [roundtrip_row]
        self.status.value = (
            f"Selected row={row:.2f}, col={column:.2f}; patient="
            f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) mm; "
            f"round-trip error={error:.3g} px."
        )

    def _save_and_next(self, _) -> None:
        if self.series is None or self.clicked is None:
            self.status.value = "<span style='color:red'>Click a landmark before saving.</span>"
            return
        study_uid, landmark = self.tasks[self.task_index]
        row, column, point = self.clicked
        item = self.series.slices[self.slider.value]
        match = (
            (self.annotations.StudyInstanceUID.astype(str) == study_uid)
            & (self.annotations.landmark == landmark)
        )
        if match.sum() != 1:
            raise ValueError(f"annotation template must contain one row for {study_uid}/{landmark}")
        index = self.annotations.index[match][0]
        side = str(self.manifest.loc[
            self.manifest.StudyInstanceUID.astype(str) == study_uid, "laterality"
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
        temporary = self.annotations_path.with_suffix(self.annotations_path.suffix + ".tmp")
        self.annotations.to_csv(temporary, index=False)
        os.replace(temporary, self.annotations_path)
        self._move(1)

    def _move(self, amount: int) -> None:
        self.task_index = max(0, min(len(self.tasks) - 1, self.task_index + amount))
        self._load_task()

    def show(self) -> None:
        display(widgets.VBox([
            self.title,
            self.slider,
            self.figure,
            self.status,
            widgets.HBox([self.previous, self.save_next, self.skip]),
        ]))
