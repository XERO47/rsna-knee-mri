"""Scientific service layer: reuses existing DICOM geometry/pixel primitives.

This module performs no independent DICOM math. It imports
``read_series_geometry``, ``_read_pixels``/``_display_pixels``,
``pick_series``, and ``SliceGeometry``/``SeriesGeometry`` from the existing
``roi_geometry``/``build_roi_cache``/``annotate_roi_pilot`` modules and wires
them to HTTP-facing task resolution, bounded caching, and server-side
pixel-to-patient validation.
"""

from __future__ import annotations

import io
import math
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

_ROI_CACHE_DIR = Path(__file__).resolve().parents[1]
if str(_ROI_CACHE_DIR) not in sys.path:
    sys.path.insert(0, str(_ROI_CACHE_DIR))

import annotate_roi_pilot  # noqa: E402  (reference-series reuse, read-only import)
import build_roi_cache  # noqa: E402  (geometry/pixel reuse, read-only import)
import roi_geometry  # noqa: E402  (geometry/pixel reuse, read-only import)

from .config import ROUND_TRIP_TOLERANCE_PX

LANDMARKS = roi_geometry.LANDMARKS
REFERENCE_SLOT = annotate_roi_pilot.REFERENCE_SLOT


class ReferenceSeriesUnavailable(Exception):
    """No permitted reference series resolves for a landmark on this study."""


class DataRootUnavailable(Exception):
    """The competition root / data volume is not reachable right now."""


class RoundTripValidationFailed(ValueError):
    pass


class OutOfBounds(ValueError):
    pass


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def medial_image_side(task: "ResolvedTask", laterality: str | None) -> str | None:
    """Return the displayed left/right edge that is anatomically medial.

    The browser renders source columns left-to-right without a flip.  For a
    coronal joint-line image, compare the patient-X coordinate at its two
    horizontal edges with the DICOM laterality convention: medial is -X for a
    left knee and +X for a right knee.  Avoid a misleading label if a heavily
    oblique acquisition maps patient X mostly vertically rather than across
    the image.
    """
    if laterality not in {"L", "R"} or task.series.plane != "Coronal":
        return None
    item = task.series.middle
    row = (item.rows - 1) / 2.0
    left_x = float(item.pixel_to_patient(row, 0.0)[0])
    right_x = float(item.pixel_to_patient(row, float(item.columns - 1))[0])
    top_x = float(item.pixel_to_patient(0.0, (item.columns - 1) / 2.0)[0])
    bottom_x = float(item.pixel_to_patient(float(item.rows - 1), (item.columns - 1) / 2.0)[0])
    horizontal_change = right_x - left_x
    vertical_change = bottom_x - top_x
    if abs(horizontal_change) <= 1e-6 or abs(horizontal_change) < abs(vertical_change):
        return None
    medial_direction = -1.0 if laterality == "L" else 1.0
    return "right" if horizontal_change * medial_direction > 0 else "left"


class _BoundedByteCache:
    """LRU cache bounded by both item count and total byte size."""

    def __init__(self, max_items: int, max_bytes: int):
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._data: "OrderedDict[Any, bytes]" = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()

    def get(self, key: Any) -> bytes | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: Any, value: bytes) -> None:
        with self._lock:
            if key in self._data:
                self._total_bytes -= len(self._data.pop(key))
            self._data[key] = value
            self._total_bytes += len(value)
            self._data.move_to_end(key)
            while self._data and (
                len(self._data) > self._max_items or self._total_bytes > self._max_bytes
            ):
                _, evicted = self._data.popitem(last=False)
                self._total_bytes -= len(evicted)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes


@dataclass(frozen=True)
class ResolvedTask:
    study_uid: str
    landmark: str
    series: "roi_geometry.SeriesGeometry"
    series_row: dict[str, Any]


class ScientificService:
    """Owns the read-only competition data handles and bounded caches."""

    def __init__(
        self,
        competition_root: Path,
        slice_cache_items: int = 64,
        slice_cache_bytes: int = 200 * 1024 * 1024,
        max_concurrent_decodes: int = 2,
        series_geometry_cache_size: int = 32,
    ):
        self.competition_root = Path(competition_root)
        self.series_root = self.competition_root / "train_series"
        self._series_frame: pd.DataFrame | None = None
        self._by_study: dict[str, list[dict[str, Any]]] = {}
        self._frame_lock = threading.Lock()
        self._series_geometry_cache: "OrderedDict[tuple[str, str], ResolvedTask]" = OrderedDict()
        self._series_geometry_cache_size = series_geometry_cache_size
        self._series_geometry_lock = threading.Lock()
        self._png_cache = _BoundedByteCache(slice_cache_items, slice_cache_bytes)
        self._decode_semaphore = threading.Semaphore(max_concurrent_decodes)

    def check_data_root_available(self) -> None:
        if not (self.competition_root / "train_series.csv").exists() or not (
            self.competition_root / "train_series"
        ).is_dir():
            raise DataRootUnavailable(
                f"competition root is not reachable: {self.competition_root}"
            )

    def _ensure_series_frame(self) -> None:
        if self._series_frame is not None:
            return
        with self._frame_lock:
            if self._series_frame is not None:
                return
            self.check_data_root_available()
            frame = pd.read_csv(self.competition_root / "train_series.csv")
            frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
            frame["SeriesInstanceUID"] = frame.SeriesInstanceUID.astype(str)
            self._series_frame = frame
            self._by_study = {
                key: group.to_dict("records")
                for key, group in frame.groupby("StudyInstanceUID", sort=False)
            }

    def rows_for_study(self, study_uid: str) -> list[dict[str, Any]]:
        self._ensure_series_frame()
        return self._by_study.get(study_uid, [])

    def resolve_task(self, study_uid: str, landmark: str) -> ResolvedTask:
        if landmark not in LANDMARKS:
            raise ValueError(f"unknown landmark: {landmark}")
        cache_key = (study_uid, landmark)
        with self._series_geometry_lock:
            cached = self._series_geometry_cache.get(cache_key)
            if cached is not None:
                self._series_geometry_cache.move_to_end(cache_key)
                return cached
        rows = self.rows_for_study(study_uid)
        resolved_row = None
        for plane, fluid in REFERENCE_SLOT[landmark]:
            candidate = roi_geometry.pick_series(rows, plane, fluid, allow_fluid_fallback=False)
            if candidate is not None:
                resolved_row = candidate
                break
        if resolved_row is None:
            raise ReferenceSeriesUnavailable(
                f"no permitted reference series for {study_uid}/{landmark}"
            )
        series = build_roi_cache.read_series_geometry(study_uid, resolved_row, self.series_root)
        task = ResolvedTask(study_uid=study_uid, landmark=landmark, series=series, series_row=resolved_row)
        with self._series_geometry_lock:
            self._series_geometry_cache[cache_key] = task
            self._series_geometry_cache.move_to_end(cache_key)
            while len(self._series_geometry_cache) > self._series_geometry_cache_size:
                self._series_geometry_cache.popitem(last=False)
        return task

    def render_slice_png(self, task: ResolvedTask, slice_index: int) -> bytes:
        if not (0 <= slice_index < len(task.series.slices)):
            raise OutOfBounds(f"slice index {slice_index} out of bounds")
        item = task.series.slices[slice_index]
        cache_key = (item.sop_uid, "v1-percentile-2-98")
        cached = self._png_cache.get(cache_key)
        if cached is not None:
            return cached
        with self._decode_semaphore:
            display = annotate_roi_pilot._display_pixels(item.path)
        uint8 = np.rint(np.clip(display, 0.0, 1.0) * 255.0).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(uint8, mode="L").save(buffer, format="PNG", optimize=False)
        payload = buffer.getvalue()
        self._png_cache.put(cache_key, payload)
        return payload

    def cache_stats(self) -> dict[str, int]:
        return {
            "slice_cache_items": len(self._png_cache),
            "slice_cache_bytes": self._png_cache.total_bytes,
            "series_geometry_cache_items": len(self._series_geometry_cache),
        }

    def compute_and_validate_point(
        self,
        task: ResolvedTask,
        sop_uid: str,
        row_px: float,
        column_px: float,
    ) -> tuple[np.ndarray, str, int]:
        """Return ``(patient_xyz, resolved_sop_uid, slice_index)`` or raise.

        Validates that ``sop_uid`` is one of the resolved series' own slices
        (never an arbitrary client-supplied identifier), that the pixel is
        finite and in bounds, and that the pixel -> patient -> pixel round
        trip is within tolerance, mirroring the desktop annotator's
        ``_on_click`` acceptance gate.
        """
        if not _finite(row_px, column_px):
            raise OutOfBounds("row/column must be finite")
        slice_index = next(
            (index for index, item in enumerate(task.series.slices) if item.sop_uid == sop_uid),
            None,
        )
        if slice_index is None:
            raise OutOfBounds(f"SOP UID {sop_uid!r} is not part of the resolved series")
        item = task.series.slices[slice_index]
        if not (0 <= row_px < item.rows and 0 <= column_px < item.columns):
            raise OutOfBounds("row/column is outside the source image bounds")
        point = item.pixel_to_patient(row_px, column_px)
        round_trip_row, round_trip_column = item.patient_to_pixel(point)
        error = math.hypot(round_trip_row - row_px, round_trip_column - column_px)
        if not math.isfinite(error) or error > ROUND_TRIP_TOLERANCE_PX:
            raise RoundTripValidationFailed(
                f"pixel/patient round-trip error {error:.6g}px exceeds tolerance"
            )
        return point, item.sop_uid, slice_index
