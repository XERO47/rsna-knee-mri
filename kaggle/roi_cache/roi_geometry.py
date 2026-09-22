"""Deterministic DICOM geometry primitives for the Arm C ROI cache.

All coordinates are DICOM patient coordinates (LPS, millimetres). Array
indices are always expressed as ``(row, column)``. This module deliberately
contains no training or pathology-label logic, so the exact same projection
code can be reused for a future test-set cache.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


POSITION_TOLERANCE_MM = 0.1
ORIENTATION_TOLERANCE_DEGREES = 1.0

LANDMARKS = (
    "medial_joint_line",
    "lateral_joint_line",
    "patellofemoral_center",
    "suprapatellar_recess",
)

TARGET_SPECS = {
    "mcl": {
        "name": "MCL",
        "landmark": "medial_joint_line",
        "crop_mm": (90.0, 120.0),
        "laterality_required": True,
        "allow_fluid_fallback": False,
        "slots": (("Coronal", 1), ("Axial", 1)),
    },
    "lateral_meniscus": {
        "name": "Lateral Meniscus",
        "landmark": "lateral_joint_line",
        "crop_mm": (80.0, 80.0),
        "laterality_required": True,
        "allow_fluid_fallback": False,
        "slots": (("Sagittal", 0), ("Coronal", 0)),
    },
    "lateral_oa": {
        "name": "Lateral OA",
        "landmark": "lateral_joint_line",
        "crop_mm": (85.0, 75.0),
        "laterality_required": True,
        "allow_fluid_fallback": False,
        "slots": (("Coronal", 0), ("Sagittal", 0)),
    },
    "pf_oa": {
        "name": "PF OA",
        "landmark": "patellofemoral_center",
        "crop_mm": (95.0, 85.0),
        "laterality_required": False,
        "allow_fluid_fallback": True,
        "slots": (("Axial", 1), ("Sagittal", 1)),
    },
    "synovitis": {
        "name": "Synovitis",
        "landmark": "suprapatellar_recess",
        "crop_mm": (100.0, 100.0),
        "laterality_required": False,
        "allow_fluid_fallback": False,
        "slots": (("Axial", 1), ("Sagittal", 1)),
        # The clinically approved policy retains the complete 100 x 100 mm
        # field but positions the suprapatellar landmark 25 mm below the
        # sagittal crop centre.  It leaves a 25 mm superior margin for an
        # expanded pouch and adds useful inferior joint context.
        "crop_center_offset_lps_mm_by_plane": {"Sagittal": (0.0, 0.0, -25.0)},
        "side_independent_window_strategy": "nearest_distinct_triplets",
    },
}


def _unit(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("direction vector has zero or non-finite length")
    return value / norm


def _finite_vector(value: object, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class SliceGeometry:
    path: Path
    sop_uid: str
    ipp: np.ndarray
    row_dir: np.ndarray
    col_dir: np.ndarray
    normal: np.ndarray
    pixel_spacing: tuple[float, float]
    rows: int
    columns: int
    position: float

    @classmethod
    def from_header(cls, path: Path, header: object) -> "SliceGeometry":
        iop = _finite_vector(getattr(header, "ImageOrientationPatient"), 6, "IOP")
        ipp = _finite_vector(getattr(header, "ImagePositionPatient"), 3, "IPP")
        spacing = _finite_vector(getattr(header, "PixelSpacing"), 2, "PixelSpacing")
        if np.any(spacing <= 0):
            raise ValueError("PixelSpacing must be positive")
        rows = int(getattr(header, "Rows"))
        columns = int(getattr(header, "Columns"))
        if rows <= 0 or columns <= 0:
            raise ValueError("Rows and Columns must be positive")
        row_dir = _unit(iop[:3])
        col_dir = _unit(iop[3:])
        if abs(float(np.dot(row_dir, col_dir))) > 1e-3:
            raise ValueError("IOP row and column directions are not orthogonal")
        normal = _unit(np.cross(row_dir, col_dir))
        return cls(
            path=Path(path),
            sop_uid=str(getattr(header, "SOPInstanceUID", Path(path).stem)),
            ipp=ipp,
            row_dir=row_dir,
            col_dir=col_dir,
            normal=normal,
            pixel_spacing=(float(spacing[0]), float(spacing[1])),
            rows=rows,
            columns=columns,
            position=float(np.dot(ipp, normal)),
        )

    def pixel_to_patient(self, row: float, column: float) -> np.ndarray:
        row_spacing, column_spacing = self.pixel_spacing
        return (
            self.ipp
            + float(column) * column_spacing * self.row_dir
            + float(row) * row_spacing * self.col_dir
        )

    def patient_to_pixel(self, point: Sequence[float]) -> tuple[float, float]:
        delta = _finite_vector(point, 3, "patient point") - self.ipp
        row_spacing, column_spacing = self.pixel_spacing
        # ImageOrientationPatient vectors should be orthogonal, but real
        # scanner headers are often rounded enough that their dot product is
        # non-zero.  Independent dot-product projections are only an inverse
        # in the perfectly orthogonal case; their error grows with distance
        # from ImagePositionPatient.  Solve the two in-plane basis vectors
        # together instead, which is the stable inverse of pixel_to_patient
        # for both ideal and accepted near-orthogonal headers.
        basis = np.column_stack((
            column_spacing * self.row_dir,
            row_spacing * self.col_dir,
        ))
        coordinates, _residuals, rank, _singular_values = np.linalg.lstsq(
            basis, delta, rcond=None
        )
        if rank != 2 or not np.isfinite(coordinates).all():
            raise ValueError("image orientation does not define an invertible in-plane basis")
        column = float(coordinates[0])
        row = float(coordinates[1])
        return row, column

    def center(self) -> np.ndarray:
        return self.pixel_to_patient((self.rows - 1) / 2, (self.columns - 1) / 2)


@dataclass(frozen=True)
class SeriesGeometry:
    study_uid: str
    series_uid: str
    plane: str
    fluid_sensitive: int
    slices: tuple[SliceGeometry, ...]
    laterality_votes: tuple[str, ...] = ()
    manufacturer: str = ""
    field_strength: float | None = None
    slice_thickness: float | None = None

    @property
    def middle(self) -> SliceGeometry:
        return self.slices[len(self.slices) // 2]

    @property
    def center(self) -> np.ndarray:
        return self.middle.center()

    def nearest_slice_index(self, point: Sequence[float]) -> int:
        value = _finite_vector(point, 3, "patient point")
        distances = [abs(float(np.dot(value - item.ipp, item.normal))) for item in self.slices]
        return int(np.argmin(distances))


def orientation_angle_degrees(a: Sequence[float], b: Sequence[float]) -> float:
    cosine = float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def sort_deduplicate_slices(
    slices: Iterable[SliceGeometry],
    tolerance_mm: float = POSITION_TOLERANCE_MM,
    orientation_tolerance_degrees: float = ORIENTATION_TOLERANCE_DEGREES,
) -> tuple[SliceGeometry, ...]:
    """Sort by physical position and remove positions within ``tolerance_mm``.

    Duplicate selection is deterministic: SOP UID and path break position ties.
    A series is rejected unless at least three unique, consistently oriented
    slices remain.
    """
    candidates = list(slices)
    if not candidates:
        raise ValueError("series contains no geometrically valid slices")
    # Position every slice on one series-level normal.  Using each slice's own
    # (slightly noisy) normal can perturb ordering in otherwise valid oblique
    # acquisitions even when all orientations are within the 1-degree gate.
    reference = min(candidates, key=lambda item: (item.sop_uid, str(item.path)))
    for item in candidates:
        if orientation_angle_degrees(reference.row_dir, item.row_dir) > orientation_tolerance_degrees:
            raise ValueError("row orientation varies by more than 1 degree")
        if orientation_angle_degrees(reference.col_dir, item.col_dir) > orientation_tolerance_degrees:
            raise ValueError("column orientation varies by more than 1 degree")
    projected = [
        replace(item, position=float(np.dot(item.ipp, reference.normal)))
        for item in candidates
    ]
    ordered = sorted(projected, key=lambda item: (item.position, item.sop_uid, str(item.path)))
    unique: list[SliceGeometry] = []
    for item in ordered:
        if not unique or item.position - unique[-1].position > tolerance_mm:
            unique.append(item)
    if len(unique) < 3:
        raise ValueError("series contains fewer than three unique slice positions")
    positions = np.asarray([item.position for item in unique])
    if not np.all(np.diff(positions) > 0):
        raise ValueError("slice positions are not strictly monotonic")
    return tuple(unique)


def normalize_laterality(value: object) -> str | None:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT"}:
        return "L"
    if text in {"R", "RIGHT"}:
        return "R"
    return None


def resolve_laterality(votes: Iterable[object]) -> tuple[str, str]:
    normalized = [value for value in (normalize_laterality(vote) for vote in votes) if value]
    left, right = normalized.count("L"), normalized.count("R")
    if left == right:
        return "UNKNOWN", "absent" if not normalized else "conflict"
    return ("L", "dicom_majority") if left > right else ("R", "dicom_majority")


def canonical_axes(laterality: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = normalize_laterality(laterality)
    if side is None:
        raise ValueError("canonical medial axis requires known laterality")
    medial = np.array([-1.0, 0.0, 0.0]) if side == "L" else np.array([1.0, 0.0, 0.0])
    anterior = np.array([0.0, -1.0, 0.0])
    superior = np.array([0.0, 0.0, 1.0])
    return medial, anterior, superior


def study_origin(series: Iterable[SeriesGeometry]) -> np.ndarray:
    centers = np.asarray([item.center for item in series], dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] != 3:
        raise ValueError("at least one valid series is required for a study origin")
    return np.median(centers, axis=0)


def patient_to_canonical(point: Sequence[float], origin: Sequence[float], laterality: str) -> np.ndarray:
    delta = _finite_vector(point, 3, "patient point") - _finite_vector(origin, 3, "origin")
    return np.asarray([float(np.dot(delta, axis)) for axis in canonical_axes(laterality)])


def canonical_to_patient(offset: Sequence[float], origin: Sequence[float], laterality: str) -> np.ndarray:
    values = _finite_vector(offset, 3, "canonical offset")
    axes = canonical_axes(laterality)
    return _finite_vector(origin, 3, "origin") + sum(value * axis for value, axis in zip(values, axes))


def centerline_offset_to_patient(offset: Sequence[float], origin: Sequence[float]) -> np.ndarray:
    """Place a side-independent PF/suprapatellar prior for unknown laterality."""
    values = _finite_vector(offset, 3, "canonical offset")
    result = _finite_vector(origin, 3, "origin").copy()
    result += values[1] * np.array([0.0, -1.0, 0.0])
    result += values[2] * np.array([0.0, 0.0, 1.0])
    return result


def anatomical_crop_axes(plane: str, laterality: str | None) -> tuple[np.ndarray, np.ndarray]:
    # The sign of an extent axis is irrelevant. Patient X is therefore valid
    # for side-independent ML extents when laterality is unavailable.
    medial = canonical_axes(laterality)[0] if normalize_laterality(laterality) else np.array([1.0, 0.0, 0.0])
    anterior = np.array([0.0, -1.0, 0.0])
    superior = np.array([0.0, 0.0, 1.0])
    normalized = plane.strip().lower()
    if normalized == "coronal":
        return medial, superior
    if normalized == "sagittal":
        return anterior, superior
    if normalized == "axial":
        return medial, anterior
    raise ValueError(f"unsupported anatomical plane: {plane}")


def crop_bounds(
    geometry: SliceGeometry,
    center_patient: Sequence[float],
    plane: str,
    crop_mm: tuple[float, float],
    laterality: str | None,
) -> tuple[int, int, int, int]:
    """Return requested pixel bounds as ``row0,row1,col0,col1`` (end exclusive)."""
    width_axis, height_axis = anatomical_crop_axes(plane, laterality)
    half_width, half_height = float(crop_mm[0]) / 2, float(crop_mm[1]) / 2
    center = _finite_vector(center_patient, 3, "crop center")
    corners = [
        center + width_sign * half_width * width_axis + height_sign * half_height * height_axis
        for width_sign in (-1.0, 1.0)
        for height_sign in (-1.0, 1.0)
    ]
    pixels = np.asarray([geometry.patient_to_pixel(point) for point in corners])
    row0, col0 = np.floor(pixels.min(axis=0)).astype(int)
    row1, col1 = np.ceil(pixels.max(axis=0)).astype(int)
    if row1 <= row0 or col1 <= col0:
        raise ValueError("projected crop has zero area")
    return int(row0), int(row1), int(col0), int(col1)


def padding_fraction(bounds: tuple[int, int, int, int], rows: int, columns: int) -> float:
    row0, row1, col0, col1 = bounds
    requested = max(row1 - row0, 0) * max(col1 - col0, 0)
    if requested <= 0:
        return 1.0
    overlap_rows = max(0, min(row1, rows) - max(row0, 0))
    overlap_columns = max(0, min(col1, columns) - max(col0, 0))
    return float(1.0 - (overlap_rows * overlap_columns) / requested)


def crop_with_padding(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    row0, row1, col0, col1 = bounds
    output = np.zeros((row1 - row0, col1 - col0), dtype=image.dtype)
    src_row0, src_row1 = max(row0, 0), min(row1, image.shape[0])
    src_col0, src_col1 = max(col0, 0), min(col1, image.shape[1])
    if src_row1 > src_row0 and src_col1 > src_col0:
        dst_row0, dst_col0 = src_row0 - row0, src_col0 - col0
        output[
            dst_row0:dst_row0 + src_row1 - src_row0,
            dst_col0:dst_col0 + src_col1 - src_col0,
        ] = image[src_row0:src_row1, src_col0:src_col1]
    return output


def triplet_indices(
    center: int,
    length: int,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    candidates = ((center - 2, center - 1, center), (center, center + 1, center + 2))
    return tuple(
        item if min(item) >= 0 and max(item) < length else None
        for item in candidates
    )  # type: ignore[return-value]


def nearest_distinct_triplet_indices(
    center: int,
    length: int,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    """Return two distinct, in-bounds triplets closest to ``center``.

    Candidate triplets are all physically consecutive slices.  Ranking by the
    middle index's distance from the physically nearest landmark slice keeps
    the selection deterministic while avoiding boundary clamping or duplicate
    slices.
    """
    candidates = [(index - 1, index, index + 1) for index in range(1, length - 1)]
    ranked = sorted(candidates, key=lambda item: (abs(item[1] - center), item[1]))
    selected = ranked[:2]
    while len(selected) < 2:
        selected.append(None)
    return selected[0], selected[1]


def centered_triplet_indices(center: int, length: int) -> tuple[int, int, int] | None:
    """Return ``center-1, center, center+1`` without edge clamping."""
    candidate = (center - 1, center, center + 1)
    return candidate if min(candidate) >= 0 and max(candidate) < length else None


def pick_series(
    rows: Sequence[dict],
    plane: str,
    fluid_sensitive: int,
    allow_fluid_fallback: bool = True,
) -> dict | None:
    """Existing deterministic first-match rule, with no plane substitution."""
    candidates = [row for row in rows if str(row.get("Anatomical_Plane", "")) == plane]
    preferred = [
        row for row in candidates
        if int(float(row.get("Fluid_Sensitive", 0) or 0)) == fluid_sensitive
    ]
    if preferred:
        return preferred[0]
    return candidates[0] if candidates and allow_fluid_fallback else None


def assert_exact_id_alignment(global_ids: Sequence[object], roi_ids: Sequence[object]) -> None:
    expected = np.asarray(global_ids).astype(str)
    actual = np.asarray(roi_ids).astype(str)
    if expected.ndim != 1 or actual.ndim != 1 or not np.array_equal(expected, actual):
        raise ValueError("ROI IDs do not exactly match the authoritative global-cache IDs")
