"""Build and validate anatomy-specific Arm C ROI caches from raw DICOMs.

Typical Kaggle workflow::

    # 1. Select the label-free twenty-study pilot.
    python build_roi_cache.py select-pilot --output-dir /kaggle/working/roi_pilot

    # 2. After annotating pilot_annotations.csv, run the LOO gate.
    python build_roi_cache.py fit-priors --phase pilot \
        --annotations /kaggle/working/roi_pilot/pilot_annotations.csv \
        --output /kaggle/working/roi_pilot/pilot_priors.json

    # 3. Expand to 100; this freezes the 80/20 partition before annotation.
    python build_roi_cache.py select-cohort \
        --pilot-manifest /kaggle/working/roi_pilot/pilot_manifest.csv \
        --output-dir /kaggle/working/roi_annotations

    # 4. After annotating all 100 studies, fit and audit final priors.
    python build_roi_cache.py fit-priors --phase final \
        --annotations /kaggle/working/roi_annotations/roi_annotations.csv \
        --manifest /kaggle/working/roi_annotations/annotation_manifest.csv \
        --output /kaggle/working/roi_priors.json

    # 5. Run as two CPU jobs.
    python build_roi_cache.py build --shard 0 --priors .../roi_priors.json
    python build_roi_cache.py build --shard 1 --priors .../roi_priors.json

The authoritative row order is read from ``ids_shard<S>.npy``. Unknown
laterality produces paired left/right candidates. A corrupt study, missing
slot, or out-of-FOV crop produces zero data, a false mask, and a provenance
failure reason; it never terminates the shard.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import pydicom
from numpy.lib.format import open_memmap
from pydicom.pixel_data_handlers.util import apply_modality_lut

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_geometry import (  # noqa: E402
    LANDMARKS,
    TARGET_SPECS,
    SeriesGeometry,
    SliceGeometry,
    canonical_to_patient,
    centered_triplet_indices,
    centerline_offset_to_patient,
    crop_bounds,
    crop_with_padding,
    nearest_distinct_triplet_indices,
    padding_fraction,
    patient_to_canonical,
    pick_series,
    resolve_laterality,
    sort_deduplicate_slices,
    study_origin,
    triplet_indices,
)


SEED = 20260919
IMAGE_SIZE = 336
WINDOWS_PER_TARGET = 4
PADDING_LIMIT = 0.20
PROVENANCE_FIELDS = (
    "StudyInstanceUID",
    "target",
    "window_index",
    "SeriesInstanceUID",
    "plane",
    "fluid_sensitive",
    "sop_uid_0",
    "sop_uid_1",
    "sop_uid_2",
    "slice_position_0_mm",
    "slice_position_1_mm",
    "slice_position_2_mm",
    "center_x_mm",
    "center_y_mm",
    "center_z_mm",
    "crop_center_x_mm",
    "crop_center_y_mm",
    "crop_center_z_mm",
    "row0",
    "row1",
    "column0",
    "column1",
    "padding_fraction",
    "intensity_low",
    "intensity_high",
    "laterality",
    "laterality_source",
    "candidate_laterality",
    "window_strategy",
    "valid",
    "failure_reason",
)

cv2.setNumThreads(1)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_competition_root(explicit: str | None = None) -> Path:
    if explicit:
        root = Path(explicit)
        if (root / "train_series.csv").exists() and (root / "train_series").exists():
            return root
        raise FileNotFoundError(f"not a competition root: {root}")
    for directory, subdirectories, filenames in os.walk("/kaggle/input"):
        subdirectories[:] = [name for name in subdirectories if name not in {"train_series", "test_series"}]
        if "train_series.csv" in filenames and "train_series" in subdirectories:
            return Path(directory)
        # os.walk filtering above removes the marker; inspect on disk as well.
        candidate = Path(directory)
        if (candidate / "train_series.csv").exists() and (candidate / "train_series").is_dir():
            return candidate
    raise FileNotFoundError("competition train_series.csv/train_series were not found under /kaggle/input")


def find_named_file(name: str, roots: Iterable[Path] | None = None) -> Path:
    search_roots = list(roots or (Path("/kaggle/input"), Path("/kaggle/working")))
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            matches.append(direct)
        for directory, subdirectories, filenames in os.walk(root):
            subdirectories[:] = [
                item
                for item in subdirectories
                if item not in {"train_series", "test_series"}
            ]
            if name in filenames:
                matches.append(Path(directory) / name)
    unique = sorted(set(matches))
    if not unique:
        raise FileNotFoundError(f"could not find {name}")
    if len(unique) > 1:
        log(f"multiple {name} files found; using {unique[0]}")
    return unique[0]


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def read_series_geometry(
    study_uid: str,
    row: dict[str, Any],
    series_root: Path,
) -> SeriesGeometry:
    series_uid = str(row["SeriesInstanceUID"])
    directory = series_root / study_uid / series_uid
    files = sorted(directory.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no DICOMs in {directory}")
    slices: list[SliceGeometry] = []
    laterality_votes: list[str] = []
    manufacturer = ""
    field_strength = None
    slice_thickness = None
    for path in files:
        header = pydicom.dcmread(path, stop_before_pixels=True)
        slices.append(SliceGeometry.from_header(path, header))
        laterality_votes.extend([
            str(getattr(header, "ImageLaterality", "") or ""),
            str(getattr(header, "Laterality", "") or ""),
        ])
        manufacturer = manufacturer or str(getattr(header, "Manufacturer", "") or "")
        field_strength = field_strength or _safe_float(getattr(header, "MagneticFieldStrength", None))
        slice_thickness = slice_thickness or _safe_float(getattr(header, "SliceThickness", None))
    return SeriesGeometry(
        study_uid=study_uid,
        series_uid=series_uid,
        plane=str(row.get("Anatomical_Plane", "")),
        fluid_sensitive=int(float(row.get("Fluid_Sensitive", 0) or 0)),
        slices=sort_deduplicate_slices(slices),
        laterality_votes=tuple(laterality_votes),
        manufacturer=manufacturer,
        field_strength=field_strength,
        slice_thickness=slice_thickness,
    )


def _read_pixels(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(path)
    image = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = image.max() - image
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"invalid pixel array in {path}")
    return image


def intensity_limits(series: SeriesGeometry) -> tuple[float, float]:
    positions = np.asarray([item.position for item in series.slices])
    low_position = positions[0] + 0.02 * (positions[-1] - positions[0])
    high_position = positions[0] + 0.98 * (positions[-1] - positions[0])
    desired_positions = np.linspace(low_position, high_position, 16)
    sample_indices = np.unique([
        int(np.argmin(np.abs(positions - position)))
        for position in desired_positions
    ])
    sample = np.concatenate([_read_pixels(series.slices[index].path).ravel() for index in sample_indices])
    low, high = np.percentile(sample, [2.0, 98.0])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("series intensity percentiles are degenerate")
    return float(low), float(high)


def _scale_uint8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def load_study_geometry(
    study_uid: str,
    rows: list[dict[str, Any]],
    series_root: Path,
) -> tuple[dict[str, SeriesGeometry], dict[str, str]]:
    """Read every declared study series for a stable, acquisition-wide origin.

    Target series are still chosen by :func:`pick_series`.  Loading all series
    geometry prevents the canonical origin from shifting merely because a
    preferred slot is absent in an imperfect study.
    """
    geometries: dict[str, SeriesGeometry] = {}
    failures: dict[str, str] = {}
    unique_rows = {
        str(row["SeriesInstanceUID"]): row
        for row in rows
    }
    for series_uid in sorted(unique_rows):
        row = unique_rows[series_uid]
        try:
            geometries[series_uid] = read_series_geometry(study_uid, row, series_root)
        except Exception as error:
            failures[series_uid] = f"invalid_series:{type(error).__name__}:{error}"[:500]
    return geometries, failures


def _blank_provenance(
    study_uid: str,
    target: str,
    window_index: int,
    laterality: str,
    laterality_source: str,
    failure_reason: str,
    row: dict[str, Any] | None = None,
    candidate_laterality: str = "",
    window_strategy: str = "",
) -> dict[str, Any]:
    result = {field: "" for field in PROVENANCE_FIELDS}
    result.update({
        "StudyInstanceUID": study_uid,
        "target": target,
        "window_index": window_index,
        "SeriesInstanceUID": "" if row is None else str(row.get("SeriesInstanceUID", "")),
        "plane": "" if row is None else str(row.get("Anatomical_Plane", "")),
        "fluid_sensitive": "" if row is None else int(float(row.get("Fluid_Sensitive", 0) or 0)),
        "laterality": laterality,
        "laterality_source": laterality_source,
        "candidate_laterality": candidate_laterality,
        "window_strategy": window_strategy,
        "valid": 0,
        "failure_reason": failure_reason,
    })
    return result


def _landmark_point(
    landmark: str,
    priors: dict[str, Any],
    origin: np.ndarray,
    laterality: str,
    laterality_required: bool,
) -> np.ndarray:
    offset = priors["landmarks"][landmark]["median_offset_mm"]
    if not laterality_required:
        return centerline_offset_to_patient(offset, origin)
    return canonical_to_patient(offset, origin, laterality)


def _window_candidates(
    specification: dict[str, Any],
    priors: dict[str, Any],
    origin: np.ndarray,
    laterality: str,
) -> list[tuple[str, np.ndarray]]:
    """Return the fixed candidate centers used in every requested plane.

    Side-specific targets always receive both left- and right-knee hypotheses.
    This keeps the four-window tensor useful for the 50% of studies whose
    laterality metadata is absent, without trusting a noisy inferred side.
    """
    landmark = specification["landmark"]
    if specification["laterality_required"]:
        return [
            (side, _landmark_point(landmark, priors, origin, side, True))
            for side in ("L", "R")
        ]
    return [("CENTER", _landmark_point(landmark, priors, origin, laterality, False))]


def _side_independent_window_strategy(specification: dict[str, Any]) -> str:
    return str(specification.get("side_independent_window_strategy", "center_overlapping"))


def _crop_center(
    specification: dict[str, Any],
    plane: str,
    landmark_center: np.ndarray,
) -> np.ndarray:
    """Return target/plane-specific physical crop centre in LPS millimetres."""
    offsets = specification.get("crop_center_offset_lps_mm_by_plane", {})
    offset = offsets.get(plane, (0.0, 0.0, 0.0))
    value = np.asarray(offset, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"invalid crop centre offset for {specification['name']}/{plane}")
    return landmark_center + value


def build_study(
    study_uid: str,
    rows: list[dict[str, Any]],
    series_root: Path,
    priors: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    data = {
        target: np.zeros((WINDOWS_PER_TARGET, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for target in TARGET_SPECS
    }
    masks = {target: np.zeros(WINDOWS_PER_TARGET, dtype=np.uint8) for target in TARGET_SPECS}
    provenance: list[dict[str, Any]] = []
    geometries, series_failures = load_study_geometry(study_uid, rows, series_root)
    series_votes = [resolve_laterality(geometry.laterality_votes)[0] for geometry in geometries.values()]
    laterality, laterality_source = resolve_laterality(series_votes)
    if not geometries:
        for target in TARGET_SPECS:
            for window_index in range(WINDOWS_PER_TARGET):
                provenance.append(_blank_provenance(
                    study_uid, target, window_index, laterality, laterality_source,
                    "no_valid_selected_series",
                ))
        return data, masks, provenance
    origin = study_origin(geometries.values())
    intensity_cache: dict[str, tuple[float, float] | Exception] = {}
    pixel_cache: dict[tuple[str, int], np.ndarray] = {}

    for target, specification in TARGET_SPECS.items():
        try:
            candidates = _window_candidates(
                specification,
                priors,
                origin,
                laterality,
            )
        except Exception as error:
            for window_index in range(WINDOWS_PER_TARGET):
                provenance.append(_blank_provenance(
                    study_uid, target, window_index, laterality, laterality_source,
                    f"landmark_projection_failed:{type(error).__name__}:{error}"[:500],
                ))
            continue
        for slot_index, (plane, fluid) in enumerate(specification["slots"]):
            row = pick_series(
                rows,
                plane,
                fluid,
                allow_fluid_fallback=specification["allow_fluid_fallback"],
            )
            base_window = slot_index * 2
            if row is None:
                for local_index in range(2):
                    candidate_side = (
                        candidates[local_index][0]
                        if specification["laterality_required"]
                        else "CENTER"
                    )
                    provenance.append(_blank_provenance(
                        study_uid, target, base_window + local_index, laterality,
                        laterality_source, "missing_required_series", None,
                        candidate_laterality=candidate_side,
                        window_strategy=(
                            "paired_side_centered" if specification["laterality_required"]
                        else _side_independent_window_strategy(specification)
                        ),
                    ))
                continue
            series_uid = str(row["SeriesInstanceUID"])
            series = geometries.get(series_uid)
            if series is None:
                reason = series_failures.get(series_uid, "invalid_required_series")
                for local_index in range(2):
                    candidate_side = (
                        candidates[local_index][0]
                        if specification["laterality_required"]
                        else "CENTER"
                    )
                    provenance.append(_blank_provenance(
                        study_uid, target, base_window + local_index, laterality,
                        laterality_source, reason, row,
                        candidate_laterality=candidate_side,
                        window_strategy=(
                            "paired_side_centered" if specification["laterality_required"]
                        else _side_independent_window_strategy(specification)
                        ),
                    ))
                continue
            if series_uid not in intensity_cache:
                try:
                    intensity_cache[series_uid] = intensity_limits(series)
                except Exception as error:
                    intensity_cache[series_uid] = error
            limits = intensity_cache[series_uid]
            if isinstance(limits, Exception):
                for local_index in range(2):
                    candidate_side = (
                        candidates[local_index][0]
                        if specification["laterality_required"]
                        else "CENTER"
                    )
                    provenance.append(_blank_provenance(
                        study_uid, target, base_window + local_index, laterality,
                        laterality_source,
                        f"intensity_failed:{type(limits).__name__}:{limits}"[:500], row,
                        candidate_laterality=candidate_side,
                        window_strategy=(
                            "paired_side_centered" if specification["laterality_required"]
                        else _side_independent_window_strategy(specification)
                        ),
                    ))
                continue
            if specification["laterality_required"]:
                planned_windows = [
                    (
                        candidate_side,
                        center,
                        centered_triplet_indices(
                            series.nearest_slice_index(center),
                            len(series.slices),
                        ),
                        "paired_side_centered",
                    )
                    for candidate_side, center in candidates
                ]
            else:
                center = candidates[0][1]
                strategy = _side_independent_window_strategy(specification)
                nearest_index = series.nearest_slice_index(center)
                if strategy == "nearest_distinct_triplets":
                    selected_triplets = nearest_distinct_triplet_indices(
                        nearest_index,
                        len(series.slices),
                    )
                elif strategy == "center_overlapping":
                    selected_triplets = triplet_indices(nearest_index, len(series.slices))
                else:
                    raise ValueError(f"unsupported side-independent strategy: {strategy}")
                planned_windows = [
                    ("CENTER", center, indices, strategy)
                    for indices in selected_triplets
                ]
            for local_index, (candidate_side, center, indices, strategy) in enumerate(planned_windows):
                window_index = base_window + local_index
                if indices is None:
                    provenance.append(_blank_provenance(
                        study_uid, target, window_index, laterality,
                        laterality_source, "triplet_crosses_series_boundary", row,
                        candidate_laterality=candidate_side,
                        window_strategy=strategy,
                    ))
                    continue
                middle = series.slices[indices[1]]
                try:
                    crop_center = _crop_center(specification, plane, center)
                    bounds = crop_bounds(
                        middle, crop_center, plane, specification["crop_mm"], laterality,
                    )
                    pad = padding_fraction(bounds, middle.rows, middle.columns)
                    if pad > PADDING_LIMIT + 1e-12:
                        raise ValueError(f"padding_fraction_exceeds_0.20:{pad:.6f}")
                    channels = []
                    triplet_padding = []
                    for index in indices:
                        item = series.slices[index]
                        cache_key = (series_uid, index)
                        if cache_key not in pixel_cache:
                            pixel_cache[cache_key] = _scale_uint8(
                                _read_pixels(item.path), limits[0], limits[1],
                            )
                        image = pixel_cache[cache_key]
                        item_bounds = crop_bounds(
                            item, crop_center, plane, specification["crop_mm"], laterality,
                        )
                        item_pad = padding_fraction(item_bounds, item.rows, item.columns)
                        if item_pad > PADDING_LIMIT + 1e-12:
                            raise ValueError(f"padding_fraction_exceeds_0.20:{item_pad:.6f}")
                        triplet_padding.append(item_pad)
                        crop = crop_with_padding(image, item_bounds)
                        channels.append(cv2.resize(
                            crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR,
                        ))
                    data[target][window_index] = np.stack(channels)
                    masks[target][window_index] = 1
                    provenance.append({
                        "StudyInstanceUID": study_uid,
                        "target": target,
                        "window_index": window_index,
                        "SeriesInstanceUID": series_uid,
                        "plane": plane,
                        "fluid_sensitive": fluid,
                        "sop_uid_0": series.slices[indices[0]].sop_uid,
                        "sop_uid_1": series.slices[indices[1]].sop_uid,
                        "sop_uid_2": series.slices[indices[2]].sop_uid,
                        "slice_position_0_mm": series.slices[indices[0]].position,
                        "slice_position_1_mm": series.slices[indices[1]].position,
                        "slice_position_2_mm": series.slices[indices[2]].position,
                        "center_x_mm": float(center[0]),
                        "center_y_mm": float(center[1]),
                        "center_z_mm": float(center[2]),
                        "crop_center_x_mm": float(crop_center[0]),
                        "crop_center_y_mm": float(crop_center[1]),
                        "crop_center_z_mm": float(crop_center[2]),
                        "row0": bounds[0],
                        "row1": bounds[1],
                        "column0": bounds[2],
                        "column1": bounds[3],
                        "padding_fraction": max(triplet_padding),
                        "intensity_low": limits[0],
                        "intensity_high": limits[1],
                        "laterality": laterality,
                        "laterality_source": laterality_source,
                        "candidate_laterality": candidate_side,
                        "window_strategy": strategy,
                        "valid": 1,
                        "failure_reason": "",
                    })
                except Exception as error:
                    provenance.append(_blank_provenance(
                        study_uid, target, window_index, laterality, laterality_source,
                        f"crop_failed:{type(error).__name__}:{error}"[:500], row,
                        candidate_laterality=candidate_side,
                        window_strategy=strategy,
                    ))
    if len(provenance) != len(TARGET_SPECS) * WINDOWS_PER_TARGET:
        raise RuntimeError(f"internal provenance count error for {study_uid}: {len(provenance)}")
    return data, masks, provenance


_WORKER_CONTEXT: dict[str, Any] = {}


def _worker_initialize(by_study: dict[str, list[dict]], series_root: str, priors: dict) -> None:
    _WORKER_CONTEXT.update(by_study=by_study, series_root=Path(series_root), priors=priors)
    cv2.setNumThreads(1)


def _worker_job(item: tuple[int, str]):
    index, study_uid = item
    try:
        data, masks, provenance = build_study(
            study_uid,
            _WORKER_CONTEXT["by_study"].get(study_uid, []),
            _WORKER_CONTEXT["series_root"],
            _WORKER_CONTEXT["priors"],
        )
        return index, data, masks, provenance, None
    except Exception as error:
        data = {
            target: np.zeros((WINDOWS_PER_TARGET, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            for target in TARGET_SPECS
        }
        masks = {target: np.zeros(WINDOWS_PER_TARGET, dtype=np.uint8) for target in TARGET_SPECS}
        provenance = [
            _blank_provenance(
                study_uid,
                target,
                window,
                "UNKNOWN",
                "error",
                f"study_failed:{type(error).__name__}:{error}"[:500],
            )
            for target in TARGET_SPECS
            for window in range(WINDOWS_PER_TARGET)
        ]
        return index, data, masks, provenance, repr(error)[:500]


def command_build(arguments: argparse.Namespace) -> None:
    started = time.time()
    competition = find_competition_root(arguments.competition_root)
    ids_path = (
        Path(arguments.ids_path)
        if arguments.ids_path
        else find_named_file(f"ids_shard{arguments.shard}.npy")
    )
    ids = np.load(ids_path, allow_pickle=True).astype(str)
    if ids.ndim != 1 or len(ids) != len(set(ids.tolist())):
        raise ValueError(f"authoritative IDs are not a unique vector: {ids_path}")
    priors = json.loads(Path(arguments.priors).read_text())
    if not priors.get("accepted", False):
        raise ValueError("roi_priors.json has not passed its annotation coverage gate")
    if priors.get("phase") != "final" and not getattr(arguments, "allow_pilot_priors", False):
        raise ValueError("full shard builds require final 80/20 priors, not pilot priors")
    frame = pd.read_csv(competition / "train_series.csv")
    frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
    frame["SeriesInstanceUID"] = frame.SeriesInstanceUID.astype(str)
    by_study = {key: group.to_dict("records") for key, group in frame.groupby("StudyInstanceUID", sort=False)}
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = len(ids) if not arguments.limit else min(len(ids), arguments.limit)
    ids = ids[:count]
    arrays = {
        target: open_memmap(
            output / f"roi_{target}_shard{arguments.shard}.npy",
            mode="w+", dtype=np.uint8,
            shape=(count, WINDOWS_PER_TARGET, 3, IMAGE_SIZE, IMAGE_SIZE),
        )
        for target in TARGET_SPECS
    }
    masks = {
        target: open_memmap(
            output / f"roi_{target}_masks_shard{arguments.shard}.npy",
            mode="w+", dtype=np.uint8,
            shape=(count, WINDOWS_PER_TARGET),
        )
        for target in TARGET_SPECS
    }
    provenance_path = output / f"roi_provenance_shard{arguments.shard}.csv.gz"
    failures: list[tuple[str, str]] = []
    last_log = time.time()
    with gzip.open(provenance_path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROVENANCE_FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(
            max_workers=arguments.workers,
            initializer=_worker_initialize,
            initargs=(by_study, str(competition / "train_series"), priors),
        ) as pool:
            jobs = pool.map(_worker_job, enumerate(ids.tolist()), chunksize=1)
            for completed, (index, study_data, study_masks, study_provenance, error) in enumerate(jobs, 1):
                for target in TARGET_SPECS:
                    arrays[target][index] = study_data[target]
                    masks[target][index] = study_masks[target]
                writer.writerows(study_provenance)
                if error:
                    failures.append((str(ids[index]), error))
                if time.time() - last_log >= 60 or completed == count:
                    elapsed = time.time() - started
                    eta = elapsed / completed * (count - completed) / 60
                    log(
                        f"{completed}/{count} studies | {elapsed / 60:.1f} min | "
                        f"eta {eta:.1f} min | study failures {len(failures)}"
                    )
                    last_log = time.time()
    for item in arrays.values():
        item.flush()
    for item in masks.values():
        item.flush()
    np.save(output / f"roi_ids_shard{arguments.shard}.npy", ids)
    Path(output / "roi_priors.json").write_text(json.dumps(priors, indent=2, sort_keys=True))
    deterministic_studies: list[str] = []
    rng = np.random.default_rng(SEED + arguments.shard)
    for index in rng.choice(count, min(3, count), replace=False):
        study_uid = str(ids[index])
        rebuilt_data, rebuilt_masks, _ = build_study(
            study_uid,
            by_study.get(study_uid, []),
            competition / "train_series",
            priors,
        )
        for target in TARGET_SPECS:
            if not np.array_equal(rebuilt_data[target], np.asarray(arrays[target][index])):
                raise RuntimeError(f"non-deterministic ROI pixels for {study_uid}/{target}")
            if not np.array_equal(rebuilt_masks[target], np.asarray(masks[target][index])):
                raise RuntimeError(f"non-deterministic ROI mask for {study_uid}/{target}")
        deterministic_studies.append(study_uid)
    known_laterality = 0
    with gzip.open(provenance_path, "rt", newline="") as stream:
        seen: dict[str, str] = {}
        for row in csv.DictReader(stream):
            seen.setdefault(row["StudyInstanceUID"], row["laterality"])
        known_laterality = sum(value in {"L", "R"} for value in seen.values())
    coverage = {}
    for target, mask in masks.items():
        coverage[target] = float((np.asarray(mask).sum(axis=1) >= 2).mean())
    meta = {
        "version": 3,
        "window_contract": {
            "laterality_dependent": "paired_side_centered",
            "side_independent": "target_declared",
        },
        "shard": arguments.shard,
        "ids_source": str(ids_path),
        "ids_sha256": sha256_file(ids_path),
        "train_series_csv_sha256": sha256_file(competition / "train_series.csv"),
        "priors_sha256": sha256_file(Path(arguments.priors)),
        "study_count": count,
        "shape": [count, WINDOWS_PER_TARGET, 3, IMAGE_SIZE, IMAGE_SIZE],
        "dtype": "uint8",
        "mask_dtype": "uint8",
        "targets": TARGET_SPECS,
        "at_least_two_valid_windows_rate": coverage,
        "known_laterality_rate": known_laterality / max(count, 1),
        "deterministic_rebuild_studies": deterministic_studies,
        "deterministic_rebuild_passed": True,
        "study_failures": failures,
        "minutes": (time.time() - started) / 60,
    }
    (output / f"roi_meta_shard{arguments.shard}.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    log(f"finished shard {arguments.shard}; run the validate command before training")


def _header_summary(path: Path) -> dict[str, Any]:
    header = pydicom.dcmread(path, stop_before_pixels=True)
    spacing = getattr(header, "PixelSpacing", [None, None])
    laterality, source = resolve_laterality([
        getattr(header, "ImageLaterality", ""), getattr(header, "Laterality", ""),
    ])
    return {
        "laterality": laterality,
        "laterality_source": source,
        "manufacturer": str(getattr(header, "Manufacturer", "") or "UNKNOWN"),
        "field_strength": _safe_float(getattr(header, "MagneticFieldStrength", None)),
        "pixel_spacing": _safe_float(spacing[0]) if len(spacing) else None,
        "slice_thickness": _safe_float(getattr(header, "SliceThickness", None)),
    }


def _normalize_manufacturer(value: object) -> str:
    text = str(value or "").upper()
    if "SIEMENS" in text:
        return "Siemens"
    if "PHILIPS" in text:
        return "Philips"
    if "GE " in f"{text} " or text in {"GE", "GEHC"} or "GENERAL ELECTRIC" in text:
        return "GE"
    if "CANON" in text or "TOSHIBA" in text:
        return "Canon/Toshiba"
    if "FUJIFILM" in text or "HITACHI" in text:
        return "Fujifilm/Hitachi"
    return "Other/unknown"


def _range_bin(value: float | None, edges: tuple[float, ...]) -> str:
    if value is None:
        return "unknown"
    lower = float("-inf")
    for upper in edges:
        if value < upper:
            return f"[{lower:g},{upper:g})"
        lower = upper
    return f"[{lower:g},inf)"


def _annotation_template(selected: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "StudyInstanceUID", "landmark", "SeriesInstanceUID", "SOPInstanceUID",
        "row_px", "column_px", "patient_x_mm", "patient_y_mm", "patient_z_mm",
        "laterality", "annotator", "qc_status",
    ]
    records = [
        {
            "StudyInstanceUID": item["StudyInstanceUID"],
            "landmark": landmark,
            "laterality": item["laterality"],
        }
        for item in selected
        for landmark in LANDMARKS
    ]
    return pd.DataFrame(records, columns=columns)


def _collect_annotation_candidates(
    competition: Path,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build label-free, study-level metadata used for cohort selection."""
    candidates: list[dict[str, Any]] = []
    needed = {
        ("Coronal", 1),
        ("Axial", 1),
        ("Sagittal", 0),
        ("Coronal", 0),
        ("Sagittal", 1),
    }
    for study_uid, group in frame.groupby("StudyInstanceUID", sort=True):
        rows = group.to_dict("records")
        available = {
            (
                str(row.get("Anatomical_Plane", "")),
                int(float(row.get("Fluid_Sensitive", 0) or 0)),
            )
            for row in rows
        }
        header_summaries: list[dict[str, Any]] = []
        series_depths: list[int] = []
        for row in rows:
            directory = competition / "train_series" / study_uid / str(row["SeriesInstanceUID"])
            files = sorted(directory.glob("*.dcm"))
            if not files:
                continue
            series_depths.append(len(files))
            try:
                header_summaries.append(_header_summary(files[len(files) // 2]))
            except Exception:
                continue
        if not header_summaries:
            continue
        laterality, laterality_source = resolve_laterality(
            summary["laterality"] for summary in header_summaries
        )
        # Canonical medial/lateral offsets can only be fitted from a trusted
        # side.  Unknown studies remain usable during production because the
        # builder creates symmetric hypotheses, but they are not annotations.
        if laterality not in {"L", "R"}:
            continue

        def median_value(field: str) -> float | None:
            values = [
                float(item[field])
                for item in header_summaries
                if item[field] is not None
            ]
            return float(np.median(values)) if values else None

        manufacturer = Counter(
            _normalize_manufacturer(item["manufacturer"])
            for item in header_summaries
        ).most_common(1)[0][0]
        field_strength = median_value("field_strength")
        pixel_spacing = median_value("pixel_spacing")
        slice_thickness = median_value("slice_thickness")
        complete = needed.issubset(available)
        has_reference_views = (
            any(plane == "Coronal" for plane, _ in available)
            and ("Axial", 1) in available
            and ("Sagittal", 1) in available
        )
        if not has_reference_views:
            continue
        minimum_depth = min(series_depths) if series_depths else 0
        candidates.append({
            "StudyInstanceUID": study_uid,
            "laterality": laterality,
            "laterality_source": laterality_source,
            "manufacturer": manufacturer,
            "field_strength": field_strength,
            "pixel_spacing": pixel_spacing,
            "slice_thickness": slice_thickness,
            "complete_preferred_planes": complete,
            "available_slots": len(needed.intersection(available)),
            "minimum_series_depth": minimum_depth,
            "manufacturer_bin": manufacturer,
            "field_strength_bin": _range_bin(field_strength, (1.25, 2.25, 3.25)),
            "pixel_spacing_bin": _range_bin(pixel_spacing, (0.2, 0.3, 0.4, 0.6)),
            "slice_thickness_bin": _range_bin(slice_thickness, (2.0, 3.0, 4.0)),
            "series_depth_bin": _range_bin(float(minimum_depth), (18.0, 46.0)),
        })
    return candidates


SELECTION_FEATURES = (
    "manufacturer_bin",
    "field_strength_bin",
    "pixel_spacing_bin",
    "slice_thickness_bin",
    "series_depth_bin",
)


def _selection_tiebreak(study_uid: str, salt: str) -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{study_uid}".encode()).hexdigest()


def _extend_representative_selection(
    pool: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    additions: int,
    final_side_counts: dict[str, int],
    population: list[dict[str, Any]],
    salt: str,
) -> None:
    """Greedily match population metadata without over-sampling rare strata."""
    available = {
        item["StudyInstanceUID"]: item
        for item in pool
        if item["StudyInstanceUID"] not in {row["StudyInstanceUID"] for row in selected}
    }
    population_counts = {
        feature: Counter(item[feature] for item in population)
        for feature in SELECTION_FEATURES
    }
    total_population = max(len(population), 1)
    for _ in range(additions):
        side_counts = Counter(item["laterality"] for item in selected)
        allowed_sides = {
            side
            for side, target in final_side_counts.items()
            if side_counts[side] < target
        }
        candidates = [
            item for item in available.values()
            if item["laterality"] in allowed_sides
        ]
        if not candidates:
            raise RuntimeError("not enough side-balanced studies for annotation selection")
        next_size = len(selected) + 1
        selected_counts = {
            feature: Counter(item[feature] for item in selected)
            for feature in SELECTION_FEATURES
        }

        def objective(item: dict[str, Any]) -> tuple[float, str]:
            error = 0.0
            for feature in SELECTION_FEATURES:
                categories = population_counts[feature]
                for category, population_count in categories.items():
                    expected = next_size * population_count / total_population
                    observed = selected_counts[feature][category]
                    if item[feature] == category:
                        observed += 1
                    error += (observed - expected) ** 2 / (expected + 0.5)
            return error, _selection_tiebreak(item["StudyInstanceUID"], salt)

        winner = min(candidates, key=objective)
        selected.append(winner)
        del available[winner["StudyInstanceUID"]]


def _write_annotation_selection(
    selected: list[dict[str, Any]],
    output: Path,
    manifest_name: str,
    annotations_name: str,
    ids_name: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(selected, key=lambda item: item["StudyInstanceUID"])
    pd.DataFrame(ordered).to_csv(output / manifest_name, index=False)
    np.save(output / ids_name, np.asarray([item["StudyInstanceUID"] for item in ordered]))
    _annotation_template(ordered).to_csv(output / annotations_name, index=False)


def command_select_pilot(arguments: argparse.Namespace) -> None:
    competition = find_competition_root(arguments.competition_root)
    frame = pd.read_csv(competition / "train_series.csv")
    frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
    frame["SeriesInstanceUID"] = frame.SeriesInstanceUID.astype(str)
    candidates = _collect_annotation_candidates(competition, frame)
    selected: list[dict[str, Any]] = []
    complete = [item for item in candidates if item["complete_preferred_planes"]]
    imperfect = [item for item in candidates if not item["complete_preferred_planes"]]
    # A 20-study representative sample would round the small GE and
    # Fujifilm/Hitachi cohorts to zero.  Retain exactly one complete study from
    # every eligible vendor family, then distribution-match the remaining
    # slots.  This exercises vendor-specific geometry without repeating the
    # previous pilot's four-fold over-sampling of the 1.16 T cohort.
    for manufacturer in sorted({item["manufacturer"] for item in complete}):
        if manufacturer in {"Siemens", "Philips"}:
            continue
        vendor_pool = [item for item in complete if item["manufacturer"] == manufacturer]
        if vendor_pool:
            winner = min(
                vendor_pool,
                key=lambda item: _selection_tiebreak(
                    item["StudyInstanceUID"], f"pilot-vendor-{manufacturer}",
                ),
            )
            winner["selection_role"] = "pilot_vendor_coverage"
            selected.append(winner)
    _extend_representative_selection(
        complete,
        selected,
        16 - len(selected),
        {"L": 8, "R": 8},
        candidates,
        "pilot-complete",
    )
    for item in selected:
        item.setdefault("selection_role", "pilot_complete")
        item["prior_partition"] = "loo_audit"
    _extend_representative_selection(
        imperfect, selected, 4, {"L": 10, "R": 10}, candidates, "pilot-imperfect",
    )
    for item in selected:
        item.setdefault("selection_role", "pilot_imperfect")
        item.setdefault("prior_partition", "loo_audit")
    output = Path(arguments.output_dir)
    _write_annotation_selection(
        selected,
        output,
        "pilot_manifest.csv",
        "pilot_annotations.csv",
        "pilot_ids.npy",
    )
    log(f"wrote deterministic 20-study pilot to {output}")


def command_select_cohort(arguments: argparse.Namespace) -> None:
    """Expand the frozen pilot to a representative 100-study annotation set."""
    competition = find_competition_root(arguments.competition_root)
    frame = pd.read_csv(competition / "train_series.csv")
    frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
    frame["SeriesInstanceUID"] = frame.SeriesInstanceUID.astype(str)
    candidates = _collect_annotation_candidates(competition, frame)
    by_uid = {item["StudyInstanceUID"]: item for item in candidates}
    pilot = pd.read_csv(arguments.pilot_manifest, dtype={"StudyInstanceUID": str})
    pilot_ids = pilot.StudyInstanceUID.astype(str).tolist()
    if len(pilot_ids) != 20 or len(set(pilot_ids)) != 20:
        raise ValueError("pilot manifest must contain exactly 20 unique studies")
    missing = sorted(set(pilot_ids).difference(by_uid))
    if missing:
        raise ValueError(f"pilot studies are no longer eligible: {missing[:3]}")
    selected = [dict(by_uid[study]) for study in pilot_ids]
    for item in selected:
        item["selection_role"] = "pilot_retained"
    complete = [item for item in candidates if item["complete_preferred_planes"]]
    imperfect = [item for item in candidates if not item["complete_preferred_planes"]]
    current_complete = sum(item["complete_preferred_planes"] for item in selected)
    _extend_representative_selection(
        complete,
        selected,
        80 - current_complete,
        # The retained pilot already contributes two imperfect studies per
        # side, so 80 complete + 4 retained imperfect = 42/42 here.
        {"L": 42, "R": 42},
        candidates,
        "cohort-complete",
    )
    for item in selected:
        item.setdefault("selection_role", "cohort_complete")
    _extend_representative_selection(
        imperfect,
        selected,
        100 - len(selected),
        {"L": 50, "R": 50},
        candidates,
        "cohort-imperfect",
    )
    for item in selected:
        item.setdefault("selection_role", "cohort_imperfect")

    # Freeze a representative 16-complete/4-imperfect audit before annotation.
    audit: list[dict[str, Any]] = []
    selected_complete = [item for item in selected if item["complete_preferred_planes"]]
    selected_imperfect = [item for item in selected if not item["complete_preferred_planes"]]
    _extend_representative_selection(
        selected_complete, audit, 16, {"L": 8, "R": 8}, selected, "audit-complete",
    )
    _extend_representative_selection(
        selected_imperfect, audit, 4, {"L": 10, "R": 10}, selected, "audit-imperfect",
    )
    audit_ids = {item["StudyInstanceUID"] for item in audit}
    for item in selected:
        item["prior_partition"] = (
            "audit" if item["StudyInstanceUID"] in audit_ids else "fit"
        )
    output = Path(arguments.output_dir)
    _write_annotation_selection(
        selected,
        output,
        "annotation_manifest.csv",
        "roi_annotations.csv",
        "annotation_ids.npy",
    )
    log(f"wrote deterministic 100-study annotation cohort to {output}")


def command_build_pilot(arguments: argparse.Namespace) -> None:
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids_path = output / "pilot_ids.npy"
    if arguments.manifest:
        manifest = pd.read_csv(arguments.manifest, dtype={"StudyInstanceUID": str})
        np.save(ids_path, manifest.StudyInstanceUID.astype(str).to_numpy())
    elif not ids_path.exists():
        raise FileNotFoundError("pilot_ids.npy is missing; pass --manifest from select-pilot")
    forwarded = argparse.Namespace(
        competition_root=arguments.competition_root,
        ids_path=str(ids_path),
        shard=0,
        priors=arguments.priors,
        output_dir=str(output),
        workers=arguments.workers,
        limit=0,
        allow_pilot_priors=True,
    )
    command_build(forwarded)


def _coverage_for_offset(actual: np.ndarray, predicted: np.ndarray, laterality: str, target: str) -> bool:
    specification = TARGET_SPECS[target]
    residual = actual - predicted
    # A landmark must be visible in both requested plane crops. Dot products
    # use the same anatomical axes as the production crop projection.
    from roi_geometry import anatomical_crop_axes
    for plane, _ in specification["slots"]:
        width_axis, height_axis = anatomical_crop_axes(plane, laterality)
        if abs(float(np.dot(residual, width_axis))) > specification["crop_mm"][0] / 2:
            return False
        if abs(float(np.dot(residual, height_axis))) > specification["crop_mm"][1] / 2:
            return False
    return True


def _annotation_offsets(
    annotations: pd.DataFrame,
    by_study: dict[str, list[dict]],
    series_root: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    offsets: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    laterality_by_study: dict[str, str] = {}
    for study_uid, group in annotations.groupby("StudyInstanceUID", sort=True):
        rows = by_study.get(str(study_uid), [])
        geometries, _ = load_study_geometry(str(study_uid), rows, series_root)
        if not geometries:
            raise ValueError(f"no valid selected series for annotated study {study_uid}")
        origin = study_origin(geometries.values())
        annotated_sides = [str(value) for value in group.laterality if str(value) in {"L", "R"}]
        dicom_side, _ = resolve_laterality([
            resolve_laterality(geometry.laterality_votes)[0]
            for geometry in geometries.values()
        ])
        side = Counter(annotated_sides).most_common(1)[0][0] if annotated_sides else dicom_side
        if side not in {"L", "R"}:
            raise ValueError(f"unknown laterality for annotated study {study_uid}")
        laterality_by_study[str(study_uid)] = side
        for row in group.itertuples(index=False):
            point = np.asarray([row.patient_x_mm, row.patient_y_mm, row.patient_z_mm], dtype=np.float64)
            approved = str(row.qc_status).lower() in {"pass", "passed", "approved"}
            if not np.isfinite(point).all() or not approved:
                raise ValueError(f"annotation is incomplete or not approved: {study_uid}/{row.landmark}")
            offsets[str(study_uid)][str(row.landmark)] = patient_to_canonical(point, origin, side)
    return offsets, laterality_by_study


def _prior_statistics(studies: list[str], offsets: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for landmark in LANDMARKS:
        values = np.asarray([offsets[study][landmark] for study in studies])
        median = np.median(values, axis=0)
        result[landmark] = {
            "median_offset_mm": median.tolist(),
            "mad_mm": np.median(np.abs(values - median), axis=0).tolist(),
            "p95_abs_deviation_mm": np.percentile(np.abs(values - median), 95, axis=0).tolist(),
            "n": len(values),
        }
    return result


def _audit_coverage(
    audit_studies: list[str],
    fit_studies: list[str],
    offsets: dict[str, dict[str, np.ndarray]],
    laterality: dict[str, str],
    leave_one_out: bool,
) -> dict[str, Any]:
    target_results: dict[str, list[bool]] = {target: [] for target in TARGET_SPECS}
    mirror_errors = 0
    for study in audit_studies:
        training = [item for item in fit_studies if item != study] if leave_one_out else fit_studies
        for landmark in LANDMARKS:
            predicted_offset = np.median([offsets[item][landmark] for item in training], axis=0)
            actual = canonical_to_patient(offsets[study][landmark], np.zeros(3), laterality[study])
            for target, specification in TARGET_SPECS.items():
                if specification["landmark"] == landmark:
                    predicted = (
                        canonical_to_patient(
                            predicted_offset,
                            np.zeros(3),
                            laterality[study],
                        )
                        if specification["laterality_required"]
                        else centerline_offset_to_patient(predicted_offset, np.zeros(3))
                    )
                    target_results[target].append(
                        _coverage_for_offset(actual, predicted, laterality[study], target)
                    )
        # A coordinate origin can legitimately be off-center.  Mirror auditing
        # therefore compares the two annotated points to each other rather
        # than requiring each one to have a particular sign around the origin.
        if (
            offsets[study]["medial_joint_line"][0]
            <= offsets[study]["lateral_joint_line"][0]
        ):
            mirror_errors += 1
    rates = {target: float(np.mean(values)) if values else 0.0 for target, values in target_results.items()}
    return {
        "coverage_by_target": rates,
        "minimum_target_coverage": min(rates.values()),
        "mirror_errors": mirror_errors,
    }


def command_fit_priors(arguments: argparse.Namespace) -> None:
    annotations = pd.read_csv(arguments.annotations)
    required = {
        "StudyInstanceUID", "landmark", "patient_x_mm", "patient_y_mm", "patient_z_mm",
        "laterality", "qc_status",
    }
    missing = sorted(required.difference(annotations.columns))
    if missing:
        raise ValueError(f"annotations are missing columns: {missing}")
    annotations["StudyInstanceUID"] = annotations.StudyInstanceUID.astype(str)
    duplicates = annotations.duplicated(["StudyInstanceUID", "landmark"])
    if duplicates.any():
        raise ValueError("annotations contain duplicate study/landmark rows")
    counts = annotations.groupby("StudyInstanceUID").landmark.nunique()
    if not (counts == len(LANDMARKS)).all():
        raise ValueError("every annotated study must have exactly four unique landmarks")
    expected = 20 if arguments.phase == "pilot" else 100
    if len(counts) != expected:
        raise ValueError(
            f"{arguments.phase} requires exactly {expected} fully annotated studies; "
            f"received {len(counts)}"
        )
    competition = find_competition_root(arguments.competition_root)
    frame = pd.read_csv(competition / "train_series.csv")
    frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
    frame["SeriesInstanceUID"] = frame.SeriesInstanceUID.astype(str)
    by_study = {key: group.to_dict("records") for key, group in frame.groupby("StudyInstanceUID", sort=False)}
    offsets, laterality = _annotation_offsets(annotations, by_study, competition / "train_series")
    studies = sorted(offsets)
    if arguments.phase == "pilot":
        fit_studies = studies
        audit_studies = studies
        audit = _audit_coverage(studies, studies, offsets, laterality, leave_one_out=True)
        threshold = 0.90
    else:
        if not arguments.manifest:
            raise ValueError(
                "final prior fitting requires --manifest from select-cohort so "
                "the 80/20 split is frozen before annotation"
            )
        manifest = pd.read_csv(arguments.manifest, dtype={"StudyInstanceUID": str})
        if not {"StudyInstanceUID", "prior_partition"}.issubset(manifest.columns):
            raise ValueError(
                "final manifest must contain StudyInstanceUID and prior_partition"
            )
        if set(manifest.StudyInstanceUID.astype(str)) != set(studies):
            raise ValueError("annotation and manifest study IDs do not match exactly")
        partitions = dict(zip(
            manifest.StudyInstanceUID.astype(str),
            manifest.prior_partition.astype(str),
        ))
        fit_studies = sorted(study for study in studies if partitions[study] == "fit")
        audit_studies = sorted(study for study in studies if partitions[study] == "audit")
        if len(fit_studies) != 80 or len(audit_studies) != 20:
            raise ValueError(
                "final manifest must freeze exactly 80 fit and 20 audit studies"
            )
        audit = _audit_coverage(audit_studies, fit_studies, offsets, laterality, leave_one_out=False)
        threshold = 0.95
    laterality_rate = float(np.mean([laterality.get(study) in {"L", "R"} for study in audit_studies]))
    accepted = (
        audit["minimum_target_coverage"] >= threshold
        and audit["mirror_errors"] == 0
        and laterality_rate >= 0.95
    )
    result = {
        "version": 2,
        "seed": SEED,
        "phase": arguments.phase,
        "accepted": accepted,
        "coverage_threshold": threshold,
        "fit_study_ids": fit_studies,
        "audit_study_ids": audit_studies,
        "laterality_availability": laterality_rate,
        "audit": audit,
        "landmarks": _prior_statistics(fit_studies, offsets),
    }
    Path(arguments.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    log(
        f"wrote {arguments.phase} priors: accepted={accepted}, "
        f"minimum coverage={audit['minimum_target_coverage']:.3f}"
    )
    if not accepted:
        raise RuntimeError("annotation gate failed; review annotations/geometry before cache generation")


def command_validate(arguments: argparse.Namespace) -> None:
    directory = Path(arguments.cache_dir)
    failures: list[str] = []
    reports: dict[int, dict[str, Any]] = {}
    priors_path = directory / "roi_priors.json"
    if not priors_path.exists():
        failures.append("missing roi_priors.json")
    else:
        priors = json.loads(priors_path.read_text())
        if not priors.get("accepted", False):
            failures.append("roi_priors.json did not pass its held-out annotation gate")
        if priors.get("phase") != "final":
            failures.append("roi_priors.json is not a final 80/20 prior fit")
    for shard in (0, 1):
        meta_path = directory / f"roi_meta_shard{shard}.json"
        if not meta_path.exists():
            failures.append(f"missing {meta_path.name}")
        else:
            meta = json.loads(meta_path.read_text())
            expected_contract = {
                "laterality_dependent": "paired_side_centered",
                "side_independent": "target_declared",
            }
            if meta.get("version") != 3 or meta.get("window_contract") != expected_contract:
                failures.append(f"{meta_path.name} does not declare the v3 window contract")
            if not meta.get("deterministic_rebuild_passed", False):
                failures.append(f"{meta_path.name} did not pass deterministic rebuild")
        global_ids_path = (
            Path(arguments.ids_dir) / f"ids_shard{shard}.npy"
            if arguments.ids_dir
            else find_named_file(f"ids_shard{shard}.npy")
        )
        global_ids = np.load(global_ids_path, allow_pickle=True).astype(str)
        roi_ids_path = directory / f"roi_ids_shard{shard}.npy"
        if not roi_ids_path.exists():
            failures.append(f"missing {roi_ids_path.name}")
            continue
        roi_ids = np.load(roi_ids_path, allow_pickle=True).astype(str)
        if not np.array_equal(global_ids, roi_ids):
            failures.append(f"shard {shard} ROI IDs do not exactly equal global IDs")
        target_rates: dict[str, float] = {}
        target_masks: dict[str, np.ndarray] = {}
        for target in TARGET_SPECS:
            data_path = directory / f"roi_{target}_shard{shard}.npy"
            mask_path = directory / f"roi_{target}_masks_shard{shard}.npy"
            if not data_path.exists() or not mask_path.exists():
                failures.append(f"missing target files for {target} shard {shard}")
                continue
            data = np.load(data_path, mmap_mode="r")
            mask = np.load(mask_path, mmap_mode="r")
            expected_shape = (len(global_ids), WINDOWS_PER_TARGET, 3, IMAGE_SIZE, IMAGE_SIZE)
            if data.shape != expected_shape or data.dtype != np.uint8:
                failures.append(
                    f"{data_path.name} shape/dtype is {data.shape}/{data.dtype}, "
                    f"expected {expected_shape}/uint8"
                )
            if mask.shape != (len(global_ids), WINDOWS_PER_TARGET) or mask.dtype != np.uint8:
                failures.append(f"{mask_path.name} shape/dtype is {mask.shape}/{mask.dtype}")
            if not np.isin(np.unique(mask), [0, 1]).all():
                failures.append(f"{mask_path.name} is not binary")
            invalid_pixels_nonzero = False
            for start in range(0, len(mask), 8):
                stop = min(start + 8, len(mask))
                invalid = np.asarray(mask[start:stop]) == 0
                if invalid.any() and np.asarray(data[start:stop])[invalid].any():
                    invalid_pixels_nonzero = True
                    break
            if invalid_pixels_nonzero:
                failures.append(f"{data_path.name} contains nonzero pixels under a false mask")
            target_masks[target] = np.asarray(mask)
        provenance_path = directory / f"roi_provenance_shard{shard}.csv.gz"
        if not provenance_path.exists():
            failures.append(f"missing {provenance_path.name}")
            provenance_rows, padding_values, seen_laterality = 0, [], {}
        else:
            provenance_rows, padding_values, seen_laterality = 0, [], {}
            with gzip.open(provenance_path, "rt", newline="") as stream:
                reader = csv.DictReader(stream)
                missing_fields = set(PROVENANCE_FIELDS).difference(reader.fieldnames or [])
                if missing_fields:
                    failures.append(
                        f"{provenance_path.name} missing fields {sorted(missing_fields)}"
                    )
                for row in reader:
                    provenance_rows += 1
                    seen_laterality.setdefault(row["StudyInstanceUID"], row["laterality"])
                    if row["valid"] == "1":
                        padding_values.append(float(row["padding_fraction"]))
                        sop_uids = [row[f"sop_uid_{index}"] for index in range(3)]
                        positions = [float(row[f"slice_position_{index}_mm"]) for index in range(3)]
                        if len(set(sop_uids)) != 3:
                            failures.append(f"duplicate SOP UID in valid triplet shard {shard}")
                        if not positions[0] < positions[1] < positions[2]:
                            failures.append(f"non-monotonic slice positions in valid triplet shard {shard}")
                        expected_strategy = (
                            "paired_side_centered"
                            if TARGET_SPECS[row["target"]]["laterality_required"]
                            else _side_independent_window_strategy(TARGET_SPECS[row["target"]])
                        )
                        if row.get("window_strategy") != expected_strategy:
                            failures.append(
                                f"wrong window strategy for {row['target']} shard {shard}"
                            )
                        if expected_strategy == "paired_side_centered" and row.get(
                            "candidate_laterality"
                        ) not in {"L", "R"}:
                            failures.append(
                                f"missing paired-side candidate for {row['target']} shard {shard}"
                            )
            expected_rows = len(global_ids) * len(TARGET_SPECS) * WINDOWS_PER_TARGET
            if provenance_rows != expected_rows:
                failures.append(f"{provenance_path.name} has {provenance_rows}, expected {expected_rows}")
        known = np.asarray([seen_laterality.get(study, "UNKNOWN") in {"L", "R"} for study in global_ids])
        known_laterality_rate = float(known.mean()) if len(known) else 0.0
        for target, mask in target_masks.items():
            target_rates[target] = float((mask.sum(1) >= 2).mean())
        p95_padding = float(np.percentile(padding_values, 95)) if padding_values else 1.0
        if p95_padding > PADDING_LIMIT + 1e-12:
            failures.append(f"shard {shard} p95 padding {p95_padding:.3f} exceeds 0.20")
        for target, rate in target_rates.items():
            if rate < 0.90:
                failures.append(f"shard {shard} {target} two-window coverage is {rate:.3f}, below 0.90")
        reports[shard] = {
            "study_count": len(global_ids),
            "target_two_window_coverage": target_rates,
            "known_laterality_rate": known_laterality_rate,
            "provenance_rows": provenance_rows,
            "p95_padding_fraction": p95_padding,
        }
    result = {"passed": not failures, "failures": failures, "shards": reports}
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"ROI cache validation failed with {len(failures)} issue(s)")


def command_smoke(arguments: argparse.Namespace) -> None:
    """Build the supplied example study twice and render all target windows."""
    import matplotlib.pyplot as plt

    dicom_path = Path(arguments.dicom_path)
    if not dicom_path.is_file() or dicom_path.suffix.lower() != ".dcm":
        raise FileNotFoundError(f"example DICOM does not exist: {dicom_path}")
    study_uid = dicom_path.parent.parent.name
    competition = find_competition_root(arguments.competition_root)
    frame = pd.read_csv(competition / "train_series.csv")
    frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
    frame["SeriesInstanceUID"] = frame.SeriesInstanceUID.astype(str)
    rows = frame.loc[frame.StudyInstanceUID == study_uid].to_dict("records")
    if not rows:
        raise ValueError(f"example study {study_uid} is absent from train_series.csv")
    priors = json.loads(Path(arguments.priors).read_text())
    first = build_study(study_uid, rows, competition / "train_series", priors)
    second = build_study(study_uid, rows, competition / "train_series", priors)
    for target in TARGET_SPECS:
        if not np.array_equal(first[0][target], second[0][target]):
            raise RuntimeError(f"smoke rebuild differs for {target}")
        if not np.array_equal(first[1][target], second[1][target]):
            raise RuntimeError(f"smoke mask rebuild differs for {target}")
    figure, axes = plt.subplots(len(TARGET_SPECS), 5, figsize=(16, 15))
    provenance_by_target = defaultdict(list)
    for record in first[2]:
        provenance_by_target[record["target"]].append(record)
    for target_index, (target, specification) in enumerate(TARGET_SPECS.items()):
        records = sorted(provenance_by_target[target], key=lambda item: int(item["window_index"]))
        context_axis = axes[target_index, 0]
        valid_record = next((item for item in records if int(item["valid"])), None)
        if valid_record:
            geometry = read_series_geometry(
                study_uid,
                next(
                    row
                    for row in rows
                    if str(row["SeriesInstanceUID"])
                    == valid_record["SeriesInstanceUID"]
                ),
                competition / "train_series",
            )
            center_slice = geometry.slices[geometry.nearest_slice_index([
                valid_record["center_x_mm"], valid_record["center_y_mm"], valid_record["center_z_mm"],
            ])]
            context = _scale_uint8(_read_pixels(center_slice.path), *intensity_limits(geometry))
            row_px, column_px = center_slice.patient_to_pixel([
                valid_record["center_x_mm"], valid_record["center_y_mm"], valid_record["center_z_mm"],
            ])
            context_axis.imshow(context, cmap="gray")
            context_axis.plot(column_px, row_px, "r+", markersize=12, markeredgewidth=2)
        else:
            context_axis.text(0.5, 0.5, "no valid context", ha="center", va="center")
        context_axis.set_title(f"{specification['name']} context")
        context_axis.axis("off")
        for window in range(WINDOWS_PER_TARGET):
            axis = axes[target_index, window + 1]
            axis.imshow(first[0][target][window, 1], cmap="gray", vmin=0, vmax=255)
            state = "valid" if first[1][target][window] else records[window]["failure_reason"]
            candidate = records[window].get("candidate_laterality", "")
            axis.set_title(f"window {window} [{candidate}]: {state}", fontsize=8)
            axis.axis("off")
    figure.tight_layout()
    output = Path(arguments.output)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    log(f"smoke test and deterministic rebuild passed; wrote {output}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select-pilot")
    select.add_argument("--competition-root")
    select.add_argument("--output-dir", default="/kaggle/working/roi_pilot")
    select.set_defaults(function=command_select_pilot)

    cohort = subparsers.add_parser("select-cohort")
    cohort.add_argument("--pilot-manifest", required=True)
    cohort.add_argument("--competition-root")
    cohort.add_argument("--output-dir", default="/kaggle/working/roi_annotations")
    cohort.set_defaults(function=command_select_cohort)

    fit = subparsers.add_parser("fit-priors")
    fit.add_argument("--phase", choices=("pilot", "final"), required=True)
    fit.add_argument("--annotations", required=True)
    fit.add_argument("--manifest")
    fit.add_argument("--output", required=True)
    fit.add_argument("--competition-root")
    fit.set_defaults(function=command_fit_priors)

    build = subparsers.add_parser("build")
    build.add_argument("--shard", type=int, choices=(0, 1), required=True)
    build.add_argument("--priors", required=True)
    build.add_argument("--competition-root")
    build.add_argument("--ids-path")
    build.add_argument("--output-dir", default="/kaggle/working")
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--limit", type=int, default=0)
    build.set_defaults(function=command_build)

    pilot_build = subparsers.add_parser("build-pilot")
    pilot_build.add_argument("--manifest")
    pilot_build.add_argument("--priors", required=True)
    pilot_build.add_argument("--competition-root")
    pilot_build.add_argument("--output-dir", default="/kaggle/working/roi_pilot/cache")
    pilot_build.add_argument("--workers", type=int, default=4)
    pilot_build.set_defaults(function=command_build_pilot)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--cache-dir", required=True)
    validate.add_argument("--ids-dir")
    validate.set_defaults(function=command_validate)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--dicom-path", required=True)
    smoke.add_argument("--priors", required=True)
    smoke.add_argument("--competition-root")
    smoke.add_argument("--output", default="/kaggle/working/roi_smoke.png")
    smoke.set_defaults(function=command_smoke)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_arguments()
    options.function(options)
