"""Build and validate anatomy-specific Arm C ROI caches from raw DICOMs.

Typical Kaggle workflow::

    # 1. Select the label-free twenty-study pilot.
    python build_roi_cache.py select-pilot --output-dir /kaggle/working/roi_pilot

    # 2. After annotating pilot_annotations.csv, run the LOO gate.
    python build_roi_cache.py fit-priors --phase pilot \
        --annotations /kaggle/working/roi_pilot/pilot_annotations.csv \
        --output /kaggle/working/roi_pilot/pilot_priors.json

    # 3. After expanding to 100 annotations, freeze 80/20 and fit final priors.
    python build_roi_cache.py fit-priors --phase final \
        --annotations /kaggle/input/my-annotations/roi_annotations.csv \
        --output /kaggle/working/roi_priors.json

    # 4. Run as two CPU Kaggle jobs.
    python build_roi_cache.py build --shard 0 --priors .../roi_priors.json
    python build_roi_cache.py build --shard 1 --priors .../roi_priors.json

The authoritative row order is read from ``ids_shard<S>.npy``. One corrupt
study, missing slot, unknown laterality, or out-of-FOV crop produces zero data,
a false mask, and a provenance failure reason; it never terminates the shard.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
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
    centerline_offset_to_patient,
    crop_bounds,
    crop_with_padding,
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
    "row0",
    "row1",
    "column0",
    "column1",
    "padding_fraction",
    "intensity_low",
    "intensity_high",
    "laterality",
    "laterality_source",
    "valid",
    "failure_reason",
)

cv2.setNumThreads(1)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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


def selected_series_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for specification in TARGET_SPECS.values():
        for plane, fluid in specification["slots"]:
            row = pick_series(
                rows,
                plane,
                fluid,
                allow_fluid_fallback=specification["allow_fluid_fallback"],
            )
            if row is not None:
                selected.setdefault(str(row["SeriesInstanceUID"]), row)
    return selected


def load_study_geometry(
    study_uid: str,
    rows: list[dict[str, Any]],
    series_root: Path,
) -> tuple[dict[str, SeriesGeometry], dict[str, str]]:
    geometries: dict[str, SeriesGeometry] = {}
    failures: dict[str, str] = {}
    for series_uid, row in selected_series_rows(rows).items():
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
        "valid": 0,
        "failure_reason": failure_reason,
    })
    return result


def _landmark_point(
    landmark: str,
    priors: dict[str, Any],
    origin: np.ndarray,
    laterality: str,
) -> np.ndarray:
    offset = priors["landmarks"][landmark]["median_offset_mm"]
    if laterality == "UNKNOWN":
        return centerline_offset_to_patient(offset, origin)
    return canonical_to_patient(offset, origin, laterality)


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
        if specification["laterality_required"] and laterality == "UNKNOWN":
            for window_index in range(WINDOWS_PER_TARGET):
                provenance.append(_blank_provenance(
                    study_uid, target, window_index, laterality, laterality_source,
                    "unknown_laterality",
                ))
            continue
        try:
            center = _landmark_point(specification["landmark"], priors, origin, laterality)
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
                    provenance.append(_blank_provenance(
                        study_uid, target, base_window + local_index, laterality,
                        laterality_source, "missing_required_series", None,
                    ))
                continue
            series_uid = str(row["SeriesInstanceUID"])
            series = geometries.get(series_uid)
            if series is None:
                reason = series_failures.get(series_uid, "invalid_required_series")
                for local_index in range(2):
                    provenance.append(_blank_provenance(
                        study_uid, target, base_window + local_index, laterality,
                        laterality_source, reason, row,
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
                    provenance.append(_blank_provenance(
                        study_uid, target, base_window + local_index, laterality,
                        laterality_source,
                        f"intensity_failed:{type(limits).__name__}:{limits}"[:500], row,
                    ))
                continue
            center_index = series.nearest_slice_index(center)
            for local_index, indices in enumerate(triplet_indices(center_index, len(series.slices))):
                window_index = base_window + local_index
                if indices is None:
                    provenance.append(_blank_provenance(
                        study_uid, target, window_index, laterality,
                        laterality_source, "triplet_crosses_series_boundary", row,
                    ))
                    continue
                middle = series.slices[indices[1]]
                try:
                    bounds = crop_bounds(
                        middle, center, plane, specification["crop_mm"], laterality,
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
                            item, center, plane, specification["crop_mm"], laterality,
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
                        "row0": bounds[0],
                        "row1": bounds[1],
                        "column0": bounds[2],
                        "column1": bounds[3],
                        "padding_fraction": max(triplet_padding),
                        "intensity_low": limits[0],
                        "intensity_high": limits[1],
                        "laterality": laterality,
                        "laterality_source": laterality_source,
                        "valid": 1,
                        "failure_reason": "",
                    })
                except Exception as error:
                    provenance.append(_blank_provenance(
                        study_uid, target, window_index, laterality, laterality_source,
                        f"crop_failed:{type(error).__name__}:{error}"[:500], row,
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
    known_mask = np.asarray([seen.get(study, "UNKNOWN") in {"L", "R"} for study in ids])
    coverage = {}
    for target, mask in masks.items():
        eligible = known_mask if TARGET_SPECS[target]["laterality_required"] else np.ones(count, dtype=bool)
        coverage[target] = (
            float((np.asarray(mask)[eligible].sum(axis=1) >= 2).mean())
            if eligible.any()
            else 0.0
        )
    meta = {
        "version": 1,
        "shard": arguments.shard,
        "ids_source": str(ids_path),
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


def _diversity_bin(value: float | None, width: float) -> str:
    return "unknown" if value is None else str(round(value / width) * width)


def command_select_pilot(arguments: argparse.Namespace) -> None:
    competition = find_competition_root(arguments.competition_root)
    frame = pd.read_csv(competition / "train_series.csv")
    frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
    frame["SeriesInstanceUID"] = frame.SeriesInstanceUID.astype(str)
    candidates: list[dict[str, Any]] = []
    needed = {("Coronal", 1), ("Axial", 1), ("Sagittal", 0), ("Coronal", 0), ("Sagittal", 1)}
    for study_uid, group in frame.groupby("StudyInstanceUID", sort=True):
        rows = group.to_dict("records")
        available = {
            (str(row.get("Anatomical_Plane", "")), int(float(row.get("Fluid_Sensitive", 0) or 0)))
            for row in rows
        }
        header_summaries = []
        for row in rows:
            directory = competition / "train_series" / study_uid / str(row["SeriesInstanceUID"])
            files = sorted(directory.glob("*.dcm"))
            if not files:
                continue
            try:
                header_summaries.append(_header_summary(files[len(files) // 2]))
            except Exception:
                continue
        if not header_summaries:
            continue
        laterality, laterality_source = resolve_laterality(
            summary["laterality"] for summary in header_summaries
        )
        summary = {
            "laterality": laterality,
            "laterality_source": laterality_source,
            "manufacturer": Counter(
                item["manufacturer"] for item in header_summaries
            ).most_common(1)[0][0],
            "field_strength": next(
                (item["field_strength"] for item in header_summaries if item["field_strength"] is not None),
                None,
            ),
            "pixel_spacing": next(
                (item["pixel_spacing"] for item in header_summaries if item["pixel_spacing"] is not None),
                None,
            ),
            "slice_thickness": next(
                (item["slice_thickness"] for item in header_summaries if item["slice_thickness"] is not None),
                None,
            ),
        }
        summary.update({
            "StudyInstanceUID": study_uid,
            "complete_preferred_planes": needed.issubset(available),
            "available_slots": len(needed.intersection(available)),
            "manufacturer_bin": summary["manufacturer"].upper(),
            "field_strength_bin": _diversity_bin(summary["field_strength"], 0.5),
            "pixel_spacing_bin": _diversity_bin(summary["pixel_spacing"], 0.1),
            "slice_thickness_bin": _diversity_bin(summary["slice_thickness"], 0.5),
        })
        candidates.append(summary)
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    selected: list[dict[str, Any]] = []
    used: Counter[tuple[str, str]] = Counter()

    def choose(pool: list[dict[str, Any]], number: int) -> None:
        for _ in range(number):
            if not pool:
                raise RuntimeError("not enough studies to construct the annotation pilot")
            def score(item: dict[str, Any]) -> tuple[float, str]:
                categories = (
                    ("manufacturer", item["manufacturer_bin"]),
                    ("field", item["field_strength_bin"]),
                    ("spacing", item["pixel_spacing_bin"]),
                    ("thickness", item["slice_thickness_bin"]),
                )
                novelty = sum(1.0 / (1 + used[category]) for category in categories)
                side_count = sum(existing["laterality"] == item["laterality"] for existing in selected)
                side_bonus = 1.0 / (1 + side_count) if item["laterality"] in {"L", "R"} else -1.0
                return novelty + side_bonus, item["StudyInstanceUID"]
            winner = max(pool, key=score)
            pool.remove(winner)
            selected.append(winner)
            for category in (
                ("manufacturer", winner["manufacturer_bin"]),
                ("field", winner["field_strength_bin"]),
                ("spacing", winner["pixel_spacing_bin"]),
                ("thickness", winner["slice_thickness_bin"]),
            ):
                used[category] += 1

    def choose_balanced(pool: list[dict[str, Any]], number: int) -> None:
        before = {item["StudyInstanceUID"] for item in selected}
        per_side = number // 2
        for side in ("L", "R"):
            side_pool = [item for item in pool if item["laterality"] == side]
            choose(side_pool, min(per_side, len(side_pool)))
        newly_selected = {
            item["StudyInstanceUID"] for item in selected
            if item["StudyInstanceUID"] not in before
        }
        pool[:] = [item for item in pool if item["StudyInstanceUID"] not in newly_selected]
        remaining = number - len(newly_selected)
        if remaining:
            choose(pool, remaining)

    complete = [item for item in candidates if item["complete_preferred_planes"]]
    available_by_study = {
        str(study_uid): {
            (str(row.get("Anatomical_Plane", "")), int(float(row.get("Fluid_Sensitive", 0) or 0)))
            for row in group.to_dict("records")
        }
        for study_uid, group in frame.groupby("StudyInstanceUID", sort=False)
    }
    imperfect = [
        item
        for item in candidates
        if not item["complete_preferred_planes"]
        and item["available_slots"] >= 3
        and any(plane == "Coronal" for plane, _ in available_by_study[item["StudyInstanceUID"]])
        and any(plane == "Axial" for plane, _ in available_by_study[item["StudyInstanceUID"]])
        and ("Sagittal", 1) in available_by_study[item["StudyInstanceUID"]]
    ]
    choose_balanced(complete, 16)
    choose_balanced(imperfect, 4)
    selected.sort(key=lambda item: item["StudyInstanceUID"])
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(output / "pilot_manifest.csv", index=False)
    np.save(output / "pilot_ids.npy", np.asarray([item["StudyInstanceUID"] for item in selected]))
    annotation_columns = [
        "StudyInstanceUID", "landmark", "SeriesInstanceUID", "SOPInstanceUID",
        "row_px", "column_px", "patient_x_mm", "patient_y_mm", "patient_z_mm",
        "laterality", "annotator", "qc_status",
    ]
    records = [
        {"StudyInstanceUID": item["StudyInstanceUID"], "landmark": landmark, "laterality": item["laterality"]}
        for item in selected for landmark in LANDMARKS
    ]
    pd.DataFrame(records, columns=annotation_columns).to_csv(output / "pilot_annotations.csv", index=False)
    log(f"wrote deterministic 20-study pilot to {output}")


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
            predicted = canonical_to_patient(predicted_offset, np.zeros(3), laterality[study])
            actual = canonical_to_patient(offsets[study][landmark], np.zeros(3), laterality[study])
            for target, specification in TARGET_SPECS.items():
                if specification["landmark"] == landmark:
                    target_results[target].append(
                        _coverage_for_offset(actual, predicted, laterality[study], target)
                    )
            if landmark == "medial_joint_line" and offsets[study][landmark][0] < 0:
                mirror_errors += 1
            if landmark == "lateral_joint_line" and offsets[study][landmark][0] > 0:
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
        shuffled = studies.copy()
        random.Random(SEED).shuffle(shuffled)
        fit_studies, audit_studies = shuffled[:80], shuffled[80:100]
        audit = _audit_coverage(audit_studies, fit_studies, offsets, laterality, leave_one_out=False)
        threshold = 0.95
    laterality_rate = float(np.mean([laterality.get(study) in {"L", "R"} for study in audit_studies]))
    accepted = (
        audit["minimum_target_coverage"] >= threshold
        and audit["mirror_errors"] == 0
        and laterality_rate >= 0.95
    )
    result = {
        "version": 1,
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
            target_masks[target] = np.asarray(mask)
        provenance_path = directory / f"roi_provenance_shard{shard}.csv.gz"
        if not provenance_path.exists():
            failures.append(f"missing {provenance_path.name}")
            provenance_rows, padding_values, seen_laterality = 0, [], {}
        else:
            provenance_rows, padding_values, seen_laterality = 0, [], {}
            with gzip.open(provenance_path, "rt", newline="") as stream:
                for row in csv.DictReader(stream):
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
            expected_rows = len(global_ids) * len(TARGET_SPECS) * WINDOWS_PER_TARGET
            if provenance_rows != expected_rows:
                failures.append(f"{provenance_path.name} has {provenance_rows}, expected {expected_rows}")
        known = np.asarray([seen_laterality.get(study, "UNKNOWN") in {"L", "R"} for study in global_ids])
        known_laterality_rate = float(known.mean()) if len(known) else 0.0
        if known_laterality_rate < 0.95:
            failures.append(f"shard {shard} known laterality rate is {known_laterality_rate:.3f}, below 0.95")
        for target, mask in target_masks.items():
            eligible = (
                known
                if TARGET_SPECS[target]["laterality_required"]
                else np.ones(len(global_ids), dtype=bool)
            )
            target_rates[target] = float((mask[eligible].sum(1) >= 2).mean()) if eligible.any() else 0.0
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
            axis.set_title(f"window {window}: {state}", fontsize=8)
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

    fit = subparsers.add_parser("fit-priors")
    fit.add_argument("--phase", choices=("pilot", "final"), required=True)
    fit.add_argument("--annotations", required=True)
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
