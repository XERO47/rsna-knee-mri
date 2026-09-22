#!/usr/bin/env python3
"""Reproducible, read-only EDA for the RSNA knee MRI competition data.

The script never reads pixel data except for a small deterministic validation
sample. It inspects every CSV row and, by default, every series directory and
one representative DICOM header per series. A smaller, deterministic subset of
series receives slice-by-slice geometry checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pydicom


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
UID_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
SEED = 20260919
T = TypeVar("T")
GZIP_OPTIONS = {"method": "gzip", "compresslevel": 6, "mtime": 0}


def parse_args() -> argparse.Namespace:
    project_parent = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_parent / "data-root",
        help="Directory containing train.csv and train_series/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "eda_outputs",
    )
    parser.add_argument(
        "--dicom-series-limit",
        type=int,
        default=0,
        help="Representative headers to read; 0 means every series.",
    )
    parser.add_argument(
        "--geometry-series-sample",
        type=int,
        default=300,
        help="Series for which every slice header is checked.",
    )
    parser.add_argument(
        "--pixel-samples",
        type=int,
        default=64,
        help="Representative DICOMs whose pixels are decoded; 0 disables.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not generate PNG plots.",
    )
    return parser.parse_args()


def require_layout(root: Path) -> None:
    required = [
        "train.csv",
        "train_series.csv",
        "test.csv",
        "test_series.csv",
        "sample_submission.csv",
        "train_series",
        "test_series",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing from {root}: {', '.join(missing)}")


def clean_text(value: Any, default: str = "unknown") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def scalar_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        if isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def list_float(value: Any, expected: int) -> list[float] | None:
    if value is None:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if len(result) == expected else None


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def describe(series: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0}
    quantiles = numeric.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    result: dict[str, float | int | None] = {
        "count": int(numeric.size),
        "min": float(numeric.min()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std()) if numeric.size > 1 else None,
        "max": float(numeric.max()),
    }
    result.update({f"p{int(q * 100):02d}": float(v) for q, v in quantiles.items()})
    return result


def wilson_interval(positives: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = positives / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return center - half, center + half


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "column": column,
            "dtype": str(frame[column].dtype),
            "missing": int(frame[column].isna().sum()),
            "unique_non_null": int(frame[column].nunique(dropna=True)),
        }
        for column in frame.columns
    ]


def analyze_csvs(root: Path, output: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    dtype = {"StudyInstanceUID": "string", "SeriesInstanceUID": "string"}
    train = pd.read_csv(root / "train.csv", dtype=dtype)
    train_series = pd.read_csv(root / "train_series.csv", dtype=dtype)
    test = pd.read_csv(root / "test.csv", dtype=dtype)
    test_series = pd.read_csv(root / "test_series.csv", dtype=dtype)
    submission = pd.read_csv(root / "sample_submission.csv", dtype=dtype)

    frames = {
        "train": train,
        "train_series": train_series,
        "test": test,
        "test_series": test_series,
        "sample_submission": submission,
    }
    schemas = []
    for name, frame in frames.items():
        for row in dataframe_schema(frame):
            schemas.append({"file": f"{name}.csv", **row})
    pd.DataFrame(schemas).to_csv(output / "csv_schema.csv", index=False)

    labeled = train.dropna(subset=TARGETS)
    prevalence_rows = []
    for target in TARGETS:
        values = labeled[target].dropna().astype(int)
        positives = int(values.sum())
        lower, upper = wilson_interval(positives, len(values))
        prevalence_rows.append(
            {
                "target": target,
                "labeled": int(len(values)),
                "positive": positives,
                "negative": int(len(values) - positives),
                "prevalence": positives / len(values) if len(values) else math.nan,
                "wilson_95_low": lower,
                "wilson_95_high": upper,
            }
        )
    prevalence = pd.DataFrame(prevalence_rows)
    prevalence.to_csv(output / "target_prevalence.csv", index=False)
    correlation = labeled[TARGETS].astype(float).corr()
    correlation.to_csv(output / "target_phi_correlations.csv")
    co_positive = pd.DataFrame(
        np.asarray(labeled[TARGETS].fillna(0).astype(int).T) @ np.asarray(labeled[TARGETS].fillna(0).astype(int)),
        index=TARGETS,
        columns=TARGETS,
    )
    co_positive.to_csv(output / "target_co_positive_counts.csv")

    series_per_study = (
        train_series.groupby("StudyInstanceUID", observed=True)
        .size()
        .rename("series_count")
        .reset_index()
    )
    series_per_study.to_csv(output / "study_series_counts.csv", index=False)

    slot_rows = []
    study_total = train_series["StudyInstanceUID"].nunique()
    for plane in ["Axial", "Coronal", "Sagittal"]:
        for fluid in [0, 1]:
            subset = train_series[
                (train_series["Anatomical_Plane"] == plane)
                & (train_series["Fluid_Sensitive"] == fluid)
            ]
            studies = subset["StudyInstanceUID"].nunique()
            slot_rows.append(
                {
                    "plane": plane,
                    "fluid_sensitive": fluid,
                    "series": int(len(subset)),
                    "studies": int(studies),
                    "study_coverage": studies / study_total,
                }
            )
    slot_coverage = pd.DataFrame(slot_rows)
    slot_coverage.to_csv(output / "slot_coverage.csv", index=False)

    combo = (
        train_series.groupby(
            ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"],
            dropna=False,
            observed=True,
        )
        .size()
        .rename("series")
        .reset_index()
    )
    combo.to_csv(output / "series_plane_sequence_counts.csv", index=False)

    reports = train["Report"].fillna("").astype(str)
    normalized = reports.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    report_stats = pd.DataFrame(
        {
            "StudyInstanceUID": train["StudyInstanceUID"],
            "characters": reports.str.len(),
            "words": reports.str.split().str.len(),
            "lines": reports.str.count("\n") + 1,
            "normalized_md5": normalized.map(lambda value: hashlib.md5(value.encode("utf-8")).hexdigest()),
        }
    )
    report_stats.to_csv(output / "report_statistics.csv.gz", index=False, compression=GZIP_OPTIONS)

    language_probes = {
        "English": r"\btear\b|\bmeniscus\b|\beffusion\b",
        "Spanish_or_Portuguese": r"\brotura\b|\bmenisco\b|\bderrame\b",
        "Italian": r"\blesione\b|\bmenisco\b|\bversamento\b",
        "French": r"\bd.chirure\b|\bm.nisque\b|\b.panchement\b",
        "German": r"\briss\b|\bmeniskus\b|\berguss\b",
        "Turkish": r"\byirtik\b|\bmenisk.s\b|\befuzyon\b",
    }
    lowered = reports.str.lower()
    probe_rows = []
    for name, pattern in language_probes.items():
        hits = lowered.str.contains(pattern, regex=True, na=False)
        probe_rows.append({"probe": name, "reports_matching": int(hits.sum()), "share": float(hits.mean())})
    language_probe_df = pd.DataFrame(probe_rows)
    language_probe_df.to_csv(output / "report_language_fingerprint_counts.csv", index=False)

    train_ids = set(train["StudyInstanceUID"])
    series_train_ids = set(train_series["StudyInstanceUID"])
    test_ids = set(test["StudyInstanceUID"])
    series_test_ids = set(test_series["StudyInstanceUID"])
    submission_ids = set(submission["StudyInstanceUID"])
    uid_columns = [
        ("train.StudyInstanceUID", train["StudyInstanceUID"]),
        ("train_series.StudyInstanceUID", train_series["StudyInstanceUID"]),
        ("train_series.SeriesInstanceUID", train_series["SeriesInstanceUID"]),
        ("test.StudyInstanceUID", test["StudyInstanceUID"]),
        ("test_series.SeriesInstanceUID", test_series["SeriesInstanceUID"]),
    ]

    summary = {
        "tables": {name: {"rows": len(frame), "columns": len(frame.columns)} for name, frame in frames.items()},
        "train_studies": int(train["StudyInstanceUID"].nunique()),
        "train_series": int(train_series["SeriesInstanceUID"].nunique()),
        "test_studies": int(test["StudyInstanceUID"].nunique()),
        "test_series": int(test_series["SeriesInstanceUID"].nunique()),
        "fully_labeled_studies": int(len(labeled)),
        "partially_labeled_studies": int(train[TARGETS].notna().any(axis=1).sum() - len(labeled)),
        "unlabeled_studies": int(train[TARGETS].isna().all(axis=1).sum()),
        "invalid_label_values": {
            target: sorted(set(train[target].dropna()) - {0.0, 1.0}) for target in TARGETS
        },
        "duplicate_train_study_ids": int(train["StudyInstanceUID"].duplicated().sum()),
        "duplicate_series_ids": int(train_series["SeriesInstanceUID"].duplicated().sum()),
        "duplicate_normalized_reports": int(report_stats["normalized_md5"].duplicated(keep=False).sum()),
        "duplicate_normalized_report_groups": int(
            (report_stats["normalized_md5"].value_counts() > 1).sum()
        ),
        "report_characters": describe(report_stats["characters"]),
        "report_words": describe(report_stats["words"]),
        "report_lines": describe(report_stats["lines"]),
        "empty_reports": int((report_stats["characters"] == 0).sum()),
        "series_per_study": describe(series_per_study["series_count"]),
        "fluid_equals_fat_suppression_all_rows": bool(
            train_series["Fluid_Sensitive"].equals(train_series["Fat_Suppression"])
        ),
        "train_ids_missing_from_series_csv": sorted(train_ids - series_train_ids),
        "series_csv_ids_missing_from_train_csv": sorted(series_train_ids - train_ids),
        "test_ids_missing_from_series_csv": sorted(test_ids - series_test_ids),
        "series_test_ids_missing_from_test_csv": sorted(series_test_ids - test_ids),
        "submission_id_symmetric_difference": sorted(test_ids ^ submission_ids),
        "submission_columns_match_targets": list(submission.columns) == ["StudyInstanceUID", *TARGETS],
        "uid_format_failures": {
            name: int((~values.fillna("").str.match(UID_RE)).sum()) for name, values in uid_columns
        },
    }
    return summary, {
        **frames,
        "labeled": labeled,
        "prevalence": prevalence,
        "correlation": correlation,
        "slot_coverage": slot_coverage,
        "series_per_study": series_per_study,
        "report_stats": report_stats,
    }


def series_directories(root: Path, split: str) -> list[tuple[str, str, Path]]:
    base = root / f"{split}_series"
    rows: list[tuple[str, str, Path]] = []
    for study_dir in sorted(base.iterdir()):
        if not study_dir.is_dir():
            continue
        for series_dir in sorted(study_dir.iterdir()):
            if series_dir.is_dir():
                rows.append((study_dir.name, series_dir.name, series_dir))
    return rows


def deterministic_subset(items: list[T], limit: int, salt: str) -> list[T]:
    if limit <= 0 or limit >= len(items):
        return items
    ranked = sorted(
        items,
        key=lambda item: hashlib.sha256(f"{SEED}:{salt}:{item}".encode("utf-8")).digest(),
    )
    return ranked[:limit]


def representative_header_record(
    split: str,
    path_study_uid: str,
    path_series_uid: str,
    series_dir: Path,
) -> tuple[dict[str, Any], list[Path]]:
    entries = []
    ignored_files = []
    total_bytes = 0
    with os.scandir(series_dir) as iterator:
        for entry in iterator:
            if not entry.is_file(follow_symlinks=False):
                continue
            if entry.name.lower().endswith(".dcm"):
                entries.append(Path(entry.path))
                try:
                    total_bytes += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    pass
            else:
                ignored_files.append(entry.name)
    entries.sort()
    record: dict[str, Any] = {
        "split": split,
        "path_study_uid": path_study_uid,
        "path_series_uid": path_series_uid,
        "slice_files": len(entries),
        "series_bytes": total_bytes,
        "non_dicom_files": "|".join(sorted(ignored_files)),
        "header_read_ok": False,
        "header_error": "",
    }
    if not entries:
        record["header_error"] = "no_dicom_files"
        return record, entries

    representative = entries[len(entries) // 2]
    try:
        ds = pydicom.dcmread(representative, stop_before_pixels=True, force=True)
    except Exception as exc:
        record["header_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return record, entries

    spacing = list_float(getattr(ds, "PixelSpacing", None), 2)
    orientation = list_float(getattr(ds, "ImageOrientationPatient", None), 6)
    position = list_float(getattr(ds, "ImagePositionPatient", None), 3)
    transfer = clean_text(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None))
    orientation_plane = "unknown"
    if orientation is not None:
        normal = np.abs(
            np.cross(
                np.asarray(orientation[:3], dtype=float),
                np.asarray(orientation[3:], dtype=float),
            )
        )
        orientation_plane = ["Sagittal", "Coronal", "Axial"][int(np.argmax(normal))]
    record.update(
        {
            "header_read_ok": True,
            "dicom_study_uid": clean_text(getattr(ds, "StudyInstanceUID", None)),
            "dicom_series_uid": clean_text(getattr(ds, "SeriesInstanceUID", None)),
            "representative_sop_uid": clean_text(getattr(ds, "SOPInstanceUID", None)),
            "modality": clean_text(getattr(ds, "Modality", None)),
            "manufacturer": clean_text(getattr(ds, "Manufacturer", None)),
            "model": clean_text(getattr(ds, "ManufacturerModelName", None)),
            "field_strength_t": scalar_float(getattr(ds, "MagneticFieldStrength", None)),
            "body_part": clean_text(getattr(ds, "BodyPartExamined", None)),
            "series_description": clean_text(getattr(ds, "SeriesDescription", None)),
            "protocol_name": clean_text(getattr(ds, "ProtocolName", None)),
            "institution_name": clean_text(getattr(ds, "InstitutionName", None)),
            "station_name": clean_text(getattr(ds, "StationName", None)),
            "software_versions": clean_text(getattr(ds, "SoftwareVersions", None)),
            "receive_coil_name": clean_text(getattr(ds, "ReceiveCoilName", None)),
            "scanning_sequence": clean_text(getattr(ds, "ScanningSequence", None)),
            "sequence_variant": clean_text(getattr(ds, "SequenceVariant", None)),
            "scan_options": clean_text(getattr(ds, "ScanOptions", None)),
            "mr_acquisition_type": clean_text(getattr(ds, "MRAcquisitionType", None)),
            "repetition_time_ms": scalar_float(getattr(ds, "RepetitionTime", None)),
            "echo_time_ms": scalar_float(getattr(ds, "EchoTime", None)),
            "flip_angle_deg": scalar_float(getattr(ds, "FlipAngle", None)),
            "echo_train_length": scalar_float(getattr(ds, "EchoTrainLength", None)),
            "pixel_bandwidth": scalar_float(getattr(ds, "PixelBandwidth", None)),
            "patient_sex": clean_text(getattr(ds, "PatientSex", None)),
            "patient_age": clean_text(getattr(ds, "PatientAge", None)),
            "has_study_date": bool(clean_text(getattr(ds, "StudyDate", None), default="")),
            "image_laterality": clean_text(getattr(ds, "ImageLaterality", None)),
            "laterality": clean_text(getattr(ds, "Laterality", None)),
            "rows": int(getattr(ds, "Rows", 0) or 0),
            "columns": int(getattr(ds, "Columns", 0) or 0),
            "pixel_spacing_row_mm": spacing[0] if spacing else math.nan,
            "pixel_spacing_column_mm": spacing[1] if spacing else math.nan,
            "slice_thickness_mm": scalar_float(getattr(ds, "SliceThickness", None)),
            "spacing_between_slices_mm": scalar_float(getattr(ds, "SpacingBetweenSlices", None)),
            "has_iop": orientation is not None,
            "has_ipp": position is not None,
            "orientation_plane": orientation_plane,
            "photometric": clean_text(getattr(ds, "PhotometricInterpretation", None)),
            "bits_allocated": int(getattr(ds, "BitsAllocated", 0) or 0),
            "bits_stored": int(getattr(ds, "BitsStored", 0) or 0),
            "pixel_representation": int(getattr(ds, "PixelRepresentation", -1)),
            "rescale_slope": scalar_float(getattr(ds, "RescaleSlope", None)),
            "rescale_intercept": scalar_float(getattr(ds, "RescaleIntercept", None)),
            "transfer_syntax_uid": transfer,
            "study_uid_matches_path": clean_text(getattr(ds, "StudyInstanceUID", None)) == path_study_uid,
            "series_uid_matches_path": clean_text(getattr(ds, "SeriesInstanceUID", None)) == path_series_uid,
        }
    )
    return record, entries


def geometry_record(
    split: str,
    study_uid: str,
    series_uid: str,
    files: list[Path],
) -> dict[str, Any]:
    positions = []
    normals = []
    missing_iop = 0
    missing_ipp = 0
    errors = 0
    for filename_index, path in enumerate(files):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            errors += 1
            continue
        iop = list_float(getattr(ds, "ImageOrientationPatient", None), 6)
        ipp = list_float(getattr(ds, "ImagePositionPatient", None), 3)
        if iop is None:
            missing_iop += 1
        if ipp is None:
            missing_ipp += 1
        if iop is None or ipp is None:
            continue
        row = np.asarray(iop[:3], dtype=float)
        column = np.asarray(iop[3:], dtype=float)
        normal = np.cross(row, column)
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm
        if normals and np.dot(normal, normals[0]) < 0:
            normal *= -1
        normals.append(normal)
        positions.append((filename_index, float(np.dot(np.asarray(ipp, dtype=float), normal))))

    orientation_max_angle = math.nan
    if normals:
        reference = normals[0]
        dots = np.clip(np.asarray(normals) @ reference, -1, 1)
        orientation_max_angle = float(np.degrees(np.arccos(dots)).max())
    physical = np.asarray([position for _, position in positions], dtype=float)
    diffs_filename = np.diff(physical)
    filename_order_monotonic = bool(
        len(diffs_filename) == 0 or np.all(diffs_filename > 0) or np.all(diffs_filename < 0)
    )
    sorted_positions = np.sort(physical)
    diffs_sorted = np.diff(sorted_positions)
    duplicate_positions = int(np.sum(np.abs(diffs_sorted) <= 0.1)) if len(diffs_sorted) else 0
    positive_diffs = diffs_sorted[diffs_sorted > 0.1]
    return {
        "split": split,
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "slice_files": len(files),
        "headers_read": len(files) - errors,
        "read_errors": errors,
        "missing_iop_slices": missing_iop,
        "missing_ipp_slices": missing_ipp,
        "projectable_slices": len(positions),
        "orientation_max_angle_deg": orientation_max_angle,
        "orientation_within_1_degree": bool(
            np.isfinite(orientation_max_angle) and orientation_max_angle <= 1.0
        ),
        "filename_order_is_physical_monotonic": filename_order_monotonic,
        "positions_within_0_1mm": duplicate_positions,
        "median_physical_slice_step_mm": float(np.median(positive_diffs)) if len(positive_diffs) else math.nan,
        "physical_span_mm": float(sorted_positions[-1] - sorted_positions[0]) if len(sorted_positions) > 1 else 0.0,
    }


def decode_pixel_sample(record: dict[str, Any], files: list[Path]) -> dict[str, Any]:
    result = {
        "split": record["split"],
        "StudyInstanceUID": record["path_study_uid"],
        "SeriesInstanceUID": record["path_series_uid"],
        "decode_ok": False,
        "error": "",
    }
    if not files:
        result["error"] = "no_dicom_files"
        return result
    path = files[len(files) // 2]
    try:
        ds = pydicom.dcmread(path, force=True)
        pixels = np.asarray(ds.pixel_array)
        physical = pixels.astype(np.float32)
        physical = physical * float(getattr(ds, "RescaleSlope", 1) or 1)
        physical = physical + float(getattr(ds, "RescaleIntercept", 0) or 0)
        result.update(
            {
                "decode_ok": True,
                "shape": "x".join(map(str, pixels.shape)),
                "dtype": str(pixels.dtype),
                "stored_min": float(np.min(pixels)),
                "stored_max": float(np.max(pixels)),
                "physical_p02": float(np.percentile(physical, 2)),
                "physical_p50": float(np.percentile(physical, 50)),
                "physical_p98": float(np.percentile(physical, 98)),
                "photometric": clean_text(getattr(ds, "PhotometricInterpretation", None)),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return result


def normalized_manufacturer(value: Any) -> str:
    text = clean_text(value).upper()
    if "SIEMENS" in text:
        return "Siemens"
    if text.startswith("GE") or "GENERAL ELECTRIC" in text:
        return "GE"
    if "PHILIPS" in text:
        return "Philips"
    if "TOSHIBA" in text or "CANON" in text:
        return "Canon/Toshiba"
    if "HITACHI" in text or "FUJIFILM" in text:
        return "Fujifilm/Hitachi"
    return "Unknown/other"


def derive_series_features(
    inventory: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    output: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    merged_parts = []
    for split in ("train", "test"):
        inv = inventory[inventory["split"] == split].copy()
        declared = frames[f"{split}_series"]
        merged_parts.append(
            inv.merge(
                declared,
                left_on=["path_study_uid", "path_series_uid"],
                right_on=["StudyInstanceUID", "SeriesInstanceUID"],
                how="left",
                validate="one_to_one",
            )
        )
    merged = pd.concat(merged_parts, ignore_index=True)
    merged["manufacturer_normalized"] = merged["manufacturer"].map(normalized_manufacturer)
    merged["row_fov_mm"] = merged["rows"] * merged["pixel_spacing_row_mm"]
    merged["column_fov_mm"] = merged["columns"] * merged["pixel_spacing_column_mm"]
    merged["declared_plane_matches_iop"] = merged["Anatomical_Plane"] == merged["orientation_plane"]
    merged.to_csv(output / "dicom_series_with_declared_metadata.csv.gz", index=False, compression=GZIP_OPTIONS)

    train = merged[merged["split"] == "train"].copy()
    depth = (
        train.groupby(["Anatomical_Plane", "Fluid_Sensitive"], observed=True)["slice_files"]
        .agg(
            series="count",
            min="min",
            p01=lambda x: x.quantile(0.01),
            p05=lambda x: x.quantile(0.05),
            p25=lambda x: x.quantile(0.25),
            median="median",
            p75=lambda x: x.quantile(0.75),
            p95=lambda x: x.quantile(0.95),
            p99=lambda x: x.quantile(0.99),
            max="max",
            mean="mean",
        )
        .reset_index()
    )
    depth.to_csv(output / "series_depth_by_slot.csv", index=False)

    geometry_rows = []
    for plane, group in train.groupby("Anatomical_Plane", observed=True):
        row: dict[str, Any] = {"plane": plane, "series": len(group)}
        for column in (
            "pixel_spacing_row_mm",
            "pixel_spacing_column_mm",
            "row_fov_mm",
            "column_fov_mm",
            "slice_thickness_mm",
            "spacing_between_slices_mm",
        ):
            values = group[column].dropna()
            for quantile in (0.05, 0.5, 0.95):
                row[f"{column}_p{int(quantile * 100):02d}"] = float(values.quantile(quantile))
        geometry_rows.append(row)
    pd.DataFrame(geometry_rows).to_csv(output / "geometry_by_plane.csv", index=False)
    (
        train["manufacturer_normalized"]
        .value_counts()
        .rename_axis("manufacturer_normalized")
        .reset_index(name="series")
        .to_csv(output / "manufacturer_normalized_counts.csv", index=False)
    )

    demographic_counts = {
        "patient_sex": train["patient_sex"].value_counts(dropna=False).to_dict(),
        "patient_age": train["patient_age"].value_counts(dropna=False).to_dict(),
        "study_date_present_series": int(train["has_study_date"].sum()),
    }
    return {
        "declared_metadata_missing_after_join": int(train["Anatomical_Plane"].isna().sum()),
        "declared_plane_matches_iop": int(train["declared_plane_matches_iop"].sum()),
        "declared_plane_mismatches_iop": int((~train["declared_plane_matches_iop"]).sum()),
        "row_fov_mm": describe(train["row_fov_mm"]),
        "column_fov_mm": describe(train["column_fov_mm"]),
        "protocol_parameters": {
            "repetition_time_ms": describe(train["repetition_time_ms"]),
            "echo_time_ms": describe(train["echo_time_ms"]),
            "flip_angle_deg": describe(train["flip_angle_deg"]),
            "echo_train_length": describe(train["echo_train_length"]),
            "pixel_bandwidth": describe(train["pixel_bandwidth"]),
        },
        "demographic_tag_counts": demographic_counts,
    }, merged


def analyze_dicom(
    root: Path,
    output: Path,
    series_limit: int,
    geometry_sample_size: int,
    pixel_sample_size: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    all_dirs = {
        split: series_directories(root, split) for split in ("train", "test")
    }
    selected_train = deterministic_subset(all_dirs["train"], series_limit, "headers")
    selected = {"train": selected_train, "test": all_dirs["test"]}
    geometry_keys = {
        (split, study, series)
        for split, rows in selected.items()
        for study, series, _ in deterministic_subset(rows, geometry_sample_size, f"geometry-{split}")
    }
    pixel_keys = {
        (split, study, series)
        for split, rows in selected.items()
        for study, series, _ in deterministic_subset(rows, pixel_sample_size, f"pixels-{split}")
    }

    records = []
    geometry_rows = []
    pixel_rows = []
    for split, rows in selected.items():
        print(f"[eda] {split}: inspecting {len(rows):,} representative DICOM headers", flush=True)
        for index, (study, series, series_dir) in enumerate(rows, start=1):
            record, files = representative_header_record(split, study, series, series_dir)
            records.append(record)
            key = (split, study, series)
            if key in geometry_keys:
                geometry_rows.append(geometry_record(split, study, series, files))
            if key in pixel_keys:
                pixel_rows.append(decode_pixel_sample(record, files))
            if index % 1000 == 0:
                print(f"[eda] {split}: {index:,}/{len(rows):,} series", flush=True)

    inventory = pd.DataFrame(records)
    geometry = pd.DataFrame(geometry_rows)
    pixels = pd.DataFrame(pixel_rows)
    inventory.to_csv(output / "dicom_series_inventory.csv.gz", index=False, compression=GZIP_OPTIONS)
    geometry.to_csv(output / "dicom_geometry_audit.csv", index=False)
    pixels.to_csv(output / "dicom_pixel_decode_sample.csv", index=False)

    train_inventory = inventory[inventory["split"] == "train"].copy()
    counts_to_save = {
        "manufacturer": "manufacturer_counts.csv",
        "model": "scanner_model_counts.csv",
        "field_strength_t": "field_strength_counts.csv",
        "body_part": "body_part_counts.csv",
        "photometric": "photometric_counts.csv",
        "transfer_syntax_uid": "transfer_syntax_counts.csv",
    }
    for column, filename in counts_to_save.items():
        (
            train_inventory[column]
            .fillna("missing")
            .value_counts(dropna=False)
            .rename_axis(column)
            .reset_index(name="series")
            .to_csv(output / filename, index=False)
        )

    split_summary = {}
    for split, rows in all_dirs.items():
        inv = inventory[inventory["split"] == split]
        split_summary[split] = {
            "filesystem_series_directories": len(rows),
            "series_headers_inspected": int(len(inv)),
            "header_read_failures": int((~inv["header_read_ok"]).sum()),
            "slice_files_in_inspected_series": int(inv["slice_files"].sum()),
            "bytes_in_inspected_series": int(inv["series_bytes"].sum()),
            "series_with_no_dicoms": int((inv["slice_files"] == 0).sum()),
            "study_uid_path_mismatches": int((inv["header_read_ok"] & ~inv["study_uid_matches_path"]).sum()),
            "series_uid_path_mismatches": int((inv["header_read_ok"] & ~inv["series_uid_matches_path"]).sum()),
            "slice_count": describe(inv["slice_files"]),
        }

    readable = train_inventory[train_inventory["header_read_ok"]]
    study_laterality = []
    for study, group in readable.groupby("path_study_uid", observed=True):
        values = []
        for column in ("image_laterality", "laterality"):
            values.extend(
                value.upper()
                for value in group[column].dropna().astype(str)
                if value.upper() in {"L", "R", "LEFT", "RIGHT"}
            )
        normalized = ["L" if value in {"L", "LEFT"} else "R" for value in values]
        counts = Counter(normalized)
        if not counts:
            status = "unknown"
        elif counts["L"] == counts["R"]:
            status = "conflict_or_tie"
        else:
            status = counts.most_common(1)[0][0]
        study_laterality.append({"StudyInstanceUID": study, "laterality_status": status})
    study_laterality_df = pd.DataFrame(study_laterality)
    study_laterality_df.to_csv(output / "study_laterality_availability.csv", index=False)

    geometry_summary = {}
    if not geometry.empty:
        geometry_summary = {
            "series_checked": int(len(geometry)),
            "series_with_all_projectable_slices": int(
                ((geometry["projectable_slices"] == geometry["slice_files"]) & (geometry["slice_files"] > 0)).sum()
            ),
            "series_orientation_within_1_degree": int(geometry["orientation_within_1_degree"].sum()),
            "series_filename_order_physical_monotonic": int(
                geometry["filename_order_is_physical_monotonic"].sum()
            ),
            "series_with_duplicate_positions_within_0_1mm": int(
                (geometry["positions_within_0_1mm"] > 0).sum()
            ),
            "physical_slice_step_mm": describe(geometry["median_physical_slice_step_mm"]),
            "physical_span_mm": describe(geometry["physical_span_mm"]),
        }

    summary = {
        "by_split": split_summary,
        "representative_header_sampling": {
            "limit": series_limit,
            "train_is_complete": series_limit <= 0 or series_limit >= len(all_dirs["train"]),
            "selection_seed": SEED,
        },
        "rows": describe(readable["rows"]),
        "columns": describe(readable["columns"]),
        "pixel_spacing_row_mm": describe(readable["pixel_spacing_row_mm"]),
        "pixel_spacing_column_mm": describe(readable["pixel_spacing_column_mm"]),
        "slice_thickness_mm": describe(readable["slice_thickness_mm"]),
        "spacing_between_slices_mm": describe(readable["spacing_between_slices_mm"]),
        "series_missing_pixel_spacing": int(readable["pixel_spacing_row_mm"].isna().sum()),
        "series_missing_iop": int((~readable["has_iop"]).sum()),
        "series_missing_ipp": int((~readable["has_ipp"]).sum()),
        "monochrome1_series": int((readable["photometric"] == "MONOCHROME1").sum()),
        "modality_counts": readable["modality"].value_counts(dropna=False).to_dict(),
        "laterality_study_counts": study_laterality_df["laterality_status"].value_counts().to_dict(),
        "geometry_audit": geometry_summary,
        "pixel_decode_sample": {
            "attempted": int(len(pixels)),
            "successful": int(pixels["decode_ok"].sum()) if not pixels.empty else 0,
        },
    }
    return summary, inventory, geometry


def reconcile_filesystem_csv(
    root: Path,
    csv_frames: dict[str, pd.DataFrame],
    output: Path,
) -> dict[str, Any]:
    rows = []
    for split in ("train", "test"):
        frame = csv_frames[f"{split}_series"]
        csv_pairs = set(zip(frame["StudyInstanceUID"], frame["SeriesInstanceUID"]))
        fs_pairs = {
            (study, series) for study, series, _ in series_directories(root, split)
        }
        for study, series in sorted(csv_pairs - fs_pairs):
            rows.append({"split": split, "status": "declared_in_csv_missing_on_disk", "StudyInstanceUID": study, "SeriesInstanceUID": series})
        for study, series in sorted(fs_pairs - csv_pairs):
            rows.append({"split": split, "status": "on_disk_missing_from_csv", "StudyInstanceUID": study, "SeriesInstanceUID": series})
    mismatch = pd.DataFrame(
        rows,
        columns=["split", "status", "StudyInstanceUID", "SeriesInstanceUID"],
    )
    mismatch.to_csv(output / "filesystem_csv_mismatches.csv", index=False)
    return {
        "mismatch_rows": int(len(mismatch)),
        "by_status": mismatch["status"].value_counts().to_dict() if len(mismatch) else {},
    }


def make_plots(
    output: Path,
    frames: dict[str, pd.DataFrame],
    inventory: pd.DataFrame,
    geometry: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plots = output / "plots"
    plots.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    frames["series_per_study"]["series_count"].hist(ax=axes[0], bins=np.arange(2.5, 15.5, 1))
    axes[0].set(title="Series per training study", xlabel="series", ylabel="studies")
    plane_counts = frames["train_series"]["Anatomical_Plane"].value_counts()
    axes[1].bar(plane_counts.index, plane_counts.values)
    axes[1].set(title="Declared acquisition plane", ylabel="series")
    slot = frames["slot_coverage"].copy()
    slot["slot"] = slot["plane"] + " / " + slot["fluid_sensitive"].map({0: "non-fluid", 1: "fluid"})
    axes[2].barh(slot["slot"], slot["study_coverage"])
    axes[2].set(title="Study-level slot coverage", xlabel="fraction of studies", xlim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(plots / "01_study_and_series_structure.png", dpi=160)
    plt.close(fig)

    prevalence = frames["prevalence"].sort_values("prevalence")
    fig, axis = plt.subplots(figsize=(9, 5.5))
    error = np.vstack(
        [
            prevalence["prevalence"] - prevalence["wilson_95_low"],
            prevalence["wilson_95_high"] - prevalence["prevalence"],
        ]
    )
    axis.barh(prevalence["target"], prevalence["prevalence"], xerr=error, capsize=3)
    axis.set(title="Target prevalence among 58 fully labeled studies (95% Wilson CI)", xlabel="positive fraction", xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(plots / "02_target_prevalence.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(frames["correlation"], vmin=-1, vmax=1, cmap="vlag", square=True, ax=axis)
    axis.set_title("Target phi correlations (n=58; exploratory only)")
    fig.tight_layout()
    fig.savefig(plots / "03_target_correlations.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    reports = frames["report_stats"]
    cutoff = reports["words"].quantile(0.99)
    reports.loc[reports["words"] <= cutoff, "words"].hist(ax=axes[0], bins=50)
    axes[0].set(title="Report length (below p99)", xlabel="words", ylabel="reports")
    reports["lines"].clip(upper=reports["lines"].quantile(0.99)).hist(ax=axes[1], bins=40)
    axes[1].set(title="Report line count (clipped at p99)", xlabel="lines")
    fig.tight_layout()
    fig.savefig(plots / "04_report_lengths.png", dpi=160)
    plt.close(fig)

    train_inv = inventory[(inventory["split"] == "train") & inventory["header_read_ok"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    train_inv["pixel_spacing_row_mm"].dropna().clip(upper=train_inv["pixel_spacing_row_mm"].quantile(0.995)).hist(ax=axes[0, 0], bins=60)
    axes[0, 0].set(title="In-plane pixel spacing", xlabel="mm/pixel")
    train_inv["slice_thickness_mm"].dropna().clip(upper=train_inv["slice_thickness_mm"].quantile(0.995)).hist(ax=axes[0, 1], bins=50)
    axes[0, 1].set(title="Slice thickness", xlabel="mm")
    train_inv["slice_files"].clip(upper=train_inv["slice_files"].quantile(0.995)).hist(ax=axes[1, 0], bins=60)
    axes[1, 0].set(title="Slices per series", xlabel="DICOM files")
    matrix = train_inv.assign(matrix=train_inv["rows"].astype(str) + "×" + train_inv["columns"].astype(str))["matrix"].value_counts().head(12)
    axes[1, 1].barh(matrix.index[::-1], matrix.values[::-1])
    axes[1, 1].set(title="Most common image matrices", xlabel="series")
    fig.tight_layout()
    fig.savefig(plots / "05_dicom_geometry.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    manufacturers = train_inv["manufacturer"].value_counts().head(12)
    axes[0].barh(manufacturers.index[::-1], manufacturers.values[::-1])
    axes[0].set(title="Manufacturer strings (top 12)", xlabel="series")
    models = train_inv["model"].value_counts().head(15)
    axes[1].barh(models.index[::-1], models.values[::-1])
    axes[1].set(title="Scanner models (top 15)", xlabel="series")
    fig.tight_layout()
    fig.savefig(plots / "06_scanner_distribution.png", dpi=160)
    plt.close(fig)

    if not geometry.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        geometry["median_physical_slice_step_mm"].dropna().clip(upper=geometry["median_physical_slice_step_mm"].quantile(0.99)).hist(ax=axes[0], bins=45)
        axes[0].set(title="Physical inter-slice step (audited series)", xlabel="mm")
        labels = ["filename order is physical", "orientation ≤1°", "all slices projectable"]
        values = [
            geometry["filename_order_is_physical_monotonic"].mean(),
            geometry["orientation_within_1_degree"].mean(),
            (geometry["projectable_slices"] == geometry["slice_files"]).mean(),
        ]
        axes[1].barh(labels, values)
        axes[1].set(title="Geometry audit pass rates", xlabel="fraction", xlim=(0, 1.05))
        fig.tight_layout()
        fig.savefig(plots / "07_geometry_audit.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    require_layout(root)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[eda] data root: {root}", flush=True)
    print(f"[eda] output:    {output}", flush=True)
    csv_summary, frames = analyze_csvs(root, output)
    dicom_summary, inventory, geometry = analyze_dicom(
        root,
        output,
        args.dicom_series_limit,
        args.geometry_series_sample,
        args.pixel_samples,
    )
    derived_summary, merged_inventory = derive_series_features(inventory, frames, output)
    dicom_summary["derived_series_features"] = derived_summary
    reconciliation = reconcile_filesystem_csv(root, frames, output)
    summary = {
        "eda_version": 1,
        "selection_seed": SEED,
        "data_root": str(root),
        "input_csv_fingerprints": {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": sha256_file(root / name),
            }
            for name in (
                "train.csv",
                "train_series.csv",
                "test.csv",
                "test_series.csv",
                "sample_submission.csv",
            )
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pydicom": pydicom.__version__,
        },
        "csv": csv_summary,
        "dicom": dicom_summary,
        "filesystem_csv_reconciliation": reconciliation,
        "scope_notes": [
            "All CSV rows are analyzed.",
            "One representative DICOM header is read per selected series.",
            "Pixel values are decoded only for a deterministic validation sample.",
            "Report language fingerprints overlap and are not language classifications.",
            "Target prevalence and correlations use only the 58 fully labeled studies.",
        ],
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_value(summary), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    if not args.skip_plots:
        make_plots(output, frames, inventory, geometry)
    print("[eda] complete", flush=True)


if __name__ == "__main__":
    main()
