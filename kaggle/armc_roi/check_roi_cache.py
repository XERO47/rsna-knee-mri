"""Check whether the Arm C ROI cache files are attached in a Kaggle notebook.

Paste this file into a Kaggle cell, or run:

    exec(open('/kaggle/working/check_roi_cache.py').read())

Set ROI_CHECK_STRICT=1 to raise an error if any required file is missing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np


ROI_TARGETS = (
    "MCL",
    "Lateral Meniscus",
    "Lateral OA",
    "PF OA",
    "Synovitis",
)
INPUT_ROOT = Path(os.environ.get("ARMC_ROI_INPUT", "/kaggle/input"))
STRICT = os.environ.get("ROI_CHECK_STRICT", "0") == "1"
SKIP_DIRECTORIES = {"train_series", "test_series", ".git", "__pycache__"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def expected_names(target: str) -> list[str]:
    target_slug = slug(target)
    return [
        f"roi_{target_slug}_shard0.npy",
        f"roi_{target_slug}_shard1.npy",
        f"roi_{target_slug}_masks_shard0.npy",
        f"roi_{target_slug}_masks_shard1.npy",
    ]


def supporting_names() -> list[str]:
    return [
        "roi_ids_shard0.npy",
        "roi_ids_shard1.npy",
        "roi_provenance_shard0.csv.gz",
        "roi_provenance_shard1.csv.gz",
        "roi_meta_shard0.json",
        "roi_meta_shard1.json",
        "roi_priors.json",
    ]


def merged_names(target: str) -> list[str]:
    target_slug = slug(target)
    return [f"roi_{target_slug}.npy", f"roi_{target_slug}_masks.npy"]


def index_files(root: Path) -> dict[str, list[Path]]:
    """Index candidate filenames while deliberately skipping raw DICOM trees."""
    found: dict[str, list[Path]] = {}
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories if name not in SKIP_DIRECTORIES
        ]
        for filename in filenames:
            lower = filename.lower()
            if "roi" in lower and (
                lower.endswith(".npy")
                or lower.endswith(".json")
                or lower.endswith(".csv.gz")
                or lower.endswith(".csv")
            ):
                path = Path(directory) / filename
                found.setdefault(filename, []).append(path)
    return found


def find_raw_dicom_roots(root: Path) -> list[Path]:
    """Find raw series directories without traversing their DICOM contents."""
    roots: list[Path] = []
    for directory, subdirectories, _ in os.walk(root):
        raw_names = {
            name
            for name in subdirectories
            if name in {"train_series", "test_series"}
        }
        roots.extend(Path(directory) / name for name in sorted(raw_names))
        subdirectories[:] = [
            name
            for name in subdirectories
            if name not in SKIP_DIRECTORIES
        ]
    return roots


def describe(path: Path) -> str:
    try:
        array = np.load(path, mmap_mode="r")
        return f"FOUND  {path} | shape={tuple(array.shape)} dtype={array.dtype}"
    except Exception as error:
        return f"INVALID {path} | {type(error).__name__}: {error}"


def main() -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Kaggle input root does not exist: {INPUT_ROOT}")

    print(f"Searching {INPUT_ROOT} (skipping raw DICOM directories)...")
    discovered = index_files(INPUT_ROOT)
    raw_dicom_roots = find_raw_dicom_roots(INPUT_ROOT)
    missing: list[str] = []
    merged_mode = any(
        name in discovered
        for target in ROI_TARGETS
        for name in merged_names(target)
    ) or "roi_ids.npy" in discovered

    print("\nRaw DICOM source directories:")
    if raw_dicom_roots:
        for path in raw_dicom_roots:
            print(f"FOUND  {path}")
    else:
        print("MISSING train_series/test_series — ROI crops cannot be built from raw DICOM here.")

    if merged_mode:
        print("\nMerged ROI cache mode")
        for target in ROI_TARGETS:
            print(f"\n{target}")
            for name in merged_names(target):
                paths = discovered.get(name, [])
                if not paths:
                    print(f"MISSING {name}")
                    missing.append(name)
                    continue
                for path in paths:
                    print(describe(path))
        for name in ("roi_ids.npy", "roi_provenance.csv.gz", "roi_meta.json", "roi_priors.json"):
            paths = discovered.get(name, [])
            if not paths and name == "roi_provenance.csv.gz":
                paths = discovered.get("roi_provenance.csv", [])
            if not paths:
                print(f"MISSING {name}")
                missing.append(name)
                continue
            for path in paths:
                print(describe(path) if name.endswith(".npy") else f"FOUND  {path}")
    else:
        for target in ROI_TARGETS:
            print(f"\n{target}")
            for name in expected_names(target):
                paths = discovered.get(name, [])
                if not paths:
                    print(f"MISSING {name}")
                    missing.append(name)
                    continue
                for path in paths:
                    print(describe(path))

        print("\nAlignment and provenance files")
        for name in supporting_names():
            paths = discovered.get(name, [])
            if not paths:
                print(f"MISSING {name}")
                missing.append(name)
                continue
            for path in paths:
                if name.endswith(".npy"):
                    print(describe(path))
                else:
                    print(f"FOUND  {path}")

    for shard in (() if merged_mode else (0, 1)):
        roi_paths = discovered.get(f"roi_ids_shard{shard}.npy", [])
        global_paths = []
        for directory, subdirectories, filenames in os.walk(INPUT_ROOT):
            subdirectories[:] = [
                name
                for name in subdirectories
                if name not in SKIP_DIRECTORIES
            ]
            if f"ids_shard{shard}.npy" in filenames:
                global_paths.append(Path(directory) / f"ids_shard{shard}.npy")
        if roi_paths and global_paths:
            roi_ids = np.load(sorted(roi_paths)[0], allow_pickle=True).astype(str)
            global_ids = np.load(sorted(global_paths)[0], allow_pickle=True).astype(str)
            if np.array_equal(roi_ids, global_ids):
                print(f"PASS   shard {shard} ROI IDs exactly match global-cache IDs")
            else:
                message = (
                    f"MISALIGNED roi_ids_shard{shard}.npy "
                    f"versus ids_shard{shard}.npy"
                )
                print(message)
                missing.append(message)

    if merged_mode:
        roi_paths = discovered.get("roi_ids.npy", [])
        global_paths: dict[int, list[Path]] = {0: [], 1: []}
        for directory, subdirectories, filenames in os.walk(INPUT_ROOT):
            subdirectories[:] = [name for name in subdirectories if name not in SKIP_DIRECTORIES]
            for shard in (0, 1):
                name = f"ids_shard{shard}.npy"
                if name in filenames:
                    global_paths[shard].append(Path(directory) / name)
        if roi_paths and global_paths[0] and global_paths[1]:
            roi_ids = np.load(sorted(roi_paths)[0], allow_pickle=True).astype(str)
            expected = np.concatenate([
                np.load(sorted(global_paths[0])[0], allow_pickle=True).astype(str),
                np.load(sorted(global_paths[1])[0], allow_pickle=True).astype(str),
            ])
            if np.array_equal(roi_ids, expected):
                print("PASS   merged ROI IDs exactly match global shard 0 followed by shard 1 IDs")
            else:
                message = "MISALIGNED roi_ids.npy versus concatenated global-cache IDs"
                print(message)
                missing.append(message)

    print("\nOther ROI-like .npy files discovered:")
    expected = (
        {name for target in ROI_TARGETS for name in merged_names(target)}.union(
            {
                "roi_ids.npy",
                "roi_provenance.csv.gz",
                "roi_provenance.csv",
                "roi_meta.json",
                "roi_priors.json",
            }
        )
        if merged_mode
        else {name for target in ROI_TARGETS for name in expected_names(target)}.union(supporting_names())
    )
    other = [
        path
        for name, paths in discovered.items()
        if name not in expected
        for path in paths
    ]
    if other:
        for path in sorted(other):
            print(describe(path) if path.name.endswith(".npy") else f"FOUND  {path}")
    else:
        print("None")

    if missing:
        print(f"\nRESULT: INCOMPLETE — {len(missing)} ROI requirements failed.")
        if STRICT:
            raise FileNotFoundError("Missing ROI files: " + ", ".join(missing))
    else:
        print(
            "\nRESULT: PASS — ROI tensors, IDs, priors, metadata, "
            "and provenance were found and aligned."
        )


if __name__ == "__main__":
    main()
