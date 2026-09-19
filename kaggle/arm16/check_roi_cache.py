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
INPUT_ROOT = Path(os.environ.get("ARM16_INPUT", "/kaggle/input"))
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


def index_files(root: Path) -> dict[str, list[Path]]:
    """Index candidate filenames while deliberately skipping raw DICOM trees."""
    found: dict[str, list[Path]] = {}
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories if name not in SKIP_DIRECTORIES
        ]
        for filename in filenames:
            lower = filename.lower()
            if "roi" in lower and lower.endswith(".npy"):
                path = Path(directory) / filename
                found.setdefault(filename, []).append(path)
    return found


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
    missing: list[str] = []

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

    print("\nOther ROI-like .npy files discovered:")
    expected = {name for target in ROI_TARGETS for name in expected_names(target)}
    other = [
        path
        for name, paths in discovered.items()
        if name not in expected
        for path in paths
    ]
    if other:
        for path in sorted(other):
            print(describe(path))
    else:
        print("None")

    if missing:
        print(f"\nRESULT: INCOMPLETE — {len(missing)} required ROI files are missing.")
        if STRICT:
            raise FileNotFoundError("Missing ROI files: " + ", ".join(missing))
    else:
        print("\nRESULT: PASS — all 20 expected ROI cache files were found.")


if __name__ == "__main__":
    main()
