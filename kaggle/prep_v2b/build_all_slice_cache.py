"""Cache v2: EVERY slice of every series, ragged layout, sharded.

Cache v1 (kaggle/prep/) stores 6 slices/slot = 30 per case. The latent case
model needs all of them (~140/case average, 80-400 range), so this caches the
lot.

Three design constraints drove the layout:

  * Ragged, not rectangular. Slice counts vary 80-400 per case; a fixed array
    padded to the max would be ~45 GB and mostly padding. Instead: one
    concatenated array of all slices + an offsets index giving each case's
    [start, end). Zero padding waste.
  * Sharded in halves. All slices at 160px is ~15.6 GB, uncomfortably close to
    Kaggle's ~20 GB kernel output limit, and ~4.6x the decode work of cache v1
    (which risks the CPU session limit). Two shards of ~8 GB each fixes both.
    Set SHARD below and push once per shard.
  * Stays on Kaggle. 15.6 GB cannot move over this connection at ~1 MB/s, so
    the training kernel attaches these kernel outputs directly via
    `kernel_sources` rather than round-tripping through a dataset upload.

Output per shard (in /kaggle/working):
  slices_shard{S}.npy    (total_slices, 160, 160) uint8, all cases concatenated
  index_shard{S}.json    {uids, offsets, series_of_slice, slot_of_slice,
                          depth_frac_of_slice, split}

The per-slice metadata is what the model's slot/depth embeddings consume --
caching it here means the training loop never touches a DICOM header.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

SHARD = 1          # 0 or 1 -- push once per shard
N_SHARDS = 2
IMG_SIZE = 160
MAX_SLICES_PER_CASE = 320   # only the extreme tail gets subsampled

SLOT_ORDER = ["Sagittal_FS", "Sagittal_nonFS", "Coronal_FS", "Coronal_nonFS",
              "Axial_FS", "Axial_nonFS", "Other"]


def find_root(input_root: Path, marker: str, skip_dirs=("train_series", "test_series")) -> Path:
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{marker!r} not found under {input_root}")


INPUT_ROOT = Path("/kaggle/input")
COMP_ROOT = find_root(INPUT_ROOT, "train_series.csv")
LABELS_ROOT = find_root(INPUT_ROOT, "gold_test.csv")
SERIES_ROOT = COMP_ROOT / "train_series"
WORK = Path("/kaggle/working")
print(f"COMP_ROOT={COMP_ROOT}")
print(f"shard {SHARD} of {N_SHARDS}, {IMG_SIZE}px, all slices")


def slot_name(plane: str, fluid_sensitive) -> str:
    """Slot identity for the model's slot embedding. Unlike arm2/arm3's rigid
    5-slot grid this is a *label*, not a capacity limit -- a case with three
    sagittal series keeps all three."""
    fs = "FS" if int(fluid_sensitive) == 1 else "nonFS"
    if plane in ("Sagittal", "Coronal"):
        return f"{plane}_{fs}"
    if plane == "Axial":
        return f"Axial_{fs}"
    return "Other"


def sorted_series_files(series_dir: Path) -> list:
    files = list(series_dir.glob("*.dcm"))
    tagged = []
    for f in files:
        try:
            n = pydicom.dcmread(str(f), stop_before_pixels=True).InstanceNumber
        except Exception:
            n = 0
        tagged.append((int(n), f))
    tagged.sort(key=lambda t: t[0])
    return [f for _, f in tagged]


def load_slice(path: Path, size: int) -> np.ndarray:
    dcm = pydicom.dcmread(str(path))
    arr = dcm.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi).astype(np.float32, copy=False)
    arr -= arr.min()
    denom = arr.max()
    if denom > 0:
        arr /= denom
    if arr.shape != (size, size):
        import cv2
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr.astype(np.float32, copy=False)


def main():
    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv")[["StudyInstanceUID"]].copy()
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")[["StudyInstanceUID"]].copy()
    gold["split"] = "gold"
    pool["split"] = "pool"
    studies = pd.concat([gold, pool], ignore_index=True)

    shard_studies = studies.iloc[SHARD::N_SHARDS].reset_index(drop=True)
    print(f"{len(shard_studies)} studies in this shard (of {len(studies)} total)")

    series_df = pd.read_csv(COMP_ROOT / "train_series.csv")
    series_by_study = {uid: g for uid, g in series_df.groupby("StudyInstanceUID")}

    all_slices = []          # list of (IMG_SIZE, IMG_SIZE) uint8
    offsets = [0]
    uids, splits = [], []
    slot_of_slice, series_of_slice, depth_frac = [], [], []

    t0 = time.time()
    last_print = t0
    for i, row in shard_studies.iterrows():
        uid = row["StudyInstanceUID"]
        sdf = series_by_study.get(uid)
        case_slices = []
        case_slots, case_series, case_depth = [], [], []

        if sdf is not None:
            for series_i, (_, srow) in enumerate(sdf.iterrows()):
                files = sorted_series_files(SERIES_ROOT / uid / srow["SeriesInstanceUID"])
                n = len(files)
                if n == 0:
                    continue
                slot = slot_name(srow["Anatomical_Plane"], srow["Fluid_Sensitive"])
                for depth_i, f in enumerate(files):
                    try:
                        case_slices.append(
                            np.clip(load_slice(f, IMG_SIZE) * 255.0, 0, 255).astype(np.uint8))
                        case_slots.append(SLOT_ORDER.index(slot))
                        case_series.append(series_i)
                        # normalized depth so the embedding is comparable across
                        # series of different lengths
                        case_depth.append(depth_i / max(n - 1, 1))
                    except Exception as e:
                        print(f"  slice error {uid} {slot} {depth_i}: {e}")

        # Only the extreme tail is subsampled; typical cases keep everything.
        if len(case_slices) > MAX_SLICES_PER_CASE:
            keep = np.linspace(0, len(case_slices) - 1, MAX_SLICES_PER_CASE).round().astype(int)
            case_slices = [case_slices[k] for k in keep]
            case_slots = [case_slots[k] for k in keep]
            case_series = [case_series[k] for k in keep]
            case_depth = [case_depth[k] for k in keep]

        all_slices.extend(case_slices)
        slot_of_slice.extend(case_slots)
        series_of_slice.extend(case_series)
        depth_frac.extend(case_depth)
        offsets.append(len(all_slices))
        uids.append(uid)
        splits.append(row["split"])

        if time.time() - last_print > 30:
            elapsed = time.time() - t0
            done = i + 1
            eta_min = (len(shard_studies) - done) * elapsed / done / 60
            gb = len(all_slices) * IMG_SIZE * IMG_SIZE / 1e9
            print(f"  {done}/{len(shard_studies)} cases, {len(all_slices)} slices "
                  f"({gb:.2f} GB), {elapsed:.0f}s elapsed, eta {eta_min:.1f} min",
                  flush=True)
            last_print = time.time()

    print(f"\nstacking {len(all_slices)} slices...")
    pixels = np.stack(all_slices) if all_slices else np.zeros((0, IMG_SIZE, IMG_SIZE), np.uint8)
    del all_slices

    np.save(WORK / f"slices_shard{SHARD}.npy", pixels)
    with open(WORK / f"index_shard{SHARD}.json", "w") as f:
        json.dump({
            "uids": uids, "split": splits, "offsets": offsets,
            "slot_of_slice": slot_of_slice,
            "series_of_slice": series_of_slice,
            "depth_frac_of_slice": [round(d, 4) for d in depth_frac],
            "slot_order": SLOT_ORDER, "img_size": IMG_SIZE,
            "shard": SHARD, "n_shards": N_SHARDS,
        }, f)

    per_case = np.diff(offsets)
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min")
    print(f"pixels: {pixels.shape}, {pixels.nbytes / 1e9:.2f} GB")
    print(f"slices/case: mean {per_case.mean():.1f}, min {per_case.min()}, "
          f"max {per_case.max()}")


if __name__ == "__main__":
    main()
