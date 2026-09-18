"""One-time pixel cache: decode every study's 5-slot bag ONCE, save as arrays.

Why this exists: arm2/arm3 re-read and re-decode the same DICOM files from
scratch every single epoch -- that decode+resize cost (not the GPU) is what's
dominating epoch time (~50 min/epoch on the full 4,349-study pool) and eating
the 6h Kaggle session cap after only 4-6 epochs. This pays that decode cost
ONCE and saves the result, so every future training run (arm4+, hyperparameter
sweeps, the SSL-vs-scratch rematch we still owe) loads a plain array from RAM
instead of touching DICOM at all. Same idea as the community's own
"session 1" pixel-cache notebooks we found early in this project
(dreaddevelopment/knee-raptor-corpus) -- this is our version of that.

Covers ALL 4,407 studies (gold_test's 58 + train_pool's 4,349) in one cache,
tagged by split, so it's reusable for training AND internal eval. Caches 6
slices/slot (not just the 4 arm2/arm3 currently use) for a little headroom --
future runs can use a subset without needing a full rebuild.

CPU-only by design (no GPU needed for decode/resize), so this can run
alongside a GPU training kernel without competing for GPU quota.

Output (saved to /kaggle/working, then upload as a dataset):
  pixels.npy       (N, 5*6, 160, 160) uint8  -- x255 of the [0,1] normalized float
  mask.npy         (N, 5*6) uint8
  study_index.json {"uids": [...], "split": [...]}  -- row order + gold/pool tag
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

SLOTS = ["Sagittal_FS", "Sagittal_nonFS", "Coronal_FS", "Coronal_nonFS", "Axial"]
SLICES_PER_SLOT = 6  # headroom over arm2/arm3's current 4, for future runs
IMG_SIZE = 160


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
print(f"LABELS_ROOT={LABELS_ROOT}")


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


def pick_slot_series(series_df: pd.DataFrame) -> dict:
    chosen = {}
    for plane, fs, slot in [
        ("Sagittal", 1, "Sagittal_FS"), ("Sagittal", 0, "Sagittal_nonFS"),
        ("Coronal", 1, "Coronal_FS"), ("Coronal", 0, "Coronal_nonFS"),
    ]:
        cand = series_df[(series_df.Anatomical_Plane == plane) & (series_df.Fluid_Sensitive == fs)]
        if not cand.empty:
            chosen[slot] = cand.iloc[0]["SeriesInstanceUID"]
    axial = series_df[series_df.Anatomical_Plane == "Axial"].sort_values("Fluid_Sensitive", ascending=False)
    if not axial.empty:
        chosen["Axial"] = axial.iloc[0]["SeriesInstanceUID"]
    return chosen


def build_bag(uid: str, series_df: pd.DataFrame) -> tuple:
    n_slots = len(SLOTS)
    bag = np.zeros((n_slots * SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE), np.float32)
    mask = np.zeros(n_slots * SLICES_PER_SLOT, np.float32)

    sdf = series_df[series_df.StudyInstanceUID == uid]
    if sdf.empty:
        return bag, mask
    chosen = pick_slot_series(sdf)
    for slot_idx, slot in enumerate(SLOTS):
        series_uid = chosen.get(slot)
        if series_uid is None:
            continue
        files = sorted_series_files(SERIES_ROOT / uid / series_uid)
        n = len(files)
        if n == 0:
            continue
        fracs = np.linspace(0.06, 0.94, SLICES_PER_SLOT)
        idxs = np.clip((fracs * (n - 1)).round().astype(int), 0, n - 1)
        for s, fi in enumerate(idxs):
            try:
                bag[slot_idx * SLICES_PER_SLOT + s] = load_slice(files[fi], IMG_SIZE)
                mask[slot_idx * SLICES_PER_SLOT + s] = 1.0
            except Exception as e:
                print(f"  slice error {uid} {slot} idx{s}: {e}")
    return bag, mask


def main():
    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv")[["StudyInstanceUID"]].copy()
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")[["StudyInstanceUID"]].copy()
    gold["split"] = "gold"
    pool["split"] = "pool"
    studies = pd.concat([gold, pool], ignore_index=True)
    n = len(studies)
    print(f"caching {n} studies ({len(gold)} gold + {len(pool)} pool), "
          f"{len(SLOTS)} slots x {SLICES_PER_SLOT} slices, {IMG_SIZE}px")

    series_df = pd.read_csv(COMP_ROOT / "train_series.csv")

    n_per_study = len(SLOTS) * SLICES_PER_SLOT
    pixels = np.zeros((n, n_per_study, IMG_SIZE, IMG_SIZE), np.uint8)
    masks = np.zeros((n, n_per_study), np.uint8)

    t0 = time.time()
    last_print = t0
    for i, row in studies.iterrows():
        bag, mask = build_bag(row["StudyInstanceUID"], series_df)
        pixels[i] = np.clip(bag * 255.0, 0, 255).astype(np.uint8)
        masks[i] = mask.astype(np.uint8)
        if time.time() - last_print > 30:
            elapsed = time.time() - t0
            eta_min = (n - i - 1) * elapsed / (i + 1) / 60
            print(f"  {i + 1}/{n} studies cached, {elapsed:.0f}s elapsed, "
                  f"eta {eta_min:.1f} min", flush=True)
            last_print = time.time()

    np.save(WORK / "pixels.npy", pixels)
    np.save(WORK / "mask.npy", masks)
    with open(WORK / "study_index.json", "w") as f:
        json.dump({"uids": studies["StudyInstanceUID"].tolist(),
                   "split": studies["split"].tolist(),
                   "slots": SLOTS, "slices_per_slot": SLICES_PER_SLOT,
                   "img_size": IMG_SIZE}, f, indent=2)

    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min")
    print(f"pixels.npy: {pixels.shape}, {pixels.nbytes / 1e9:.2f} GB")
    print(f"mask.npy: {masks.shape}")


if __name__ == "__main__":
    main()
