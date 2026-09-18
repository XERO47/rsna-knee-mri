"""arm10 step 1: reverse-engineer how the public 44-slice Raptor corpus was built.

WHY THIS RUNS BEFORE ANY TRAINING. dreaddevelopment/knee-raptor-corpus (+ -ext)
is CC0 and holds all 4,407 train studies as (44, 336, 336) uint8 stacks -- it
would save us a ~30 GB cache build. But the hidden test set has no corpus: the
submission kernel must rebuild every test study from DICOM, and a model trained
on these stacks only works if that rebuild reproduces them. The public inference
notebooks use a different 64-slice corpus (span 0.06-0.94), and no public code
states the recipe for THIS one. Guessing wrong would not crash -- it would feed
the model subtly different images at test time and quietly cost score.

So: rebuild a sample of corpus rows from the raw DICOM under every plausible
recipe and measure which one reproduces the stored pixels. CPU only.

Candidate knobs, taken from the public Raptor code paths:
  slot layout  SLOTS44 = Sag-FS 12, Sag-nonFS 10, Cor-FS 8, Cor-nonFS 6, Axial 8
  span         0.15-0.85 (the "old" crop its comments mention), 0.06-0.94,
               0.02-0.98, 0.10-0.90
  crop         140 mm (Raptor), 130 mm (DINO branches)
Fixed (identical in every public Raptor reader): order by IPP . slice normal,
modality LUT, MONOCHROME1 inversion, per-series percentile[2,98] window over the
picked slices, centre square mm-crop, cv2 INTER_AREA resize.
"""
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

N_SAMPLE = 10
SEED = 7
IMG = 336
SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
           ("Coronal", 0, 6), ("Axial", -1, 8)]
SPANS = [(0.15, 0.85), (0.06, 0.94), (0.02, 0.98), (0.10, 0.90)]
CROPS = [140.0, 130.0]


def find_file(name, root="/kaggle/input"):
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if name in fs:
            return os.path.join(d, name)
    return None


def find_comp():
    for d, dirs, fs in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if "train_series.csv" in fs and os.path.isdir(os.path.join(d, "train_series")):
            return d
    raise FileNotFoundError("competition data not attached")


def order_and_meta(sdir):
    """Verbatim logic of the public Raptor reader."""
    recs, ps_list = [], []
    for f in glob.glob(sdir + "/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            ps_list.append(ps)
            recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda x: x[0])
    med = float(np.median(ps_list)) if ps_list else 0.5
    return [(f, ps) for _, f, ps in recs], med


_PX = {}


def read_px(f):
    if f not in _PX:
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
            a = a.max() - a
        _PX[f] = a
    return _PX[f]


def mm_crop_resize(a, ps, crop_mm):
    h, w = a.shape
    c = min(int(round(crop_mm / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - c) // 2, (w - c) // 2
    return cv2.resize(a[y0:y0 + c, x0:x0 + c], (IMG, IMG), interpolation=cv2.INTER_AREA)


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane
             and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build(sid, rows, series_root, span, crop_mm, order_cache):
    maxs = sum(k for _, _, k in SLOTS44)
    vol = np.zeros((maxs, IMG, IMG), np.uint8)
    idx, used, slot_of = 0, set(), np.full(maxs, -1)
    for si, (plane, fluid, k) in enumerate(SLOTS44):
        r = pick_series(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        sdir = f"{series_root}/{sid}/{r['SeriesInstanceUID']}"
        if sdir not in order_cache:
            order_cache[sdir] = order_and_meta(sdir)
        files, med = order_cache[sdir]
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * span[0]), int(n * span[1]) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs, pss = [], []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= maxs:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            vol[idx] = (mm_crop_resize(aw, ps if ps > 0 else med, crop_mm) * 255).astype(np.uint8)
            slot_of[idx] = si
            idx += 1
    return vol, slot_of


def main():
    t0 = time.time()
    comp = find_comp()
    paths = {n: find_file(n) for n in ("all_ids.npy", "all_masks.npy", "all_vols.npy",
                                       "window_index.npy", "extra_ids.npy",
                                       "extra_masks.npy", "extra_vols.npy", "extra_meta.json")}
    for k, v in paths.items():
        print(f"  {k:18s} {v}")

    ids = np.load(paths["all_ids.npy"], allow_pickle=True)
    masks = np.load(paths["all_masks.npy"])
    vols = np.load(paths["all_vols.npy"], mmap_mode="r")
    print(f"\npart1 ids {ids.shape} {ids.dtype} | masks {masks.shape} {masks.dtype} "
          f"| vols {vols.shape} {vols.dtype}")
    wi = np.load(paths["window_index.npy"], allow_pickle=True)
    print(f"window_index {wi.shape} {wi.dtype} first rows {wi[:3].tolist()}")
    if paths["extra_ids.npy"]:
        eids = np.load(paths["extra_ids.npy"], allow_pickle=True)
        emasks = np.load(paths["extra_masks.npy"])
        evols = np.load(paths["extra_vols.npy"], mmap_mode="r")
        print(f"part2 ids {eids.shape} | masks {emasks.shape} | vols {evols.shape}")
        print("extra_meta:", open(paths["extra_meta.json"]).read())
        all_ids = np.concatenate([ids.astype(str), eids.astype(str)])
        print(f"total studies {len(all_ids)}, unique {len(set(all_ids))}")

    train = pd.read_csv(f"{comp}/train.csv")
    print(f"ids present in train.csv: {np.isin(ids.astype(str), train.StudyInstanceUID.astype(str)).mean():.3f}")
    vc = masks.sum(1)
    print(f"valid slices/study: min {vc.min()} median {np.median(vc)} max {vc.max()}")

    ser = pd.read_csv(f"{comp}/train_series.csv")
    ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str)
    by = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}
    series_root = f"{comp}/train_series"

    rng = np.random.default_rng(SEED)
    rows_sample = rng.choice(len(ids), N_SAMPLE, replace=False)
    results = {f"span{s[0]:.2f}-{s[1]:.2f}_crop{int(c)}": [] for s in SPANS for c in CROPS}
    per_slot = {k: np.zeros(len(SLOTS44)) for k in results}
    order_cache = {}
    for n_done, row in enumerate(rows_sample):
        sid = str(ids[row])
        ref = np.asarray(vols[row]).astype(np.float32)
        for span in SPANS:
            for crop in CROPS:
                key = f"span{span[0]:.2f}-{span[1]:.2f}_crop{int(crop)}"
                vol, slot_of = build(sid, by.get(sid, []), series_root, span, crop, order_cache)
                mae = float(np.abs(vol.astype(np.float32) - ref).mean())
                exact = float((vol == np.asarray(vols[row])).mean())
                results[key].append((mae, exact))
                for si in range(len(SLOTS44)):
                    sel = slot_of == si
                    if sel.any():
                        per_slot[key][si] += np.abs(vol[sel].astype(np.float32) - ref[sel]).mean()
        _PX.clear()
        print(f"  study {n_done + 1}/{N_SAMPLE} done ({time.time() - t0:.0f}s)", flush=True)

    print("\n=== RECIPE MATCH (lower MAE is better; exact = fraction of identical pixels) ===")
    ranked = sorted(results.items(), key=lambda kv: np.mean([m for m, _ in kv[1]]))
    for key, vals in ranked:
        maes = [m for m, _ in vals]
        ex = [e for _, e in vals]
        print(f"  {key:26s} MAE mean {np.mean(maes):7.3f}  max {np.max(maes):7.3f}  "
              f"exact {np.mean(ex):.4f}")
    best = ranked[0][0]
    print(f"\nbest: {best}; per-slot MAE (Sag-FS, Sag-nonFS, Cor-FS, Cor-nonFS, Axial): "
          f"{np.round(per_slot[best] / N_SAMPLE, 3).tolist()}")
    json.dump({k: v for k, v in results.items()}, open("/kaggle/working/recipe_match.json", "w"))
    print(f"DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
