"""cache64: our own wide-span corpus -- Raptor's "maxspan" recipe, 64 slices x 336.

WHY. arm10's weakest findings besides Synovitis and Lateral OA are Lateral
Meniscus (.827) and MCL (.846). Both live in peripheral slices, and the public
44-slice corpus arm10 trained on keeps only the central 0.15-0.85 of each
series. The Raptor author widened the span for exactly these structures; the
arm routed the most weight in the 0.941 public blend ("maxspan-v5", w=0.60)
uses 64 slices over 0.02-0.98. No public CC0 corpus of that layout exists, so
we build it. CPU only -- costs no GPU quota.

RECIPE. Every function below is the one arm10_verify matched pixel-for-pixel
against the public corpus (IPP . normal ordering, modality LUT, MONOCHROME1,
per-series percentile[2,98], 140 mm centre crop, INTER_AREA to 336, first
unused series matching plane + Fluid_Sensitive). Only the layout parameters
change, to the maxspan values taken from the public Raptor inference code:
    SLOTS64 = Sag-FS 18 | Sag-nonFS 14 | Cor-FS 12 | Cor-nonFS 8 | Axial 12
    SPAN    = 0.02-0.98
So the hidden-test rebuild stays exactly reproducible: same code, same params.

LAYOUT. 4,407 studies x 64 x 336 x 336 uint8 = 31.8 GB, over Kaggle's ~20 GB
output cap, so two shards (train.csv order, rows[SHARD::2]). Each shard writes
    vols_shard{S}.npy   (N, 64, 336, 336) uint8, via open_memmap (never all in RAM)
    masks_shard{S}.npy  (N, 64) uint8
    ids_shard{S}.npy    (N,) str
Gold studies are included (validation needs them); training excludes them.
"""
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from numpy.lib.format import open_memmap
from pydicom.pixel_data_handlers.util import apply_modality_lut

SHARD = 1
N_SHARDS = 2
IMG = 336
CROP_MM = 140.0
SPAN = (0.02, 0.98)
SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
           ("Coronal", 0, 8), ("Axial", -1, 12)]
DEPTH = sum(k for _, _, k in SLOTS64)
WORKERS = 4
LIMIT = 0                     # >0 for a quick smoke run
WORK = Path("/kaggle/working")

cv2.setNumThreads(1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_comp():
    for d, dirs, fs in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if "train.csv" in fs and "train_series.csv" in fs:
            return Path(d)
    raise FileNotFoundError("competition data not attached")


# ---------------- verified rebuild functions (arm10_verify / infer_arm10) ----
def order_and_meta(sdir):
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


def read_px(f):
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def mm_crop_resize(a, ps):
    h, w = a.shape
    c = min(int(round(CROP_MM / max(ps, 1e-3))), min(h, w))
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


def build_study(sid, rows, series_root):
    vol = np.zeros((DEPTH, IMG, IMG), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in SLOTS64:
        r = pick_series(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        files, med = order_and_meta(f"{series_root}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * SPAN[0]), int(n * SPAN[1]) - 1
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
            if idx >= DEPTH:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            vol[idx] = (mm_crop_resize(aw, ps if ps > 0 else med) * 255).astype(np.uint8)
            idx += 1
    mask = (vol.reshape(DEPTH, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask
# -----------------------------------------------------------------------------


_CTX = {}


def _init(by, root):
    _CTX["by"], _CTX["root"] = by, root


def _job(args):
    i, sid = args
    try:
        vol, mask = build_study(sid, _CTX["by"].get(sid, []), _CTX["root"])
        return i, vol, mask, None
    except Exception as e:                       # one bad study must not sink the shard
        return i, np.zeros((DEPTH, IMG, IMG), np.uint8), np.zeros(DEPTH, np.uint8), repr(e)[:200]


def main():
    t0 = time.time()
    comp = find_comp()
    train = pd.read_csv(comp / "train.csv")
    uids = train.StudyInstanceUID.astype(str).tolist()[SHARD::N_SHARDS]
    if LIMIT:
        uids = uids[:LIMIT]
    ser = pd.read_csv(comp / "train_series.csv")
    ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str)
    by = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}
    n = len(uids)
    gb = n * DEPTH * IMG * IMG / 1e9
    log(f"shard {SHARD}/{N_SHARDS}: {n} studies -> ({n}, {DEPTH}, {IMG}, {IMG}) = {gb:.1f} GB | "
        f"span {SPAN} | slots {SLOTS64}")

    vols = open_memmap(WORK / f"vols_shard{SHARD}.npy", mode="w+", dtype=np.uint8,
                       shape=(n, DEPTH, IMG, IMG))
    masks = np.zeros((n, DEPTH), np.uint8)
    failed = []
    last = time.time()
    with ProcessPoolExecutor(WORKERS, initializer=_init,
                             initargs=(by, str(comp / "train_series"))) as pool:
        for done, (i, vol, mask, err) in enumerate(pool.map(_job, enumerate(uids), chunksize=4), 1):
            vols[i] = vol
            masks[i] = mask
            if err:
                failed.append((uids[i], err))
            if time.time() - last > 60 or done == n:
                el = time.time() - t0
                log(f"  {done}/{n} studies | {el / 60:.1f} min | eta {el / done * (n - done) / 60:.1f} min"
                    f" | failed {len(failed)}")
                last = time.time()
    vols.flush()
    del vols
    np.save(WORK / f"masks_shard{SHARD}.npy", masks)
    np.save(WORK / f"ids_shard{SHARD}.npy", np.array(uids))

    # determinism check: rebuild a few studies and require byte equality with what was written
    check = np.load(WORK / f"vols_shard{SHARD}.npy", mmap_mode="r")
    rng = np.random.default_rng(SHARD)
    for i in rng.choice(n, min(3, n), replace=False):
        again, _ = build_study(uids[i], by.get(uids[i], []), str(comp / "train_series"))
        if not np.array_equal(again, np.asarray(check[i])):
            raise RuntimeError(f"non-deterministic rebuild for study {i}")
    valid = masks.sum(1)
    meta = {"shard": SHARD, "n_shards": N_SHARDS, "n": n, "depth": DEPTH, "img": IMG,
            "span": SPAN, "crop_mm": CROP_MM, "slots": SLOTS64, "failed": failed,
            "valid_slices_min": int(valid.min()), "valid_slices_median": float(np.median(valid)),
            "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(WORK / f"meta_shard{SHARD}.json", "w"), indent=1)
    log(f"DONE shard {SHARD}: {n} studies, failed {len(failed)}, valid slices/study min "
        f"{meta['valid_slices_min']} median {meta['valid_slices_median']} | determinism PASS | "
        f"{meta['minutes']:.1f} min")


if __name__ == "__main__":
    main()
