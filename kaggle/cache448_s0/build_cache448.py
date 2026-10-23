"""cache448: the verified cache64 recipe rendered at 448 px instead of 336.

WHY. The teammate's EDA (references/dataset/dataset_structure.md) measured in-plane
spacing across all 24,371 series: median 0.3125 mm/px, p05 0.156, p95 0.5625. Our 140 mm
centre crop rendered at 336 px is 0.4167 mm/px, so for a MEDIAN study the crop is 448
native pixels and we resize it down to 336 -- discarding 1 - (336/448)^2 = 44% of the
information. At p05 spacing it is 86%.

    spacing p05  0.156 -> 140 mm crop = 897 native px   downsampled at both 336 and 448
    spacing p50  0.3125 ->               448 native px   DOWNsampled at 336, exact at 448
    spacing p95  0.5625 ->               249 native px   upsampled at both

448 is chosen because 140/448 = 0.3125 exactly matches the median native spacing, and 448
is divisible by 32 so both ConvNeXt and CoAtNet accept it. This is NOT the "336 -> 512
interpolation" the deep-research report warned about: that would invent detail, whereas
here the detail exists in the DICOM and we are currently binning it away. I had repeated
the report's claim that our input was already near-native; the EDA shows it is not.

HONEST SCOPE. Roughly half the series are finer than 0.3125 mm/px and genuinely gain; the
coarser half get upsampled and gain nothing -- but they were ALREADY being upsampled at
336, so they lose nothing either. The cost is compute, not fidelity.

ONE DELIBERATE DEVIATION FROM THE VERIFIED RECIPE. cache64 always resizes with
cv2.INTER_AREA, which is correct for downscaling but degenerates to nearest-neighbour when
UPscaling. At 336 only the coarse tail upscaled; at 448 about half the corpus does, so the
resize now picks INTER_AREA to shrink and INTER_LINEAR to enlarge. Everything else --
slot table, span, 140 mm crop, IPP . normal ordering, modality LUT, MONOCHROME1 handling,
per-series percentile[2,98] -- is byte-identical to the recipe arm10_verify matched against
the public corpus.

LAYOUT. 4,407 x 64 x 448 x 448 uint8 = 56.6 GB, over Kaggle's ~20 GB output cap, so FOUR
shards (train.csv order, rows[SHARD::4]) of ~14.2 GB. Row order within a shard is
deterministic; the trainer concatenates shards and indexes by StudyInstanceUID, and the
fold split depends only on the id SET, so it stays identical to every previous arm.
CPU only -- costs no GPU quota.
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

SHARD = 0
N_SHARDS = 4
IMG = 448
CROP_MM = 140.0
SPAN = (0.02, 0.98)
SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
           ("Coronal", 0, 8), ("Axial", -1, 12)]
DEPTH = sum(k for _, _, k in SLOTS64)
WORKERS = 4
LIMIT = int(os.environ.get("CACHE448_LIMIT", "0"))
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
    """140 mm physical centre crop, then resize to IMG.

    INTER_AREA is the correct kernel for shrinking and is what the verified recipe used at
    336, but OpenCV's INTER_AREA falls back to nearest-neighbour when enlarging. At 448
    about half the corpus enlarges, so pick by direction.
    """
    h, w = a.shape
    c = min(int(round(CROP_MM / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - c) // 2, (w - c) // 2
    crop = a[y0:y0 + c, x0:x0 + c]
    interp = cv2.INTER_AREA if c >= IMG else cv2.INTER_LINEAR
    return cv2.resize(crop, (IMG, IMG), interpolation=interp)


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
    n_up = 0
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
        loq, hiq = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
                    if valid else (0.0, 1.0))
        for a, ps in zip(arrs, pss):
            if idx >= DEPTH:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            eff = ps if ps > 0 else med
            if min(int(round(CROP_MM / max(eff, 1e-3))), min(a.shape)) < IMG:
                n_up += 1
            vol[idx] = (mm_crop_resize(aw, eff) * 255).astype(np.uint8)
            idx += 1
    mask = (vol.reshape(DEPTH, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask, n_up


_CTX = {}


def _init(by, root):
    _CTX["by"], _CTX["root"] = by, root


def _job(args):
    i, sid = args
    try:
        vol, mask, n_up = build_study(sid, _CTX["by"].get(sid, []), _CTX["root"])
        return i, vol, mask, n_up, None
    except Exception as e:
        return (i, np.zeros((DEPTH, IMG, IMG), np.uint8), np.zeros(DEPTH, np.uint8), 0,
                repr(e)[:200])


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
        f"span {SPAN} | crop {CROP_MM} mm -> {IMG} px = {CROP_MM / IMG:.4f} mm/px")
    if gb > 19:
        raise RuntimeError(f"{gb:.1f} GB exceeds Kaggle's ~20 GB output cap")

    vols = open_memmap(WORK / f"vols448_shard{SHARD}.npy", mode="w+", dtype=np.uint8,
                       shape=(n, DEPTH, IMG, IMG))
    masks = np.zeros((n, DEPTH), np.uint8)
    failed, up_total, slice_total, last = [], 0, 0, time.time()
    with ProcessPoolExecutor(WORKERS, initializer=_init,
                             initargs=(by, str(comp / "train_series"))) as pool:
        for done, (i, vol, mask, n_up, err) in enumerate(
                pool.map(_job, enumerate(uids), chunksize=4), 1):
            vols[i] = vol
            masks[i] = mask
            up_total += n_up
            slice_total += int(mask.sum())
            if err:
                failed.append((uids[i], err))
            if time.time() - last > 60 or done == n:
                el = time.time() - t0
                log(f"  {done}/{n} | {el / 60:.1f} min | eta {el / done * (n - done) / 60:.1f} min"
                    f" | failed {len(failed)} | upsampled slices "
                    f"{100 * up_total / max(slice_total, 1):.1f}%")
                last = time.time()
    vols.flush()
    del vols
    np.save(WORK / f"masks448_shard{SHARD}.npy", masks)
    np.save(WORK / f"ids448_shard{SHARD}.npy", np.asarray(uids, dtype="U"))

    check = np.load(WORK / f"vols448_shard{SHARD}.npy", mmap_mode="r")
    rng = np.random.default_rng(SHARD)
    for i in rng.choice(n, min(3, n), replace=False):
        again, _, _ = build_study(uids[i], by.get(uids[i], []), str(comp / "train_series"))
        if not np.array_equal(again, np.asarray(check[i])):
            raise RuntimeError(f"non-deterministic rebuild for study {i}")
    valid = masks.sum(1)
    meta = {"shard": SHARD, "n_shards": N_SHARDS, "n": n, "depth": DEPTH, "img": IMG,
            "mm_per_px": CROP_MM / IMG, "span": SPAN, "crop_mm": CROP_MM, "slots": SLOTS64,
            "failed": failed, "upsampled_slice_fraction": up_total / max(slice_total, 1),
            "valid_slices_min": int(valid.min()),
            "valid_slices_median": float(np.median(valid)),
            "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(WORK / f"meta448_shard{SHARD}.json", "w"), indent=1)
    log(f"DONE shard {SHARD}: {n} studies, failed {len(failed)}, "
        f"upsampled slices {100 * meta['upsampled_slice_fraction']:.1f}% | determinism PASS | "
        f"{meta['minutes']:.1f} min")


if __name__ == "__main__":
    main()
