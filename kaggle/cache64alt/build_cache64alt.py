"""cache64alt: a SECOND valid series assignment per study, for augmentation and TTA.

WHY. `build_cache64` fills each of the five slots with `cands[0]` -- the first row matching
(plane, fluid_sensitive) in train_series.csv order. That order is arbitrary. Measured over
the corpus:

    slot         0 cand       1       2      3+
    sag_fs            0    3645     753       9
    sag_nonfs       119    3671     559      58
    cor_fs            0    4048     341      18
    cor_nonfs      1124    2715     559       9
    axial             0    3313     740     354

**1,987 of 4,407 studies (45%) have at least one slot where a different series was equally
eligible**, and we always take the same arbitrary one. That is a free augmentation axis we
throw away at training time, and a real test-time augmentation axis we never use. Unlike
depth-reversal TTA -- which scored +0.000 on the LB because reversing three adjacent slices
barely changes the input -- swapping the series changes the actual pixels the model sees,
while leaving the label untouched.

WHAT THIS BUILDS. The same verified recipe, with `pick_series` taking `cands[1]` where a
second candidate exists. Only studies whose final assignment DIFFERS from the primary cache
are built, which is why this is ~14 GB rather than another 32 GB and fits one shard under
Kaggle's output cap.

    train:  sample uniformly from {primary, alt} per study per epoch (45% of studies get
            two genuinely different views; the rest are unchanged)
    test:   average the two predictions -- a TTA axis that actually moves the input

Note the slot interaction: taking `cands[1]` in an earlier slot returns `cands[0]` to the
pool for a later slot, so more than one slot can change. That is still a valid assignment
under the same rules, which is the point.

Everything else is byte-identical to build_cache64: IPP . normal ordering, modality LUT,
MONOCHROME1 inversion, per-series percentile[2,98], 140 mm centre crop, INTER_AREA to 336,
SLOTS64, span 0.02-0.98. CPU only -- costs no GPU quota.

OUTPUT (/kaggle/working):
    vols_alt.npy   (M, 64, 336, 336) uint8, M = studies whose assignment differs
    masks_alt.npy  (M, 64) uint8
    ids_alt.npy    (M,) str
    meta_alt.json  which slots changed, per study counts, failures
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

IMG = 336
CROP_MM = 140.0
SPAN = (0.02, 0.98)
SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
           ("Coronal", 0, 8), ("Axial", -1, 12)]
DEPTH = sum(k for _, _, k in SLOTS64)
WORKERS = 4
LIMIT = int(os.environ.get("ALT_LIMIT", "0"))
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


# ---------------- verified rebuild functions, unchanged from build_cache64 ----
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


def candidates(rows, plane, fluid, used):
    """Exactly build_cache64.pick_series' candidate set, returned in full."""
    cands = [r for r in rows if r["Anatomical_Plane"] == plane
             and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref
    return cands


def assign(rows, alt):
    """Slot -> SeriesInstanceUID. alt=False reproduces build_cache64 exactly."""
    out, used = [], set()
    for plane, fluid, _ in SLOTS64:
        c = candidates(rows, plane, fluid, used)
        if not c:
            out.append(None)
            continue
        r = c[1] if (alt and len(c) > 1) else c[0]
        used.add(r["SeriesInstanceUID"])
        out.append(r["SeriesInstanceUID"])
    return out


def build_study(sid, rows, series_root, alt):
    vol = np.zeros((DEPTH, IMG, IMG), np.uint8)
    idx, used = 0, set()
    for (plane, fluid, k) in SLOTS64:
        c = candidates(rows, plane, fluid, used)
        if not c:
            idx += k
            continue
        r = c[1] if (alt and len(c) > 1) else c[0]
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
            vol[idx] = (mm_crop_resize(aw, ps if ps > 0 else med) * 255).astype(np.uint8)
            idx += 1
    mask = (vol.reshape(DEPTH, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


_CTX = {}


def _init(by, root):
    _CTX["by"], _CTX["root"] = by, root


def _job(args):
    i, sid = args
    try:
        vol, mask = build_study(sid, _CTX["by"].get(sid, []), _CTX["root"], alt=True)
        return i, vol, mask, None
    except Exception as e:
        return i, np.zeros((DEPTH, IMG, IMG), np.uint8), np.zeros(DEPTH, np.uint8), repr(e)[:200]


def main():
    t0 = time.time()
    comp = find_comp()
    train = pd.read_csv(comp / "train.csv")
    ser = pd.read_csv(comp / "train_series.csv")
    ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str)
    by = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}

    # Only build studies whose ALT assignment actually differs from the primary one.
    all_uids = train.StudyInstanceUID.astype(str).tolist()
    uids, changed_slots = [], []
    for sid in all_uids:
        rows = by.get(sid, [])
        a, b = assign(rows, False), assign(rows, True)
        if a != b:
            uids.append(sid)
            changed_slots.append(sum(1 for x, y in zip(a, b) if x != y))
    if LIMIT:
        uids, changed_slots = uids[:LIMIT], changed_slots[:LIMIT]
    n = len(uids)
    gb = n * DEPTH * IMG * IMG / 1e9
    log(f"{n}/{len(all_uids)} studies have a different alt assignment "
        f"({100.0 * n / len(all_uids):.0f}%) -> ({n}, {DEPTH}, {IMG}, {IMG}) = {gb:.1f} GB")
    log(f"slots changed per study: mean {np.mean(changed_slots):.2f}, "
        f"max {max(changed_slots) if changed_slots else 0}")
    if gb > 19:
        raise RuntimeError(f"{gb:.1f} GB exceeds Kaggle's ~20 GB output cap -- shard it")

    vols = open_memmap(WORK / "vols_alt.npy", mode="w+", dtype=np.uint8,
                       shape=(n, DEPTH, IMG, IMG))
    masks = np.zeros((n, DEPTH), np.uint8)
    failed, last = [], time.time()
    with ProcessPoolExecutor(WORKERS, initializer=_init,
                             initargs=(by, str(comp / "train_series"))) as pool:
        for done, (i, vol, mask, err) in enumerate(pool.map(_job, enumerate(uids), chunksize=4), 1):
            vols[i] = vol
            masks[i] = mask
            if err:
                failed.append((uids[i], err))
            if time.time() - last > 60 or done == n:
                el = time.time() - t0
                log(f"  {done}/{n} | {el / 60:.1f} min | eta {el / done * (n - done) / 60:.1f} min"
                    f" | failed {len(failed)}")
                last = time.time()
    vols.flush()
    del vols
    np.save(WORK / "masks_alt.npy", masks)
    np.save(WORK / "ids_alt.npy", np.array(uids))

    # Determinism, same check the primary cache passed.
    check = np.load(WORK / "vols_alt.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    for i in rng.choice(n, min(3, n), replace=False):
        again, _ = build_study(uids[i], by.get(uids[i], []), str(comp / "train_series"), alt=True)
        if not np.array_equal(again, np.asarray(check[i])):
            raise RuntimeError(f"non-deterministic rebuild for study {i}")

    # And the point of the whole exercise: the alt volume must actually DIFFER.
    # If it matched the primary we would be paying for a duplicate and calling it TTA.
    valid = masks.sum(1)
    meta = {"n": n, "n_total": len(all_uids), "depth": DEPTH, "img": IMG, "span": list(SPAN),
            "crop_mm": CROP_MM, "slots": SLOTS64, "failed": failed,
            "mean_slots_changed": float(np.mean(changed_slots)) if changed_slots else 0.0,
            "valid_slices_min": int(valid.min()), "valid_slices_median": float(np.median(valid)),
            "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(WORK / "meta_alt.json", "w"), indent=1)
    log(f"DONE: {n} studies, failed {len(failed)}, valid slices/study min "
        f"{meta['valid_slices_min']} median {meta['valid_slices_median']} | determinism PASS | "
        f"{meta['minutes']:.1f} min")


if __name__ == "__main__":
    main()
