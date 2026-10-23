"""cache64phase: the SAME series, sampled at a complementary depth phase -- slices the model
has literally never been shown.

WHY THIS IS A SEPARATE CACHE AND NOT A TRAINING-TIME AUGMENTATION. cache64's depth sampling is
baked in at BUILD time: for each slot it takes `linspace(lo, hi, k)` over the 2-98% span of the
series and renders exactly those k slices. Everything downstream -- every arm from arm10 to
arm30 -- then samples window centres from those 64 rendered slices. Jittering centres at train
time therefore re-picks among slices the model has already seen; it cannot show new anatomy.
The EDA's unresolved concern is one level below that: a 320-slice thin-slice series contributes
18 rendered slices, 5.6% of the images actually acquired, and a focal lesion in the other 94%
is invisible to every model we have trained.

This cache renders the complementary phase: the same series, the same slot layout, the same
crop and normalisation, with the linspace offset by HALF A STEP.

    primary   picks   lo + t * step                     t = 0 .. k-1
    phase     picks   lo + t * step + step/2

which is exactly the interlaced pair "slices 0, 2, 4, ..." and "slices 1, 3, 5, ..." after
resampling the physical depth interval. For a 320-slice series the half step is ~9 slices, so
every rendered image is new. For a series with barely more slices than k, the shift rounds to
the same indices and the study is skipped -- only studies whose rendered slice SET differs are
stored, as in cache64alt, which keeps the output inside Kaggle's cap.

    train:  sample uniformly over {primary, phase} (and {primary, alt} independently)
    test:   a third view, scored separately rather than averaged in blindly

Everything else is byte-identical to build_cache64: IPP . normal ordering, modality LUT,
MONOCHROME1 inversion, per-series percentile[2, 98], 140 mm centre crop, INTER_AREA to 336,
SLOTS64, span 0.02-0.98. CPU only -- costs no GPU quota.

OUTPUT (/kaggle/working):
    vols_phase.npy   (M, 64, 336, 336) uint8, M = studies whose rendered slices differ
    masks_phase.npy  (M, 64) uint8
    ids_phase.npy    (M,) study ids
    meta_phase.json  counts, and how many slots changed per study
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


N_SHARDS = int(os.environ.get("PHASE_N_SHARDS", "2"))
SHARD = int(os.environ.get("PHASE_SHARD", "0"))


def picks_for(n, k, phase):
    """The slice indices build_cache64 renders for a slot, optionally at the half-step phase.

    Kept as its own function so the selection pass below and the render pass use the SAME
    arithmetic -- otherwise a study could be selected as "different" and then rendered
    identically, or vice versa."""
    if n <= 1:
        return [0] * k
    lo, hi = int(n * SPAN[0]), int(n * SPAN[1]) - 1
    hi = max(hi, lo)
    base = np.linspace(lo, hi, k)
    if phase:
        step = (hi - lo) / max(k - 1, 1)
        base = np.clip(base + step / 2.0, lo, hi)
    return base.round().astype(int).tolist()


def assign(rows, phase, series_root=None, sid=None):
    """Slot -> (SeriesInstanceUID, picked slice indices).

    For the PHASE cache the series never changes, so a study is only worth storing if the
    half-step shift lands on different slice INDICES -- which it does not when the series has
    barely more slices than the slot wants. Counting slices needs the directory listing, not
    the headers, so this stays cheap."""
    out, used = [], set()
    for plane, fluid, k in SLOTS64:
        c = candidates(rows, plane, fluid, used)
        if not c:
            out.append(None)
            continue
        r = c[0]                      # always the primary series: only the phase differs
        used.add(r["SeriesInstanceUID"])
        n = 0
        if series_root is not None:
            n = len(glob.glob(f"{series_root}/{sid}/{r['SeriesInstanceUID']}/*.dcm"))
        out.append((r["SeriesInstanceUID"], tuple(picks_for(n, k, phase))))
    return out


def build_study(sid, rows, series_root, phase):
    vol = np.zeros((DEPTH, IMG, IMG), np.uint8)
    idx, used = 0, set()
    for (plane, fluid, k) in SLOTS64:
        c = candidates(rows, plane, fluid, used)
        if not c:
            idx += k
            continue
        r = c[0]                      # always the primary series: only the phase differs
        used.add(r["SeriesInstanceUID"])
        files, med = order_and_meta(f"{series_root}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * SPAN[0]), int(n * SPAN[1]) - 1
        hi = max(hi, lo)
        picks = picks_for(n, k, phase)
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
        vol, mask = build_study(sid, _CTX["by"].get(sid, []), _CTX["root"], phase=True)
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

    # Only build studies whose rendered SLICE SET actually differs at the half-step phase.
    # A series with barely more slices than its slot wants rounds to the same indices and is
    # skipped, which is both correct and what keeps this inside the output cap.
    root = str(comp / "train_series")
    all_uids = train.StudyInstanceUID.astype(str).tolist()
    uids, changed_slots = [], []
    for sid in all_uids:
        rows = by.get(sid, [])
        a = assign(rows, False, root, sid)
        b = assign(rows, True, root, sid)
        # The criterion is NOT "the picks differ" -- it is "the phase renders at least one
        # slice the primary never rendered". For a series with barely more slices than its
        # slot wants, the 2-98% span leaves fewer eligible slices than picks, so the primary
        # already covers all of them and a half-step shift only reshuffles duplicates. Those
        # studies show the model nothing new, and storing them would dilute the very axis this
        # cache exists to isolate.
        new_slots = 0
        for sa, sb in zip(a, b):
            if sa is None or sb is None:
                continue
            if set(sb[1]) - set(sa[1]):
                new_slots += 1
        if new_slots:
            uids.append(sid)
            changed_slots.append(new_slots)
    # Shard AFTER selection: many more studies qualify here than for cache64alt (any series
    # with roughly twice its slot's slice count contributes unseen slices), so a single output
    # would exceed Kaggle's ~20 GB cap.
    if N_SHARDS > 1:
        uids = uids[SHARD::N_SHARDS]
        changed_slots = changed_slots[SHARD::N_SHARDS]
        log(f"shard {SHARD}/{N_SHARDS}: {len(uids)} of the qualifying studies")
    if LIMIT:
        uids, changed_slots = uids[:LIMIT], changed_slots[:LIMIT]
    n = len(uids)
    gb = n * DEPTH * IMG * IMG / 1e9
    log(f"{n}/{len(all_uids)} studies render DIFFERENT SLICES at the half-step phase "
        f"({100.0 * n / len(all_uids):.0f}%) -> ({n}, {DEPTH}, {IMG}, {IMG}) = {gb:.1f} GB")
    log(f"slots contributing UNSEEN slices, per study: mean {np.mean(changed_slots):.2f}, "
        f"max {max(changed_slots) if changed_slots else 0}")
    if gb > 19:
        raise RuntimeError(f"{gb:.1f} GB exceeds Kaggle's ~20 GB output cap -- shard it")

    vols = open_memmap(WORK / f"vols_phase_shard{SHARD}.npy", mode="w+", dtype=np.uint8,
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
    np.save(WORK / f"masks_phase_shard{SHARD}.npy", masks)
    np.save(WORK / f"ids_phase_shard{SHARD}.npy", np.array(uids))

    # Determinism, same check the primary cache passed.
    check = np.load(WORK / f"vols_phase_shard{SHARD}.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    for i in rng.choice(n, min(3, n), replace=False):
        again, _ = build_study(uids[i], by.get(uids[i], []), str(comp / "train_series"), phase=True)
        if not np.array_equal(again, np.asarray(check[i])):
            raise RuntimeError(f"non-deterministic rebuild for study {i}")

    # And the point of the whole exercise: the phase volume must actually DIFFER.
    # If it matched the primary we would be paying for a duplicate and calling it TTA.
    valid = masks.sum(1)
    meta = {"n": n, "n_total": len(all_uids), "depth": DEPTH, "img": IMG, "span": list(SPAN),
            "crop_mm": CROP_MM, "slots": SLOTS64, "failed": failed,
            "mean_slots_changed": float(np.mean(changed_slots)) if changed_slots else 0.0,
            "valid_slices_min": int(valid.min()), "valid_slices_median": float(np.median(valid)),
            "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(WORK / f"meta_phase_shard{SHARD}.json", "w"), indent=1)
    log(f"DONE: {n} studies, failed {len(failed)}, valid slices/study min "
        f"{meta['valid_slices_min']} median {meta['valid_slices_median']} | determinism PASS | "
        f"{meta['minutes']:.1f} min")


if __name__ == "__main__":
    main()
