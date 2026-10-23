import os
os.environ.setdefault("FULL_NSHARDS", "16")
os.environ.setdefault("FULL_SHARD", "5")
os.environ.setdefault("FULL_LIMIT", "0")

"""full cache: EVERY slice of EVERY series, no 64-slot resampling, no dropped series.

cache64 keeps one series per (plane,fluid) slot and resamples to a fixed 64 slices @336,
span 0.02-0.98 -> it discards most native slices and all duplicate series. That shared
lossy view is why every model correlates. This build keeps it ALL:
  * every series in the study (all planes, all sequences, duplicates included)
  * every physically-ordered slice in each series (no span, no resample)
  * IMG px (default 384 > cache64's 336), physical mm crop for scale-consistency
  * per-slice metadata: plane(0 sag/1 cor/2 ax), fluid(0/1), series index, in-series pos
Variable length per study, stored CSR-style: one concatenated slice array + an index.

Output per shard (train.csv order, rows[SHARD::N_SHARDS]):
  slices_shard{S}.npy  (Ttot, IMG, IMG) uint8   -- all slices of all studies in the shard
  index_shard{S}.npy   (n_studies, 2) int64     -- [offset, count] into slices
  meta_shard{S}.npy    (Ttot, 4) float32        -- plane, fluid, series_idx, pos_norm
  ids_shard{S}.npy     (n_studies,) str
CPU only -- no GPU quota.
"""
import glob, json, os, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd, pydicom
from numpy.lib.format import open_memmap
from pydicom.pixel_data_handlers.util import apply_modality_lut

# --- per-kernel config (a shard kernel is a copy of this file with SHARD changed) ---
SHARD = int(os.environ.get("FULL_SHARD", "0"))
N_SHARDS = int(os.environ.get("FULL_NSHARDS", "16"))
LIMIT = int(os.environ.get("FULL_LIMIT", "0"))               # >0 = smoke cap on #studies
# ------------------------------------------------------------------------------------
IMG = int(os.environ.get("FULL_IMG", "448"))                 # 140mm/448 = 0.3125 mm/px = native median (cache448 analysis)
CROP_MM = float(os.environ.get("FULL_CROP_MM", "140.0"))     # the verified recipe crop; covers the knee joint
WORKERS = int(os.environ.get("FULL_WORKERS", "4"))
PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
cv2.setNumThreads(1)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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
            iop = getattr(h, "ImageOrientationPatient", None); ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                nrm = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), nrm))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None); ps = float(ps[0]) if ps is not None else 0.5
            ps_list.append(ps); recs.append((pos, f, ps))
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


def build_study_full(sid, rows, series_root):
    """All slices of all series. Returns (slices (K,IMG,IMG) uint8, meta (K,4) float32)."""
    slices, meta = [], []
    for si, r in enumerate(rows):
        plane = PLANE_ID.get(str(r["Anatomical_Plane"]), 0)
        fluid = int(r.get("Fluid_Sensitive", 0) or 0)
        files, med = order_and_meta(f"{series_root}/{sid}/{r['SeriesInstanceUID']}")
        n = len(files)
        if n == 0:
            continue
        arrs, pss = [], []
        for fp, ps in files:
            try:
                arrs.append(read_px(fp)); pss.append(ps if ps > 0 else med)
            except Exception:
                arrs.append(None); pss.append(med)
        valid = [a for a in arrs if a is not None]
        loq, hiq = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
                    if valid else (0.0, 1.0))
        for pi, (a, ps) in enumerate(zip(arrs, pss)):
            if a is None:
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            slices.append((mm_crop_resize(aw, ps) * 255).astype(np.uint8))
            meta.append((plane, fluid, si, pi / max(1, n - 1)))
    if not slices:
        return np.zeros((0, IMG, IMG), np.uint8), np.zeros((0, 4), np.float32)
    return np.stack(slices), np.asarray(meta, np.float32)


_CTX = {}
def _init(by, root): _CTX["by"], _CTX["root"] = by, root
def _job(args):
    i, sid = args
    try:
        s, m = build_study_full(sid, _CTX["by"].get(sid, []), _CTX["root"])
        return i, s, m, None
    except Exception as e:
        return i, np.zeros((0, IMG, IMG), np.uint8), np.zeros((0, 4), np.float32), repr(e)[:200]


def main():
    t0 = time.time()
    comp = find_comp()
    train = pd.read_csv(comp / "train.csv")
    uids = train.StudyInstanceUID.astype(str).tolist()[SHARD::N_SHARDS]
    if LIMIT:
        uids = uids[:LIMIT]
    ser = pd.read_csv(comp / "train_series.csv"); ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str)
    by = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}
    work = Path(os.environ.get("FULL_OUT", "/kaggle/working")); work.mkdir(parents=True, exist_ok=True)
    n = len(uids)
    log(f"shard {SHARD}/{N_SHARDS}: {n} studies | IMG {IMG} crop {CROP_MM}mm | collecting all slices...")

    # pass 1: build all, hold per-study arrays (memory: a shard of ~550 studies x ~150 slices x 384^2
    # uint8 ~ 12 GB -- fits the 4-vCPU Kaggle box; if tight, lower FULL_NSHARDS-> more shards)
    per_study, metas, counts, failed = {}, {}, [], []
    with ProcessPoolExecutor(WORKERS, initializer=_init,
                             initargs=(by, str(comp / "train_series"))) as pool:
        last = time.time()
        for done, (i, s, m, err) in enumerate(pool.map(_job, enumerate(uids), chunksize=2), 1):
            per_study[i] = s; metas[i] = m; counts.append(s.shape[0])
            if err:
                failed.append((uids[i], err))
            if time.time() - last > 60 or done == n:
                el = time.time() - t0
                log(f"  {done}/{n} | {el/60:.1f} min | eta {el/done*(n-done)/60:.1f} min | failed {len(failed)}")
                last = time.time()
    total = int(sum(counts))
    gb = total * IMG * IMG / 1e9
    log(f"shard {SHARD}: {total} slices total ({gb:.1f} GB) | slices/study med {int(np.median(counts))} "
        f"max {int(max(counts)) if counts else 0} | projected full {gb*N_SHARDS:.0f} GB")

    slices = open_memmap(work / f"slices_shard{SHARD}.npy", mode="w+", dtype=np.uint8, shape=(total, IMG, IMG))
    meta = np.zeros((total, 4), np.float32)
    index = np.zeros((n, 2), np.int64)
    off = 0
    for i in range(n):
        s = per_study[i]; c = s.shape[0]
        index[i] = (off, c)
        if c:
            slices[off:off + c] = s; meta[off:off + c] = metas[i]
        off += c
    slices.flush(); del slices
    np.save(work / f"meta_shard{SHARD}.npy", meta)
    np.save(work / f"index_shard{SHARD}.npy", index)
    np.save(work / f"ids_shard{SHARD}.npy", np.asarray(uids, dtype="U"))

    # determinism spot-check: rebuild a few studies and compare (same guard cache448 uses)
    check = np.load(work / f"slices_shard{SHARD}.npy", mmap_mode="r")
    rng = np.random.default_rng(SHARD)
    for i in rng.choice(n, min(3, n), replace=False):
        s2, _ = build_study_full(uids[i], by.get(uids[i], []), str(comp / "train_series"))
        o, c = index[i]
        if not np.array_equal(s2, np.asarray(check[o:o + c])):
            raise RuntimeError(f"non-deterministic rebuild for study {uids[i]}")

    json.dump({"shard": SHARD, "n_shards": N_SHARDS, "n_studies": n, "total_slices": total,
               "img": IMG, "crop_mm": CROP_MM, "gb": gb, "projected_full_gb": gb * N_SHARDS,
               "failed": failed, "slices_per_study_median": int(np.median(counts)) if counts else 0,
               "slices_per_study_max": int(max(counts)) if counts else 0,
               "minutes": (time.time() - t0) / 60},
              open(work / f"meta_full_shard{SHARD}.json", "w"), indent=1)
    log(f"DONE shard {SHARD}: {n} studies, {total} slices, {gb:.1f} GB, failed {len(failed)}, "
        f"determinism PASS, {(time.time()-t0)/60:.1f} min")
    print("FULL_BUILD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
