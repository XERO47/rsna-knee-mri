"""geom64: the physical-space sidecar cache64 threw away, so arm30 can place every PATCH in
patient coordinates.

WHY THIS EXISTS. `build_cache64.py` computes IPP . normal to ORDER slices and then discards
the geometry: the cache holds only rendered uint8 volumes, masks and ids. arm30's cross-plane
consensus term needs the opposite — for a patch at (row, col) of depth index d, the LPS
coordinate of that patch's centre — so that a sagittal patch scoring high for Fracture can be
checked against a coronal patch scoring high for Fracture AT THE SAME PHYSICAL POINT. Without
this file that term cannot be computed, only approximated.

WHAT IT RECORDS, per study and per one of the 64 depth slots, header-only (no pixel decode):

    iop  (6)  ImageOrientationPatient -- column direction then row direction
    ipp  (3)  ImagePositionPatient of the slice actually rendered into that slot
    ps   (1)  in-plane PixelSpacing in mm
    hw   (2)  Rows, Columns of the source image, which fix the centre crop
    plane(1)  0 sagittal / 1 coronal / 2 axial, from the slot definition

From those, a rendered pixel maps to patient space exactly as the render did it. cache64 takes
a centre crop of c = min(round(CROP_MM / ps), h, w) at (y0, x0) = ((h-c)//2, (w-c)//2) and
resizes it to IMG, so rendered (R, C) came from source

    src_row = y0 + (R + 0.5) * c / IMG - 0.5
    src_col = x0 + (C + 0.5) * c / IMG - 0.5

and DICOM places that at

    P = ipp + src_col * ps * iop[0:3] + src_row * ps * iop[3:6]

THE REPLAY MUST BE EXACT. This kernel re-runs cache64's slot selection and slice picking --
`pick_series` in CSV row order, `order_and_meta` sorted by IPP . normal, then the same
`linspace(lo, hi, k)` over SPAN -- against the same competition data, so slot d here is the
slice rendered into slot d there. Two things can legitimately differ: cache64 leaves a slot
ZERO when the pixel read of an otherwise valid header failed, and it stops at DEPTH. So this
kernel writes a `filled` flag from the headers and the build is VERIFIED against the real
cache64 masks: every slot the cache marks valid must be one this kernel also filled. The
reverse is allowed (a header we found whose pixels cache64 could not decode) and is counted.

Header-only, CPU, no GPU quota.
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

N_SHARDS = 2
IMG = 336
CROP_MM = 140.0
SPAN = (0.02, 0.98)
SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
           ("Coronal", 0, 8), ("Axial", -1, 12)]
DEPTH = sum(k for _, _, k in SLOTS64)
PLANE_CODE = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
WORK = Path(os.environ.get("GEOM_WORK", "/kaggle/working"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_comp(root=None):
    for d, dirs, fs in os.walk(root or "/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if "train_series.csv" in fs:
            return Path(d)
    raise FileNotFoundError("competition data not attached")


def order_and_meta(sdir):
    """cache64's ordering, but keeping the geometry instead of discarding it."""
    recs, ps_list = [], []
    for f in glob.glob(sdir + "/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                iop = np.array(iop, float)
                ipp = np.array(ipp, float)
                n = np.cross(iop[:3], iop[3:])
                pos = float(np.dot(ipp, n))
            else:
                iop = np.zeros(6)
                ipp = np.zeros(3)
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            rows = int(getattr(h, "Rows", 0) or 0)
            cols = int(getattr(h, "Columns", 0) or 0)
            ps_list.append(ps)
            recs.append((pos, f, ps, iop, ipp, rows, cols))
        except Exception:
            recs.append((0.0, f, 0.5, np.zeros(6), np.zeros(3), 0, 0))
    recs.sort(key=lambda x: x[0])
    med = float(np.median(ps_list)) if ps_list else 0.5
    return [r[1:] for r in recs], med


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane
             and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def geom_study(sid, rows, series_root):
    """Replay cache64's slot filling, recording geometry per slot instead of pixels."""
    iop = np.zeros((DEPTH, 6), np.float32)
    ipp = np.zeros((DEPTH, 3), np.float32)
    ps = np.zeros(DEPTH, np.float32)
    hw = np.zeros((DEPTH, 2), np.int16)
    plane_of = np.full(DEPTH, -1, np.int8)
    filled = np.zeros(DEPTH, np.uint8)
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
        for p in picks:
            if idx >= DEPTH:
                break
            _, p_ps, p_iop, p_ipp, p_rows, p_cols = files[min(p, n - 1)]
            iop[idx] = p_iop
            ipp[idx] = p_ipp
            ps[idx] = p_ps if p_ps > 0 else med
            hw[idx] = (p_rows, p_cols)
            plane_of[idx] = PLANE_CODE[plane]
            # a slot is usable only if the geometry is real and the crop is defined
            filled[idx] = 1 if (np.abs(p_iop).sum() > 0 and p_rows > 0 and p_cols > 0) else 0
            idx += 1
    return iop, ipp, ps, hw, plane_of, filled


def patch_lps(iop, ipp, ps, hw, rows, cols, img=IMG, crop_mm=CROP_MM):
    """Vectorised inverse of cache64's render: rendered (row, col) -> LPS millimetres.

    Kept here, beside the builder, because it is the only place the crop arithmetic is
    written down; arm30 imports it so the two cannot drift apart."""
    h, w = hw[..., 0].astype(np.float64), hw[..., 1].astype(np.float64)
    c = np.minimum(np.round(crop_mm / np.maximum(ps, 1e-3)), np.minimum(h, w))
    y0, x0 = (h - c) // 2, (w - c) // 2
    src_row = y0 + (rows + 0.5) * c / img - 0.5
    src_col = x0 + (cols + 0.5) * c / img - 0.5
    return (ipp + src_col[..., None] * ps[..., None] * iop[..., 0:3]
            + src_row[..., None] * ps[..., None] * iop[..., 3:6])


def shard_study_ids(train, shard, n_shards=N_SHARDS):
    uids = sorted(train.StudyInstanceUID.astype(str).unique())
    return [u for i, u in enumerate(uids) if i % n_shards == shard]


def main(args):
    t0 = time.time()
    comp = find_comp(args.input)
    WORK.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(comp / "train.csv")
    ser = pd.read_csv(comp / "train_series.csv")
    ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str)
    by = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}
    uids = shard_study_ids(train, args.shard)
    log(f"shard {args.shard}/{N_SHARDS}: {len(uids)} studies, header-only, {DEPTH} slots each")

    n = len(uids)
    out = {"iop": np.zeros((n, DEPTH, 6), np.float32),
           "ipp": np.zeros((n, DEPTH, 3), np.float32),
           "ps": np.zeros((n, DEPTH), np.float32),
           "hw": np.zeros((n, DEPTH, 2), np.int16),
           "plane": np.full((n, DEPTH), -1, np.int8),
           "filled": np.zeros((n, DEPTH), np.uint8)}
    root = str(comp / "train_series")
    for i, sid in enumerate(uids):
        a, b, c, d, e, f = geom_study(sid, by.get(sid, []), root)
        out["iop"][i], out["ipp"][i], out["ps"][i] = a, b, c
        out["hw"][i], out["plane"][i], out["filled"][i] = d, e, f
        if (i + 1) % 100 == 0 or i + 1 == n:
            el = time.time() - t0
            log(f"  {i + 1}/{n} | {el / 60:.1f} min | eta "
                f"{el / (i + 1) * (n - i - 1) / 60:.1f} min")

    np.savez_compressed(WORK / f"geom64_shard{args.shard}.npz",
                        ids=np.asarray(uids, dtype="U"), **out)

    # ---- verification against the real cache64 masks -------------------------------
    report = {"shard": args.shard, "n": n, "depth": DEPTH,
              "filled_mean": float(out["filled"].mean())}
    mpath = None
    for d, _, fs in os.walk(args.input or "/kaggle/input"):
        if f"masks_shard{args.shard}.npy" in fs:
            mpath = Path(d)
            break
    if mpath is None:
        log("cache64 masks not attached -- SKIPPING verification (do not trust this file)")
        report["verified"] = False
    else:
        cids = np.load(mpath / f"ids_shard{args.shard}.npy", allow_pickle=True).astype(str)
        masks = np.load(mpath / f"masks_shard{args.shard}.npy").astype(bool)
        pos = {u: j for j, u in enumerate(cids)}
        if [u for u in uids if u not in pos]:
            raise RuntimeError("shard study lists differ between geom64 and cache64")
        rows = np.array([pos[u] for u in uids])
        cm = masks[rows]
        gf = out["filled"].astype(bool)
        missing = int((cm & ~gf).sum())          # cache rendered a slot we found no header for
        extra = int((~cm & gf).sum())            # header found, pixels failed to decode
        log(f"verification: cache64 valid slots {int(cm.sum())} | geom64 filled "
            f"{int(gf.sum())} | cache-valid-but-no-geometry {missing} | "
            f"geometry-but-cache-empty {extra}")
        if missing:
            raise RuntimeError(f"{missing} slots are valid in cache64 but have no geometry "
                               f"here -- the replay diverged and the sidecar is unusable")
        report.update(verified=True, cache_valid=int(cm.sum()), geom_filled=int(gf.sum()),
                      cache_valid_without_geometry=missing, geometry_without_pixels=extra)

        # a physical sanity check: the mapped patch corners must span ~CROP_MM
        ok = gf & cm
        si, di = np.where(ok)
        take = np.arange(0, len(si), max(1, len(si) // 2000))
        si, di = si[take], di[take]
        g = {k: out[k][si, di] for k in ("iop", "ipp", "ps", "hw")}
        c0 = patch_lps(g["iop"], g["ipp"], g["ps"], g["hw"],
                       np.zeros(len(si)), np.zeros(len(si)))
        c1 = patch_lps(g["iop"], g["ipp"], g["ps"], g["hw"],
                       np.full(len(si), IMG - 1.0), np.full(len(si), IMG - 1.0))
        diag = np.linalg.norm(c1 - c0, axis=1)
        exp = CROP_MM * np.sqrt(2) * (IMG - 1) / IMG
        log(f"patch mapping: rendered-corner-to-corner distance median {np.median(diag):.1f} mm "
            f"(expected ~{exp:.1f} for a {CROP_MM:.0f} mm crop), "
            f"p05 {np.percentile(diag, 5):.1f} p95 {np.percentile(diag, 95):.1f}")
        if not (0.8 * exp < np.median(diag) < 1.2 * exp):
            raise RuntimeError("mapped patch extent does not match the physical crop size")
        report["corner_distance_median_mm"] = float(np.median(diag))
    report["minutes"] = (time.time() - t0) / 60
    json.dump(report, open(WORK / f"geom64_report_shard{args.shard}.json", "w"), indent=1)
    log(f"wrote geom64_shard{args.shard}.npz in {report['minutes']:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=int(os.environ.get("GEOM_SHARD", "1")))
    ap.add_argument("--input", default=os.environ.get("GEOM_INPUT"))
    main(ap.parse_args())
