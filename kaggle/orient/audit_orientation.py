"""orient: does our corpus address anatomy consistently? CPU-only audit, no GPU quota.

WHY THIS RUNS BEFORE ANY MORE TRAINING.
Our weakest findings are the *lateral* ones -- Lateral OA .796 vs Medial OA .968
(gap .172), Lateral Meniscus .856 vs Medial Meniscus .945 (gap .089) -- on
structures of near-identical size and appearance. That pattern says the model
knows the lesion is there and gets the SIDE wrong, i.e. an attribution problem,
not a resolution problem.

Now look at how we build the corpus (kaggle/cache64_s0/build_cache64.py, the
recipe verified pixel-exact against the public one):

    n   = cross(iop[:3], iop[3:])        # <- sign set by the scanner's row/col
    pos = dot(ipp, n)                    #    convention, never canonicalised
    recs.sort(key=lambda x: x[0])        # <- ascending in an arbitrary direction

and nowhere in the pipeline is knee laterality read or used. So:

  * On SAGITTAL series the through-plane axis IS the medial-lateral axis. The
    compartment is literally the slice index. But our stack runs medial->lateral
    when (n . patient-left) > 0 and lateral->medial otherwise, and that flips
    again between a left and a right knee. Two uncorrelated coin flips.
  * On CORONAL/AXIAL series medial-lateral is the in-plane horizontal axis, so
    the same two coin flips become a horizontal mirror.
  * On SAGITTAL the in-plane horizontal axis is anterior-posterior, so if its
    sign varies the patella lands on either side of the image -- which would
    hit PF OA (.831) the same way.

This kernel MEASURES those flips. It does not change anything. It reads only
DICOM headers (no pixels) for exactly the five series cache64 picks per study,
and emits, per study per slot, the flags that would canonicalise it.

WHAT WE DO WITH THE ANSWER.
If the directions really are ~50/50, canonicalisation is a load-time transform
on the EXISTING cache -- reverse a slot's slice range, mirror its width axis --
so it costs no rebuild, and it is the cheap version of the "canonicalize ->
named anatomical regions" recommendation from the 2026-09-19 deep-research
report, without that report's 50-100 hand-annotated landmark studies.
If instead everything is already consistent, the whole compartment programme
is dead and we spend the quota elsewhere. Either way this run costs nothing.

It also predicts something falsifiable about a result we already have: arm14's
position-embedding + BiGRU head lost .008 on 870 OOF studies despite starting
as an exact zero-initialised copy of the plain head. A depth axis whose
direction is a coin flip would explain that exactly, while leaving the
permutation-invariant attention head untouched. If this audit finds the flips,
re-running that A/B on a canonical axis is a real test, not a third guess.

CONVENTION. DICOM patient space is LPS: +X patient-left, +Y posterior,
+Z superior. ImageOrientationPatient is [row_dir(3), col_dir(3)] where row_dir
is the direction of increasing COLUMN index (image horizontal) and col_dir the
direction of increasing ROW index (image vertical). PixelSpacing is
[between-rows, between-columns].

Canonical targets we flag against (choice of sign is arbitrary, consistency is
the point; laterality is folded in so that every knee reads the same way):
    Sagittal  horizontal -> anterior (-Y)        vertical -> inferior (-Z)
              depth      -> lateral to medial
    Coronal   horizontal -> lateral to medial    vertical -> inferior (-Z)
    Axial     horizontal -> lateral to medial    vertical -> posterior (+Y)

OUTPUT (/kaggle/working):
    orient.npz    per-study flags, aligned to train.csv order
    orient.csv    the same, human-readable
    orient_summary.json
"""
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

# cache64's slot table. SLOTS44 uses the same five slots in the same order with
# different counts, and pick_series consumes `used` in slot order, so the
# slot -> series mapping these flags describe is correct for BOTH corpora.
SLOTS = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", -1)]
SLOT_NAMES = ["sag_fs", "sag_nonfs", "cor_fs", "cor_nonfs", "axial"]
NSLOT = len(SLOTS)
WORKERS = 4
LIMIT = int(os.environ.get("ORIENT_LIMIT", "0"))      # >0 for a smoke run
WORK = Path("/kaggle/working")

EX = np.array([1.0, 0.0, 0.0])      # patient left
EY = np.array([0.0, 1.0, 0.0])      # patient posterior
EZ = np.array([0.0, 0.0, 1.0])      # patient superior


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_comp():
    for d, dirs, fs in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if "train.csv" in fs and "train_series.csv" in fs:
            return Path(d)
    raise FileNotFoundError("competition data not attached")


def pick_series(rows, plane, fluid, used):
    """Byte-identical to build_cache64.pick_series -- the slot must resolve to the
    same series the cache was built from, or these flags describe the wrong image."""
    cands = [r for r in rows if r["Anatomical_Plane"] == plane
             and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def read_series_geometry(sdir):
    """Headers only. Returns the geometry of the series as cache64 would order it."""
    recs = []
    lat_tag = ""
    desc = ""
    for f in glob.glob(sdir + "/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        iop = getattr(h, "ImageOrientationPatient", None)
        ipp = getattr(h, "ImagePositionPatient", None)
        if iop is None or ipp is None or len(iop) != 6:
            continue
        ps = getattr(h, "PixelSpacing", None)
        ps = (float(ps[0]), float(ps[1])) if ps is not None else (0.5, 0.5)
        rows_ = int(getattr(h, "Rows", 0) or 0)
        cols_ = int(getattr(h, "Columns", 0) or 0)
        if not lat_tag:
            lat_tag = str(getattr(h, "Laterality", "")
                          or getattr(h, "ImageLaterality", "") or "")
        if not desc:
            desc = str(getattr(h, "SeriesDescription", "") or "")
        recs.append((np.array(iop, float), np.array(ipp, float), ps, rows_, cols_))
    if not recs:
        return None

    iop = recs[0][0]
    rdir = iop[:3] / (np.linalg.norm(iop[:3]) + 1e-9)     # increasing COLUMN index
    cdir = iop[3:] / (np.linalg.norm(iop[3:]) + 1e-9)     # increasing ROW index
    n = np.cross(rdir, cdir)
    n = n / (np.linalg.norm(n) + 1e-9)

    # cache64 sorts ascending by dot(ipp, n); reproduce that to get the true ends.
    pos = np.array([float(np.dot(r[1], n)) for r in recs])
    order = np.argsort(pos)

    # Volume centroid in patient space: the mean slice centre.
    ctrs = []
    for (_, ipp_i, ps_i, rr, cc) in recs:
        ctrs.append(ipp_i + (cc - 1) / 2.0 * ps_i[1] * rdir + (rr - 1) / 2.0 * ps_i[0] * cdir)
    ctr = np.mean(np.array(ctrs), axis=0)

    return dict(rdir=rdir, cdir=cdir, n=n, ctr=ctr, nslice=len(recs),
                span=float(pos[order[-1]] - pos[order[0]]),
                lat_tag=lat_tag.strip().upper(), desc=desc)


def geom_plane(n):
    """Which anatomical plane the geometry says this is, and how axis-aligned it is."""
    c = np.abs([np.dot(n, EX), np.dot(n, EY), np.dot(n, EZ)])
    k = int(np.argmax(c))
    return ["Sagittal", "Coronal", "Axial"][k], float(c[k])


def axis_flags(plane, rdir, cdir, n):
    """The corrections that put one series into the canonical frame, BEFORE laterality.

    Returns (rev_depth, flip_h, flip_v, transpose). Pure function of the direction
    cosines so it can be unit-tested without DICOM files -- see _selftest().
    """
    if plane == "Sagittal":
        # depth axis is medial-lateral: make ascending dot(ipp,n) run toward patient-left
        rev = bool(np.dot(n, EX) < 0)
        # in-plane horizontal -> anterior (-Y), vertical -> inferior (-Z)
        fh = bool(np.dot(rdir, -EY) < 0)
        fv = bool(np.dot(cdir, -EZ) < 0)
        tp = bool(abs(np.dot(rdir, EY)) < abs(np.dot(rdir, EZ)))
    elif plane == "Coronal":
        rev = False
        fh = bool(np.dot(rdir, EX) < 0)
        fv = bool(np.dot(cdir, -EZ) < 0)
        tp = bool(abs(np.dot(rdir, EX)) < abs(np.dot(rdir, EZ)))
    else:  # Axial
        rev = False
        fh = bool(np.dot(rdir, EX) < 0)
        fv = bool(np.dot(cdir, EY) < 0)
        tp = bool(abs(np.dot(rdir, EX)) < abs(np.dot(rdir, EY)))
    return rev, fh, fv, tp


def study_flags(args):
    sid, rows, series_root = args
    out = dict(
        sid=sid,
        lat_x=np.zeros(NSLOT, np.float32),
        lat_tag=[""] * NSLOT,
        desc=[""] * NSLOT,
        valid=np.zeros(NSLOT, bool),
        rev_depth_raw=np.zeros(NSLOT, bool),    # before laterality is applied
        flip_h_raw=np.zeros(NSLOT, bool),
        flip_v=np.zeros(NSLOT, bool),
        transpose=np.zeros(NSLOT, bool),
        plane_cos=np.zeros(NSLOT, np.float32),
        plane_agree=np.zeros(NSLOT, bool),
        nslice=np.zeros(NSLOT, np.int16),
    )
    used = set()
    for j, (plane, fluid) in enumerate(SLOTS):
        r = pick_series(rows, plane, fluid, used)
        if r is None:
            continue
        used.add(r["SeriesInstanceUID"])
        g = read_series_geometry(f"{series_root}/{sid}/{r['SeriesInstanceUID']}")
        if g is None:
            continue
        out["valid"][j] = True
        out["nslice"][j] = min(g["nslice"], 32767)
        out["lat_x"][j] = g["ctr"][0]
        out["lat_tag"][j] = g["lat_tag"]
        out["desc"][j] = g["desc"]
        gp, cos = geom_plane(g["n"])
        out["plane_cos"][j] = cos
        out["plane_agree"][j] = (gp == plane)

        rev, fh, fv, tp = axis_flags(plane, g["rdir"], g["cdir"], g["n"])
        out["rev_depth_raw"][j] = rev
        out["flip_h_raw"][j] = fh
        out["flip_v"][j] = fv
        out["transpose"][j] = tp
    return out


def _selftest():
    """Sign conventions are the whole product here, so check them against knees whose
    answer is known by hand. Runs on import-free numpy only; called at kernel start."""
    def iop(r, c):
        return np.array(r, float), np.array(c, float), np.cross(np.array(r, float),
                                                                np.array(c, float))

    # Most common MR sagittal: columns run posterior, rows run inferior -> n = -X.
    r, c, n = iop([0, 1, 0], [0, 0, -1])
    assert np.allclose(n, [-1, 0, 0]), n
    rev, fh, fv, tp = axis_flags("Sagittal", r, c, n)
    # ascending dot(ipp,-X) walks toward patient-RIGHT, so it must be reversed;
    # horizontal points posterior but canonical is anterior, so mirror; vertical is
    # already inferior; axes are not swapped.
    assert (rev, fh, fv, tp) == (True, True, False, False), (rev, fh, fv, tp)

    # Same slab acquired with the opposite through-plane sense -> n = +X, no reversal.
    r, c, n = iop([0, -1, 0], [0, 0, -1])
    assert np.allclose(n, [1, 0, 0]), n
    assert axis_flags("Sagittal", r, c, n)[:2] == (False, False)

    # Standard coronal and axial need nothing.
    r, c, n = iop([1, 0, 0], [0, 0, -1])
    assert axis_flags("Coronal", r, c, n) == (False, False, False, False)
    r, c, n = iop([1, 0, 0], [0, 1, 0])
    assert axis_flags("Axial", r, c, n) == (False, False, False, False)
    # Left-right mirrored coronal.
    r, c, n = iop([-1, 0, 0], [0, 0, -1])
    assert axis_flags("Coronal", r, c, n)[1] is True

    # geom_plane agrees with the normal it is given.
    assert geom_plane(np.array([-1.0, 0, 0]))[0] == "Sagittal"
    assert geom_plane(np.array([0, 1.0, 0]))[0] == "Coronal"
    assert geom_plane(np.array([0, 0, 1.0]))[0] == "Axial"

    # Laterality folding: a right knee ordered toward patient-left runs lateral->medial
    # (keep); the same geometry on a left knee runs medial->lateral (reverse again).
    for is_left, raw, want in [(False, True, True), (True, True, False),
                               (False, False, False), (True, False, True)]:
        assert (raw ^ is_left) == want, (is_left, raw)
    print("selftest OK: orientation sign conventions")


def main():
    _selftest()
    comp = find_comp()
    series_root = str(comp / "train_series")
    log(f"competition data at {comp}")

    tr = pd.read_csv(comp / "train.csv")
    ser = pd.read_csv(comp / "train_series.csv")
    sids = tr["StudyInstanceUID"].astype(str).tolist()
    if LIMIT:
        sids = sids[:LIMIT]
    by_sid = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    log(f"{len(sids)} studies, {len(ser)} series")

    jobs = [(s, by_sid.get(s, []), series_root) for s in sids]
    res = []
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(study_flags, jobs, chunksize=8)):
            res.append(r)
            if (i + 1) % 250 == 0:
                el = time.time() - t0
                log(f"{i + 1}/{len(jobs)}  {el / 60:.1f} min  "
                    f"eta {(el / (i + 1) * (len(jobs) - i - 1)) / 60:.1f} min")
    log(f"headers read in {(time.time() - t0) / 60:.1f} min")

    N = len(res)
    valid = np.array([r["valid"] for r in res])
    lat_x = np.array([r["lat_x"] for r in res])
    rev_raw = np.array([r["rev_depth_raw"] for r in res])
    fh_raw = np.array([r["flip_h_raw"] for r in res])
    fv = np.array([r["flip_v"] for r in res])
    tp = np.array([r["transpose"] for r in res])
    pcos = np.array([r["plane_cos"] for r in res])
    pagree = np.array([r["plane_agree"] for r in res])
    nsl = np.array([r["nslice"] for r in res])

    # ---- laterality: geometry first, DICOM tag as a cross-check ----------------
    # +X is patient-left, so a knee centred at X>0 is a LEFT knee.
    xs = np.where(valid, lat_x, np.nan)
    with np.errstate(invalid="ignore"):
        xmean = np.nanmean(xs, axis=1)
    xmean = np.nan_to_num(xmean, nan=0.0)
    lat_geo = np.where(xmean > 0, 1, -1).astype(np.int8)     # +1 left, -1 right
    lat_geo[np.abs(xmean) < 1.0] = 0                          # too close to midline to call

    tags = []
    for r in res:
        t = [x for x in r["lat_tag"] if x]
        tags.append(t[0] if t else "")
    tag_arr = np.array(tags)
    lat_tag_num = np.zeros(N, np.int8)
    lat_tag_num[np.isin(tag_arr, ["L", "LEFT"])] = 1
    lat_tag_num[np.isin(tag_arr, ["R", "RIGHT"])] = -1

    has_tag = lat_tag_num != 0
    agree = int(np.sum(lat_tag_num[has_tag] == lat_geo[has_tag]))
    log(f"laterality: geometry says {int(np.sum(lat_geo == 1))} left, "
        f"{int(np.sum(lat_geo == -1))} right, {int(np.sum(lat_geo == 0))} unresolved")
    log(f"DICOM Laterality tag present on {int(has_tag.sum())}/{N} studies; "
        f"agrees with geometry on {agree}/{int(has_tag.sum())}"
        + (f" ({100.0 * agree / max(int(has_tag.sum()), 1):.1f}%)" if has_tag.sum() else ""))
    log(f"|centre X| percentiles: "
        + " ".join(f"p{p}={np.percentile(np.abs(xmean), p):.0f}mm" for p in (1, 5, 25, 50)))

    is_left = (lat_geo == 1)[:, None]

    # ---- fold laterality in ---------------------------------------------------
    # Sagittal: ascending toward patient-left runs lateral->medial on a RIGHT knee
    # and medial->lateral on a LEFT knee, so a left knee needs one more reversal.
    sag = np.zeros(NSLOT, bool)
    sag[[0, 1]] = True
    rev_depth = rev_raw ^ (is_left & sag[None, :])
    # Coronal/axial: the horizontal axis carries medial-lateral, same extra flip.
    corax = np.zeros(NSLOT, bool)
    corax[[2, 3, 4]] = True
    flip_h = fh_raw ^ (is_left & corax[None, :])

    # ---- the numbers that decide the next experiment --------------------------
    summ = {"n_studies": int(N),
            "left": int(np.sum(lat_geo == 1)), "right": int(np.sum(lat_geo == -1)),
            "lat_unresolved": int(np.sum(lat_geo == 0)),
            "lat_tag_present": int(has_tag.sum()), "lat_tag_agrees": agree,
            "slots": {}}
    log("")
    log("=" * 78)
    log("PER-SLOT ORIENTATION (fraction of valid studies needing each correction)")
    log(f"{'slot':<11}{'valid':>7}{'revdepth':>10}{'flip_h':>9}{'flip_v':>9}"
        f"{'transp':>8}{'plane!=':>9}{'cos<.95':>9}")
    for j, name in enumerate(SLOT_NAMES):
        v = valid[:, j]
        nv = int(v.sum())
        if nv == 0:
            continue
        row = dict(valid=nv,
                   rev_depth=float(rev_depth[v, j].mean()),
                   flip_h=float(flip_h[v, j].mean()),
                   flip_v=float(fv[v, j].mean()),
                   transpose=float(tp[v, j].mean()),
                   plane_mismatch=float((~pagree[v, j]).mean()),
                   oblique=float((pcos[v, j] < 0.95).mean()),
                   mean_slices=float(nsl[v, j].mean()))
        summ["slots"][name] = row
        log(f"{name:<11}{nv:>7}{row['rev_depth']:>10.3f}{row['flip_h']:>9.3f}"
            f"{row['flip_v']:>9.3f}{row['transpose']:>8.3f}"
            f"{row['plane_mismatch']:>9.3f}{row['oblique']:>9.3f}")
    log("=" * 78)
    log("A value near 0.5 means that axis is a coin flip across the corpus and the")
    log("model has to infer it from the image before it can name a compartment.")
    log("A value near 0.0 means the scanners already agreed and there is nothing to fix.")
    log("")

    # How many studies are affected at all -- the honest headline number.
    any_sag_rev = rev_depth[:, [0, 1]][valid[:, [0, 1]]].mean() if valid[:, [0, 1]].any() else 0.0
    touched = np.zeros(N, bool)
    for j in range(NSLOT):
        touched |= valid[:, j] & (rev_depth[:, j] | flip_h[:, j] | fv[:, j])
    summ["studies_touched"] = int(touched.sum())
    summ["studies_touched_frac"] = float(touched.mean())
    summ["sagittal_depth_flip_frac"] = float(any_sag_rev)
    log(f"HEADLINE: {int(touched.sum())}/{N} studies ({100 * touched.mean():.1f}%) have at "
        f"least one slot that canonicalisation would change.")
    log(f"          sagittal depth direction is reversed on "
        f"{100 * any_sag_rev:.1f}% of valid sagittal slots.")

    np.savez_compressed(
        WORK / "orient.npz",
        ids=np.array([r["sid"] for r in res]),
        slot_names=np.array(SLOT_NAMES),
        valid=valid, rev_depth=rev_depth, flip_h=flip_h, flip_v=fv, transpose=tp,
        laterality=lat_geo, lat_tag=lat_tag_num, centre_x=xmean.astype(np.float32),
        plane_cos=pcos, plane_agree=pagree, nslice=nsl)

    df = pd.DataFrame({"StudyInstanceUID": [r["sid"] for r in res],
                       "laterality": lat_geo, "lat_tag": lat_tag_num,
                       "centre_x": xmean.round(1)})
    for j, name in enumerate(SLOT_NAMES):
        df[f"{name}_valid"] = valid[:, j]
        df[f"{name}_rev_depth"] = rev_depth[:, j]
        df[f"{name}_flip_h"] = flip_h[:, j]
        df[f"{name}_flip_v"] = fv[:, j]
    df.to_csv(WORK / "orient.csv", index=False)
    json.dump(summ, open(WORK / "orient_summary.json", "w"), indent=2)
    log("wrote orient.npz, orient.csv, orient_summary.json")


if __name__ == "__main__":
    main()
