"""roi_verify: can the SUBMISSION environment rebuild the Arm C ROI cache byte-for-byte?

WHY THIS RUNS BEFORE ANY ARM C TRAINING OR SUBMISSION.
Arm C ROI (codex/remote-roi-annotator) trains on five precomputed ROI caches -- MCL,
lateral meniscus, lateral OA, PF OA, synovitis -- built by kaggle/roi_cache/
build_roi_cache.py. The hidden test has no cache: the submission kernel must rebuild
every test study's ROIs from raw DICOM. A model only works at test time if that rebuild
reproduces the pixels it trained on. Getting this wrong does not crash -- it feeds the
model subtly different images and silently costs score. That is exactly what
arm10_verify guarded against for the global cache64 recipe, and this is the same check
for the ROI path.

There is also a specific reason to doubt it here. The ROI arrays in the dataset were
uploaded 2026-09-21 ~19:56. The builder's last commit (ffcffe7, "roi-annotator") is
2026-09-22 09:41, and it changes how the crop ORIGIN is computed (from the selected
series only to every declared series), which moves crop centres and therefore pixels.
The cache metadata already lists features that exist only after that commit
(paired_side_centered, nearest_distinct_triplets), so the likeliest story is that the
cache was built with this code before it was committed. Reading cannot settle that;
a byte comparison can.

WHAT THIS DOES (CPU only, no GPU quota, no internet):
  1. Imports build_study from zeroday19/rsna-knee-roi-code, which holds the exact
     branch-tip build_roi_cache.py + roi_geometry.py (sha256 recorded below). Importing
     rather than reimplementing makes byte-exactness a property of the code, not of
     careful copying.
  2. Rebuilds N_TRAIN seeded-random TRAIN studies exactly as command_build does -- same
     priors JSON, same `groupby(sort=False)` row grouping (pick_series is first-match, so
     within-study row order matters) -- and requires every target's (4, 3, 336, 336)
     uint8 array AND its (4,) mask to be byte-identical to the uploaded cache.
  3. Runs build_study on every visible TEST study from test_series, proving the test
     path executes end to end, and times it -- that per-study cost is what an Arm C
     member adds to the submission runtime.

PASS means a submission kernel that imports this code will feed a trained Arm C model
the same ROIs it trained on. FAIL means either the committed code no longer matches the
cache (rebuild the cache) or the environment differs (pin versions) -- and either way,
do not train or submit on this cache until it is resolved.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True          # the code dataset is read-only; don't try to cache .pyc

N_TRAIN = int(os.environ.get("ROI_VERIFY_N", "8"))
SEED = 7
EXPECTED_SHA = {
    "build_roi_cache.py": "50973aae81ea0e98ebee122866ea85183fbd64af72b61af2d80b5cf6c7af499d",
    "roi_geometry.py": "c935d8055f948b9785a6e5f8db7d12dccea6d3e2ba42d23a652a689d6afd5ebb",
}
EXPECTED_PRIORS_SHA = "1936df4c83bc9c630edf912833e740d0de71d1870c2ee2e77818a1f4c346ac75"
SKIP_DIRS = {"train_series", "test_series"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def walk_find(name, root="/kaggle/input"):
    """Every path under the inputs with this basename, never descending into DICOM trees."""
    out = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        if name in fs:
            out.append(Path(d) / name)
    return out


def main():
    t0 = time.time()

    # ---- 1. the exact builder code ----------------------------------------------
    hits = [p for p in walk_find("build_roi_cache.py") if (p.parent / "roi_geometry.py").exists()]
    if not hits:
        raise FileNotFoundError("build_roi_cache.py + roi_geometry.py not attached")
    code_dir = hits[0].parent
    for fname, want in EXPECTED_SHA.items():
        got = sha256(code_dir / fname)
        if got != want:
            raise RuntimeError(f"{fname} sha256 {got[:12]} != expected {want[:12]} -- wrong code")
    log(f"builder code at {code_dir} (sha256 verified: branch-tip ffcffe7)")
    sys.path.insert(0, str(code_dir))
    import build_roi_cache as B                                   # noqa: E402
    from roi_geometry import TARGET_SPECS                          # noqa: E402
    targets = list(TARGET_SPECS)
    log(f"targets: {targets}")

    # ---- 2. the uploaded ROI cache -------------------------------------------------
    cache_dirs = [p.parent for p in walk_find("roi_ids.npy")
                  if all((p.parent / f"roi_{t}.npy").exists() for t in targets)]
    if not cache_dirs:
        raise FileNotFoundError("no directory holds roi_ids.npy together with all five roi_<target>.npy")
    cache_dir = cache_dirs[0]
    if len(cache_dirs) > 1:
        log(f"{len(cache_dirs)} complete ROI caches attached; using {cache_dir}")
    ids = np.load(cache_dir / "roi_ids.npy", allow_pickle=True).astype(str)
    arrays = {t: np.load(cache_dir / f"roi_{t}.npy", mmap_mode="r") for t in targets}
    masks = {t: np.load(cache_dir / f"roi_{t}_masks.npy", mmap_mode="r") for t in targets}
    for t in targets:
        if arrays[t].shape[0] != len(ids) or masks[t].shape[0] != len(ids):
            raise RuntimeError(f"{t}: array rows {arrays[t].shape[0]} / mask rows "
                               f"{masks[t].shape[0]} != {len(ids)} ids")
    log(f"ROI cache at {cache_dir}: {len(ids)} studies, per-target shape {arrays[targets[0]].shape[1:]}")

    priors_path = cache_dir / "roi_priors.json"
    if not priors_path.exists():
        cands = [p for p in walk_find("roi_priors.json") if sha256(p) == EXPECTED_PRIORS_SHA]
        if not cands:
            raise FileNotFoundError("roi_priors.json matching the cache's priors_sha256 not found")
        priors_path = cands[0]
    psha = sha256(priors_path)
    if psha != EXPECTED_PRIORS_SHA:
        raise RuntimeError(f"priors sha256 {psha[:12]} != the cache's recorded {EXPECTED_PRIORS_SHA[:12]}")
    priors = json.loads(priors_path.read_text())
    log(f"priors {priors_path} sha256 matches the cache metadata; phase={priors.get('phase')} "
        f"accepted={priors.get('accepted')}")

    # ---- 3. competition data, grouped exactly as command_build does -------------
    comp = B.find_competition_root(None)

    def grouped(csv_name):
        frame = pd.read_csv(comp / csv_name)
        frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
        frame["SeriesInstanceUID"] = frame.SeriesInstanceUID.astype(str)
        return {k: g.to_dict("records") for k, g in frame.groupby("StudyInstanceUID", sort=False)}

    train_by = grouped("train_series.csv")
    log(f"competition root {comp}")

    # ---- 4. TRAIN: byte-exact rebuild ---------------------------------------------
    rng = np.random.default_rng(SEED)
    rows = sorted(rng.choice(len(ids), min(N_TRAIN, len(ids)), replace=False).tolist())
    log("")
    log(f"=== TRAIN byte-exact rebuild of {len(rows)} seeded-random studies ===")
    mismatches, per_study_s = [], []
    for i in rows:
        sid = ids[i]
        ts = time.time()
        data, m, _prov = B.build_study(sid, train_by.get(sid, []), comp / "train_series", priors)
        per_study_s.append(time.time() - ts)
        verdicts = []
        for t in targets:
            ref = np.asarray(arrays[t][i])
            px_ok = np.array_equal(data[t], ref)
            mk_ok = np.array_equal(m[t], np.asarray(masks[t][i]))
            if px_ok and mk_ok:
                verdicts.append(f"{t}=OK")
                continue
            diff = np.abs(data[t].astype(np.int16) - ref.astype(np.int16))
            frac = float((diff > 0).mean())
            mismatches.append((sid, t, px_ok, mk_ok, frac, int(diff.max())))
            verdicts.append(f"{t}=DIFF(px {100 * frac:.2f}% max {int(diff.max())}, "
                            f"mask {'ok' if mk_ok else 'DIFF'})")
        log(f"  row {i:5d}  {per_study_s[-1]:5.1f}s  " + "  ".join(verdicts))

    # ---- 5. TEST: the path a submission kernel actually runs ----------------------
    log("")
    log("=== TEST path: build_study on every visible test study ===")
    test_ids = pd.read_csv(comp / "test.csv").StudyInstanceUID.astype(str).tolist()
    test_by = grouped("test_series.csv")
    test_s, test_valid = [], []
    for sid in test_ids:
        ts = time.time()
        data, m, prov = B.build_study(sid, test_by.get(sid, []), comp / "test_series", priors)
        test_s.append(time.time() - ts)
        valid = {t: int(np.asarray(m[t]).sum()) for t in targets}
        test_valid.append(valid)
        # The provenance schema's field is `failure_reason` (see PROVENANCE_FIELDS). An
        # earlier draft read a non-existent "status" key, which would have printed an
        # empty list -- i.e. "no problems" -- regardless of what actually happened.
        reasons = sorted({str(r.get("failure_reason", "")) for r in prov
                          if str(r.get("failure_reason", "")).strip()})
        log(f"  {sid[-12:]}  {test_s[-1]:5.1f}s  valid windows " +
            " ".join(f"{t}={v}/4" for t, v in valid.items()) +
            (f"  | failure reasons: {reasons[:4]}" if reasons else "  | no failure reasons"))

    # ---- 6. verdict ---------------------------------------------------------------
    log("")
    log("=" * 78)
    n_checked = len(rows) * len(targets)
    log(f"TRAIN: {n_checked - len(mismatches)}/{n_checked} (study, target) pairs byte-identical "
        f"to the uploaded cache")
    log(f"build_study cost: train median {np.median(per_study_s):.1f}s/study, "
        f"test median {np.median(test_s):.1f}s/study (single process)")
    json.dump({"n_train_checked": len(rows), "mismatches": mismatches,
               "train_s_per_study": per_study_s, "test_s_per_study": test_s,
               "test_valid_windows": test_valid, "cache_dir": str(cache_dir),
               "code_sha": EXPECTED_SHA, "priors_sha": psha,
               "minutes": (time.time() - t0) / 60},
              open("/kaggle/working/roi_verify.json", "w"), indent=1)
    if mismatches:
        log("VERDICT: FAIL -- the committed builder does NOT reproduce the uploaded ROI cache.")
        log("  Do not train Arm C on this cache or submit it until resolved: either rebuild")
        log("  the cache from ffcffe7, or find the code that produced it.")
        raise SystemExit(1)
    log("VERDICT: PASS -- a submission kernel importing this exact code reproduces the")
    log("  training ROI cache byte-for-byte, and the test_series path runs end to end.")


if __name__ == "__main__":
    main()
