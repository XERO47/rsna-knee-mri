"""roi_build: rebuild the Arm C ROI cache ON KAGGLE, with the committed builder.

WHY. roi_verify (2026-09-22) rebuilt 8 train studies with the committed builder
(ffcffe7) inside a Kaggle kernel and compared them with the uploaded ROI cache:

    TRAIN: 0/40 (study, target) pairs byte-identical
    masks: all match   pixels: 8-22% differ, almost all by 1-3 grey levels;
                               two MCL crops by up to 255 (a crop-edge shift)

Matching masks mean the same series, the same valid windows and the same crop geometry.
Pixel differences of a grey level or two over ~15% of the image are the signature of
intensity rounding, and the uploaded cache was built LOCALLY on another machine -- a
different cv2/numpy build commonly resamples a pixel or two differently. Training on
that cache is fine; DEPLOYING on it is not, because at submission time Kaggle rebuilds
the test crops in its own environment, so the model would train on one rendering and be
scored on another.

WHAT THIS DOES. Runs the teammate's own entrypoint, unmodified:
    build_roi_cache.py build --shard S --priors ... --ids-path ... --output-dir ...
in the Kaggle environment. The builder then self-checks three studies byte-for-byte
before finishing. Training on this output and rebuilding the hidden test in the same
environment with the same code makes train-time and test-time crops byte-exact by
construction. CPU only -- costs no GPU quota.

INPUTS (sha256-checked, so a different copy cannot be picked up silently):
    zeroday19/rsna-knee-roi-code   build_roi_cache.py + roi_geometry.py (branch ffcffe7)
    manyakaistha/knee-roi-data     roi_priors.json (the exact priors the cache recorded)
    zeroday19/rsna-knee-cache64-sS ids_shardS.npy (authoritative cache64 row order)
"""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SHARD = 0
WORKERS = 4
EXPECTED_SHA = {
    "build_roi_cache.py": "50973aae81ea0e98ebee122866ea85183fbd64af72b61af2d80b5cf6c7af499d",
    "roi_geometry.py": "c935d8055f948b9785a6e5f8db7d12dccea6d3e2ba42d23a652a689d6afd5ebb",
}
EXPECTED_PRIORS_SHA = "1936df4c83bc9c630edf912833e740d0de71d1870c2ee2e77818a1f4c346ac75"
SKIP = {"train_series", "test_series"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def walk_find(name, root="/kaggle/input"):
    out = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        if name in fs:
            out.append(Path(d) / name)
    return out


def main():
    code = [p.parent for p in walk_find("build_roi_cache.py") if (p.parent / "roi_geometry.py").exists()]
    if not code:
        raise FileNotFoundError("builder code not attached")
    code_dir = code[0]
    for f, want in EXPECTED_SHA.items():
        if sha256(code_dir / f) != want:
            raise RuntimeError(f"{f}: sha256 mismatch -- not the ffcffe7 builder")

    priors = [p for p in walk_find("roi_priors.json") if sha256(p) == EXPECTED_PRIORS_SHA]
    if not priors:
        raise FileNotFoundError("roi_priors.json with the cache's recorded sha256 not found")

    # The teammate's dataset also carries copies of cache64's id files, so several may be
    # attached. Require every copy to be IDENTICAL rather than exactly one: that is both
    # robust and a free check that their cache64 copy has the same row order as ours.
    ids = walk_find(f"ids_shard{SHARD}.npy")
    if not ids:
        raise FileNotFoundError(f"ids_shard{SHARD}.npy not attached")
    ref = np.load(ids[0], allow_pickle=True).astype(str)
    for other in ids[1:]:
        if not np.array_equal(ref, np.load(other, allow_pickle=True).astype(str)):
            raise RuntimeError(f"{other} disagrees with {ids[0]} -- cache64 row order differs")
    log(f"{len(ids)} copies of ids_shard{SHARD}.npy attached, all identical")
    n = len(ref)
    log(f"shard {SHARD}: {n} studies | code {code_dir} | priors {priors[0]} | ids {ids[0]}")

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")     # code dir is read-only
    cmd = [sys.executable, str(code_dir / "build_roi_cache.py"), "build",
           "--shard", str(SHARD), "--priors", str(priors[0]), "--ids-path", str(ids[0]),
           "--output-dir", "/kaggle/working", "--workers", str(WORKERS)]
    log("running: " + " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, env=env)
    log("builder finished, including its own 3-study byte-exact determinism check")


if __name__ == "__main__":
    main()
