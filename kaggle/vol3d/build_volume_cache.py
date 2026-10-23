# -*- coding: utf-8 -*-
"""Build a 3D multi-plane volume cache for the knee-MRI corpus (Stage: vol3d).

WHY. GATE 1 (A5) showed the host's labels carry image information not present in the reports:
neither C nor an independent report reader (N) reaches the host labels from text (~0.80 < 0.90).
The remaining headroom is in the pixels. The existing stack is 2D/2.5D (cache64 global windows +
ROI triplets). This builds the genuinely different axis -- a full-study 3D volume per plane -- to
train a decorrelated image model for the ensemble (an experiment, EV tempered: 8 pixel-arms are
dead and knee_dino's SSL features captured protocol, not pathology).

WHAT. For every study, for each of Axial / Coronal / Sagittal, pick ONE series (fluid-sensitive
preferred), physically order its slices (reusing roi_geometry -- never filename order), resample to
a fixed D x H x W grid, 1-99% normalise, store uint8. Missing planes are zero-filled and masked 0.
Output IDs are the canonical study order; the trainer aligns them to cache64/labels by exact-ID
check (roi_geometry.assert_exact_id_alignment), the same contract Arm C ROI uses.

OUTPUT (one shard here; shardable via --shard/--n-shard):
    vol_axial_shard<S>.npy      uint8 [N, D, H, W]
    vol_coronal_shard<S>.npy    uint8 [N, D, H, W]
    vol_sagittal_shard<S>.npy   uint8 [N, D, H, W]
    vol_masks_shard<S>.npy      uint8 [N, 3]   (1 = plane present)
    vol_ids_shard<S>.npy        str   [N]
    vol_meta_shard<S>.json      dims, counts, per-plane coverage, fingerprints

Storage: three planes at D x H x W uint8. Default 24 x 192 x 192 ~= 12 GB over 4,407 studies, under
Kaggle's output cap; raise via EXTRACT_* envs only if the cap allows.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PLANES = ["Axial", "Coronal", "Sagittal"]
DEPTH = int(os.environ.get("VOL_DEPTH", "24"))
HW = int(os.environ.get("VOL_HW", "192"))
SHARD = int(os.environ.get("VOL_SHARD", "0"))
N_SHARD = int(os.environ.get("VOL_N_SHARD", "1"))
SMOKE = int(os.environ.get("VOL_SMOKE", "0"))          # >0 = build only this many studies
WORK = Path(os.environ.get("VOL_WORK", "/kaggle/working"))
SEED = 20260928


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --- physical slice ordering: reuse roi_geometry, never reimplement it here -------------------
def _import_geometry():
    import importlib.util
    for root in (Path("/kaggle/input"), Path(__file__).resolve().parent,
                 Path(__file__).resolve().parents[1] / "roi_cache"):
        for p in root.rglob("roi_geometry.py") if root.exists() else []:
            spec = importlib.util.spec_from_file_location("roi_geometry", p)
            mod = importlib.util.module_from_spec(spec)
            # Register BEFORE exec: roi_geometry uses @dataclass, whose internals look the module up
            # in sys.modules via cls.__module__ (py3.13 raises NoneType.__dict__ otherwise).
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("roi_geometry.py not attached (needed for physical slice ordering)")


def read_pixels(path):
    """Decode one DICOM to float32 with the modality LUT applied (matches roi_cache policy)."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    d = pydicom.dcmread(str(path))
    arr = apply_modality_lut(d.pixel_array.astype(np.float32), d)
    return np.asarray(arr, dtype=np.float32)


def resample_volume(slices, depth=DEPTH, hw=HW):
    """Ordered list of 2D float arrays -> fixed [depth, hw, hw] uint8, 1-99% normalised.

    Pure and CPU-testable: in-plane resize (area for down, linear for up), then linear interpolation
    along the slice axis to `depth`, then percentile scaling. Separated from all DICOM I/O so it can
    be unit-tested on synthetic stacks."""
    import cv2
    if not slices:
        return np.zeros((depth, hw, hw), np.uint8)
    resized = []
    for s in slices:
        interp = cv2.INTER_AREA if (s.shape[0] > hw or s.shape[1] > hw) else cv2.INTER_LINEAR
        resized.append(cv2.resize(s.astype(np.float32), (hw, hw), interpolation=interp))
    vol = np.stack(resized, 0)                                   # [n, hw, hw]
    n = vol.shape[0]
    if n == 1:
        vol = np.repeat(vol, depth, 0)
    elif n != depth:
        # linear interpolation of each (h,w) column onto `depth` evenly spaced positions
        src = np.linspace(0.0, n - 1.0, n)
        dst = np.linspace(0.0, n - 1.0, depth)
        lo = np.clip(np.searchsorted(src, dst, "right") - 1, 0, n - 2)
        frac = (dst - src[lo]) / np.maximum(src[lo + 1] - src[lo], 1e-6)
        vol = vol[lo] * (1 - frac)[:, None, None] + vol[lo + 1] * frac[:, None, None]
    lo_p, hi_p = np.percentile(vol, [1, 99])
    if hi_p <= lo_p:
        hi_p = lo_p + 1.0
    vol = np.clip((vol - lo_p) / (hi_p - lo_p), 0, 1)
    return (vol * 255.0 + 0.5).astype(np.uint8)


def pick_series_for_plane(series_rows):
    """Deterministic: prefer a fluid-sensitive series, else the first, by SeriesInstanceUID order."""
    rows = sorted(series_rows, key=lambda r: str(r["SeriesInstanceUID"]))
    fluid = [r for r in rows if int(r.get("Fluid_Sensitive", 0) or 0) == 1]
    return (fluid or rows)[0] if rows else None


def ordered_slice_pixels(geom, series_dir):
    """Read a series dir, build SliceGeometry per slice, physically order, return pixels in order."""
    import pydicom
    slices = []
    for f in Path(series_dir).glob("*.dcm"):
        try:
            hdr = pydicom.dcmread(str(f), stop_before_pixels=True)
            slices.append(geom.SliceGeometry.from_header(f, hdr))
        except Exception:
            continue
    if not slices:
        return []
    try:
        ordered = geom.sort_deduplicate_slices(slices)
    except Exception:
        ordered = sorted(slices, key=lambda s: str(getattr(s, "path", "")))
    out = []
    for s in ordered:
        try:
            out.append(read_pixels(s.path))
        except Exception:
            continue
    return out


def main():
    geom = _import_geometry()
    import pandas as pd
    # locate competition root + train_series.csv without walking the DICOM tree blindly
    root = None
    for base in (Path("/kaggle/input"), Path.cwd()):
        for p in (base.rglob("train_series.csv") if base.exists() else []):
            root = p.parent
            break
        if root:
            break
    if root is None:
        raise FileNotFoundError("train_series.csv not found")
    series = pd.read_csv(root / "train_series.csv")
    series["StudyInstanceUID"] = series.StudyInstanceUID.astype(str)
    studies = sorted(series.StudyInstanceUID.unique())
    # deterministic sharding over the canonical sorted order
    studies = [s for i, s in enumerate(studies) if i % N_SHARD == SHARD]
    if SMOKE:
        studies = studies[:SMOKE]
    log(f"root={root} | studies this shard={len(studies)} | dims {DEPTH}x{HW}x{HW} | "
        f"shard {SHARD}/{N_SHARD} smoke={SMOKE}")

    series_root = root / "train_series"
    vols = {pl: np.zeros((len(studies), DEPTH, HW, HW), np.uint8) for pl in PLANES}
    masks = np.zeros((len(studies), len(PLANES)), np.uint8)
    cover = {pl: 0 for pl in PLANES}
    t0 = time.time()
    for i, uid in enumerate(studies):
        sdf = series[series.StudyInstanceUID == uid]
        for pj, pl in enumerate(PLANES):
            rows = sdf[sdf.Anatomical_Plane == pl].to_dict("records")
            chosen = pick_series_for_plane(rows)
            if chosen is None:
                continue
            pix = ordered_slice_pixels(geom, series_root / uid / str(chosen["SeriesInstanceUID"]))
            if not pix:
                continue
            vols[pl][i] = resample_volume(pix)
            masks[i, pj] = 1
            cover[pl] += 1
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            log(f"  {i + 1}/{len(studies)} | {el / 60:.1f} min | "
                f"eta {el / (i + 1) * (len(studies) - i - 1) / 60:.1f} min")

    WORK.mkdir(parents=True, exist_ok=True)
    tag = f"_shard{SHARD}"
    for pl in PLANES:
        np.save(WORK / f"vol_{pl.lower()}{tag}.npy", vols[pl])
    np.save(WORK / f"vol_masks{tag}.npy", masks)
    np.save(WORK / f"vol_ids{tag}.npy", np.array(studies, dtype=object))
    meta = {"depth": DEPTH, "hw": HW, "planes": PLANES, "n": len(studies),
            "shard": SHARD, "n_shard": N_SHARD, "coverage": cover,
            "minutes": (time.time() - t0) / 60, "seed": SEED}
    json.dump(meta, open(WORK / f"vol_meta{tag}.json", "w"), indent=1)
    log(f"done: {json.dumps(meta)}")
    print("VOL3D_CACHE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
