# -*- coding: utf-8 -*-
"""Inference / submission for the vol3d multi-plane model.

Builds each hidden-test study's three plane volumes with the SAME logic as build_volume_cache.py
(24x192x192, fluid-sensitive-preferred series, physical slice order via roi_geometry) -- the volumizer
is copied verbatim from the builder so train- and test-time volumes agree; the self-check at the end
enforces the submission contract. Loads vol3d_best.pt (weights=None arch + our state_dict, no
download) and writes submission.csv matching sample_submission.csv.

Attach: competition (hidden test DICOM), rsna-knee-roi-code (roi_geometry.py),
rsna-knee-vol3d-train kernel output (vol3d_best.pt).
"""
import glob
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PLANES = ["Axial", "Coronal", "Sagittal"]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
DEPTH, HW = 24, 192          # MUST match build_volume_cache defaults


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def import_geometry():
    for p in Path("/kaggle/input").rglob("roi_geometry.py"):
        spec = importlib.util.spec_from_file_location("roi_geometry", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    raise FileNotFoundError("roi_geometry.py not attached")


# ---- volumizer: copied verbatim from build_volume_cache.py ----
def read_pixels(path):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    d = pydicom.dcmread(str(path))
    return np.asarray(apply_modality_lut(d.pixel_array.astype(np.float32), d), dtype=np.float32)


def resample_volume(slices, depth=DEPTH, hw=HW):
    import cv2
    if not slices:
        return np.zeros((depth, hw, hw), np.uint8)
    resized = []
    for s in slices:
        interp = cv2.INTER_AREA if (s.shape[0] > hw or s.shape[1] > hw) else cv2.INTER_LINEAR
        resized.append(cv2.resize(s.astype(np.float32), (hw, hw), interpolation=interp))
    vol = np.stack(resized, 0)
    n = vol.shape[0]
    if n == 1:
        vol = np.repeat(vol, depth, 0)
    elif n != depth:
        src = np.linspace(0.0, n - 1.0, n); dst = np.linspace(0.0, n - 1.0, depth)
        lo = np.clip(np.searchsorted(src, dst, "right") - 1, 0, n - 2)
        frac = (dst - src[lo]) / np.maximum(src[lo + 1] - src[lo], 1e-6)
        vol = vol[lo] * (1 - frac)[:, None, None] + vol[lo + 1] * frac[:, None, None]
    lo_p, hi_p = np.percentile(vol, [1, 99])
    if hi_p <= lo_p:
        hi_p = lo_p + 1.0
    return (np.clip((vol - lo_p) / (hi_p - lo_p), 0, 1) * 255.0 + 0.5).astype(np.uint8)


def pick_series_for_plane(rows):
    rows = sorted(rows, key=lambda r: str(r["SeriesInstanceUID"]))
    fluid = [r for r in rows if int(r.get("Fluid_Sensitive", 0) or 0) == 1]
    return (fluid or rows)[0] if rows else None


def ordered_slice_pixels(geom, series_dir):
    import pydicom
    slices = []
    for f in Path(series_dir).glob("*.dcm"):
        try:
            slices.append(geom.SliceGeometry.from_header(
                f, pydicom.dcmread(str(f), stop_before_pixels=True)))
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


def build_model():
    from torchvision.models.video import r3d_18
    backbone = r3d_18(weights=None)
    feat = backbone.fc.in_features
    backbone.fc = nn.Identity()

    class VolNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, 256),
                                      nn.GELU(), nn.Dropout(0.3), nn.Linear(256, len(LAB)))

        def forward(self, vols, mask):
            b, p = vols.shape[0], vols.shape[1]
            x = vols.reshape(b * p, 1, *vols.shape[2:]).repeat(1, 3, 1, 1, 1)
            f = self.backbone(x).reshape(b, p, -1)
            m = mask.unsqueeze(-1).float()
            return self.head((f * m).sum(1) / m.sum(1).clamp(min=1.0))

    return VolNet()


def main():
    geom = import_geometry()
    root = next(p.parent for p in Path("/kaggle/input").rglob("test_series.csv"))
    comp = next(p.parent for p in Path("/kaggle/input").rglob("sample_submission.csv"))
    ckpt_path = next(Path("/kaggle/input").rglob("vol3d_best.pt"))
    log(f"root={root} comp={comp} ckpt={ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    log(f"loaded vol3d checkpoint: gold {sd.get('gold_macro_auc')}, epoch {sd.get('epoch')}")

    series = pd.read_csv(root / "test_series.csv")
    series["StudyInstanceUID"] = series.StudyInstanceUID.astype(str)
    series_root = root / "test_series"
    studies = sorted(series.StudyInstanceUID.unique())
    log(f"{len(studies)} test studies")

    preds = {}
    for i, uid in enumerate(studies):
        sdf = series[series.StudyInstanceUID == uid]
        vols = np.zeros((len(PLANES), DEPTH, HW, HW), np.float32)
        mask = np.zeros(len(PLANES), np.float32)
        for pj, pl in enumerate(PLANES):
            chosen = pick_series_for_plane(sdf[sdf.Anatomical_Plane == pl].to_dict("records"))
            if chosen is None:
                continue
            pix = ordered_slice_pixels(geom, series_root / uid / str(chosen["SeriesInstanceUID"]))
            if pix:
                vols[pj] = resample_volume(pix).astype(np.float32) / 255.0
                mask[pj] = 1
        with torch.no_grad():
            v = torch.from_numpy(vols).unsqueeze(0).to(device)
            m = torch.from_numpy(mask).unsqueeze(0).to(device)
            preds[uid] = torch.sigmoid(model(v, m))[0].float().cpu().numpy()
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(studies)}")

    sample = pd.read_csv(comp / "sample_submission.csv")
    sample["StudyInstanceUID"] = sample.StudyInstanceUID.astype(str)
    rows = [[u] + [float(preds.get(u, np.full(len(LAB), 0.5))[j]) for j in range(len(LAB))]
            for u in sample.StudyInstanceUID]
    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LAB)
    if list(sub.columns) != list(sample.columns) or not np.isfinite(sub[LAB].to_numpy()).all():
        raise RuntimeError("submission does not match sample_submission contract")
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    log(f"wrote submission.csv {sub.shape}")
    print("VOL3D_INFER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
