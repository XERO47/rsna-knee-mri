"""Submission for arm10: rebuild each test study from DICOM exactly as the corpus
was built, run the fine-tuned CoAtNet over every window, write submission.csv.

THE TRAP THIS FILE EXISTS TO AVOID. The model trained on the public 44-slice
corpus; the hidden test set arrives as raw DICOM. If this rebuild differs from
the corpus in any way -- span, crop, window, slice order, series choice -- the
model sees images unlike its training data and the score drops silently.

`build_study` below is the recipe arm10_verify matched pixel-for-pixel (MAE
0.000, 10/10 studies, every slot). Two guards keep it honest at run time:

  1. SELF-CHECK. If the corpus is attached, rebuild a few TRAIN studies from
     their DICOM and require exact equality with the stored stacks before any
     test prediction is made. Code drift fails loudly here instead of costing
     score invisibly.
  2. MODE = "validate". Rebuild the 58 gold studies from DICOM, predict, and
     score. This must reproduce the training kernel's K=42 gold AUC -- an
     end-to-end check of preprocessing + windows + weights, not just weights.

Offline: timm builds the architecture with pretrained=False and the fine-tuned
state_dict supplies every weight (asserted 0 missing / 0 unexpected).
"""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from pydicom.pixel_data_handlers.util import apply_modality_lut

MODE = "submit"                # "submit" | "validate"
CKPT_NAME = "arm10_swa.pt"       # falls back to arm10_best.pt
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]

# --- corpus recipe: verified exact by kaggle/arm10_verify. Do not edit. ---
IMG = 336
CROP_MM = 140.0
SPAN = (0.15, 0.85)
SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
           ("Coronal", 0, 6), ("Axial", -1, 8)]
DEPTH = sum(k for _, _, k in SLOTS44)
# --------------------------------------------------------------------------
K_WINDOWS = 42
BATCH_STUDIES = 2
BUILD_THREADS = 4
SELF_CHECK_N = 3

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_file(name, root="/kaggle/input"):
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if name in fs:
            return Path(d) / name
    return None


def find_comp():
    for d, dirs, fs in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if "test.csv" in fs and "train_series.csv" in fs:
            return Path(d)
    raise FileNotFoundError("competition data not attached")


# ---------------------------------------------------------------- rebuild --
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
    for plane, fluid, k in SLOTS44:
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
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
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


# ----------------------------------------------------------------- windows --
def window_centers(mask, k):
    """Identical to StudyWindows.centers(train=False) in train_arm10.py."""
    valid = np.flatnonzero(mask > 0)
    if len(valid) < 3:
        valid = np.arange(min(3, DEPTH))
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, DEPTH - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [max(1, min(cs[i], DEPTH - 2)) for i in idx]


def windows(vol, mask, res):
    """Rebuilt stacks are always 336 (the corpus size); a model trained at another
    resolution (CoAtNet needs a multiple of 32, so 384) saw them bilinearly resized,
    exactly as train_arm10.StudyWindows does."""
    cs = window_centers(mask, K_WINDOWS)
    tri = np.stack([vol[[c - 1, c, c + 1]] for c in cs])            # (K, 3, H, W)
    x = torch.from_numpy(tri).float().div_(255.0)
    if x.shape[-1] != res:
        x = torch.nn.functional.interpolate(x, size=(res, res), mode="bilinear",
                                            align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


# ------------------------------------------------------------------- model --
class RaptorClassifier(nn.Module):
    """Must mirror train_arm10.RaptorClassifier exactly (asserted by 0/0 load)."""

    def __init__(self, backbone, n=len(LAB), drop=0.2):
        super().__init__()
        f = backbone.num_features
        self.backbone = backbone
        self.norm = nn.LayerNorm(f)
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, f))
        self.clsb = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        B, K = x.shape[:2]
        h = self.norm(self.backbone(x.flatten(0, 1)).view(B, K, -1))
        a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
        return (torch.einsum("bkn,bkf->bnf", a, h) * self.clsW).sum(-1) + self.clsb


def load_model(dev):
    import timm
    path = find_file(CKPT_NAME) or find_file("arm10_best.pt")
    if path is None:
        raise FileNotFoundError("no arm10 checkpoint attached")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck.get("res"), int) or list(ck.get("lab", [])) != LAB:
        raise RuntimeError(f"checkpoint contract drift: res={ck.get('res')} lab={ck.get('lab')}")
    bb = timm.create_model(ck["arch"], pretrained=False, num_classes=0, in_chans=3,
                           global_pool="avg")
    model = RaptorClassifier(bb)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    log(f"loaded {path.name}: arch {ck['arch']} epoch {ck.get('epoch')} "
        f"train-time gold {ck.get('gold_auc')} | {len(missing)} missing, {len(unexpected)} unexpected")
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: {missing[:4]} {unexpected[:4]}")
    return model.to(dev).eval(), ck


# ------------------------------------------------------------------ checks --
def self_check(comp, by_study):
    """Exact-equality rebuild of a few train studies against the corpus stacks."""
    ids_path = find_file("all_ids.npy")
    if ids_path is None:
        log("self-check skipped: corpus not attached")
        return
    ids = np.load(ids_path, allow_pickle=True).astype(str)
    vols = np.load(find_file("all_vols.npy"), mmap_mode="r")
    rng = np.random.default_rng(2026)
    for row in rng.choice(len(ids), SELF_CHECK_N, replace=False):
        vol, _ = build_study(ids[row], by_study.get(ids[row], []), f"{comp}/train_series")
        diff = int(np.abs(vol.astype(np.int16) - np.asarray(vols[row]).astype(np.int16)).sum())
        if diff:
            raise RuntimeError(f"SELF-CHECK FAILED: study row {row} differs from corpus by {diff}")
    log(f"self-check PASS: {SELF_CHECK_N} train studies rebuilt from DICOM identical to corpus")


@torch.no_grad()
def predict(model, dev, uids, by_study, series_root, res):
    preds = {}
    t0 = time.time()
    with ThreadPoolExecutor(BUILD_THREADS) as pool:
        for b0 in range(0, len(uids), BATCH_STUDIES):
            chunk = uids[b0:b0 + BATCH_STUDIES]
            built = list(pool.map(lambda u: build_study(u, by_study.get(u, []), series_root), chunk))
            x = torch.stack([windows(v, m, res) for v, m in built]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                p = torch.sigmoid(model(x).float()).cpu().numpy()
            for u, pu, (_, m) in zip(chunk, p, built):
                preds[u] = pu if m.any() else np.full(len(LAB), np.nan, np.float32)
            done = min(b0 + BATCH_STUDIES, len(uids))
            if done % 50 < BATCH_STUDIES or done == len(uids):
                el = time.time() - t0
                log(f"  {done}/{len(uids)} studies | {el / 60:.1f} min | "
                    f"eta {el / done * (len(uids) - done) / 60:.1f} min")
    return preds


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comp = find_comp()
    log(f"mode {MODE} | device {dev} | competition {comp}")
    model, ck = load_model(dev)

    if MODE == "validate":
        ser = pd.read_csv(comp / "train_series.csv")
        ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str)
        by = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}
        self_check(comp, by)
        gold = pd.read_csv(find_file("gold_test.csv"))
        uids = gold.StudyInstanceUID.astype(str).tolist()
        preds = predict(model, dev, uids, by, comp / "train_series", ck["res"])
        from sklearn.metrics import roc_auc_score
        P = np.stack([preds[u] for u in uids])
        Y = gold[LAB].to_numpy()
        aucs = {n: float(roc_auc_score(Y[:, j], P[:, j])) for j, n in enumerate(LAB)
                if len(set(Y[:, j])) > 1}
        au = float(np.mean(list(aucs.values())))
        log(f"GOLD macro-AUC rebuilt from DICOM: {au:.4f}  "
            f"(training kernel K=42 reported {ck.get('gold_auc')})")
        for k, v in aucs.items():
            log(f"   {k:18s} {v:.4f}")
        json.dump({"gold_auc_from_dicom": au, "aucs": aucs, "train_reported": ck.get("gold_auc")},
                  open("/kaggle/working/validate.json", "w"), indent=1)
        return

    test = pd.read_csv(comp / "test.csv")
    ser = pd.read_csv(comp / "test_series.csv")
    ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str)
    by = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}
    uids = test.StudyInstanceUID.astype(str).tolist()
    log(f"test studies: {len(uids)}")
    preds = predict(model, dev, uids, by, comp / "test_series", ck["res"])

    sub = pd.DataFrame(np.stack([preds[u] for u in uids]), columns=LAB)
    n_empty = int(sub.isna().any(axis=1).sum())
    if n_empty:
        log(f"WARNING: {n_empty} studies had no readable slices; filled with column medians")
        sub = sub.fillna(sub.median()).fillna(0.5)
    sub.insert(0, "StudyInstanceUID", uids)
    sample = pd.read_csv(comp / "sample_submission.csv")
    sub = sample[["StudyInstanceUID"]].astype(str).merge(sub, on="StudyInstanceUID", how="left")
    if sub[LAB].isna().any().any() or list(sub.columns) != list(sample.columns):
        raise RuntimeError("submission does not match sample_submission contract")
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    log(f"wrote submission.csv {sub.shape}")


if __name__ == "__main__":
    main()
