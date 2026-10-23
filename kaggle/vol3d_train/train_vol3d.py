# -*- coding: utf-8 -*-
"""Train a 3D multi-plane volume model as a decorrelated ensemble member (Stage: vol3d).

CONTEXT. GATE 1 (A5) showed the host labels need image information the reports lack, so the remaining
headroom is pixel-side. The current stack is 2D/2.5D (cache64 + ROI triplets); this adds the genuinely
different axis -- full-study 3D volumes, one per plane -- as an ENSEMBLE ADDITION, not a replacement.
EV is tempered (8 pixel-arms are dead; knee_dino's SSL caught protocol not pathology), so this is an
experiment judged by CV + a deliberate LB check, never trained-and-submitted blind.

MODEL. One shared 3D backbone (torchvision r3d_18, Kinetics-pretrained) encodes each plane's volume;
the three plane features are masked-mean fused (a missing plane contributes exactly zero) and a small
head emits 12 logits. Trained on train_pool (soft C) with BCE; evaluated on the 58 gold studies by
macro AUC; a held-out split yields OOF predictions for blending.

SELF-TEST. `VOL_SELFTEST=1` builds the model and runs a forward+backward on synthetic tensors on CPU
(weights=None, no download) to validate shapes / masking / finiteness, then exits. This is the smoke
that runs on a Kaggle CPU kernel -- no volume cache, no GPU, no quota.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PLANES = ["axial", "coronal", "sagittal"]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEED = 20260928
SELFTEST = int(os.environ.get("VOL_SELFTEST", "0"))
# Full-power defaults (priority-1 single-model run). Override via env for probes.
EPOCHS = int(os.environ.get("VOL_EPOCHS", "22"))
BATCH = int(os.environ.get("VOL_BATCH", "10"))
LR = float(os.environ.get("VOL_LR", "3e-4"))
WARMUP_FRAC = float(os.environ.get("VOL_WARMUP_FRAC", "0.1"))
WEIGHT_DECAY = float(os.environ.get("VOL_WD", "0.05"))
GRAD_CLIP = float(os.environ.get("VOL_CLIP", "3.0"))
AUGMENT = int(os.environ.get("VOL_AUGMENT", "1"))
MAX_STUDIES = int(os.environ.get("VOL_MAX_STUDIES", "0"))   # >0 = cap pool studies (GPU smoke)
VAL_FRACTION = float(os.environ.get("VOL_VAL_FRACTION", "0.15"))
WORK = Path(os.environ.get("VOL_WORK", "/kaggle/working"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def augment_volume(v, rng):
    """MRI-safe per-volume augmentation (NO flips, per roi/knee_dino policy).
    v [3planes, D, H, W] float in [0,1] -> augmented copy (gamma / brightness / noise)."""
    out = v.copy()
    for p in range(out.shape[0]):
        if rng.random() < 0.8:
            out[p] = np.power(np.clip(out[p], 0, 1), float(rng.uniform(0.75, 1.35)))
        if rng.random() < 0.8:
            out[p] = out[p] * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.06, 0.06))
        if rng.random() < 0.4:
            out[p] = out[p] + rng.normal(0, float(rng.uniform(0, 0.03)), out[p].shape).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


def build_model(pretrained):
    import torch.nn as nn
    from torchvision.models.video import r3d_18
    weights = "KINETICS400_V1" if pretrained else None
    backbone = r3d_18(weights=weights)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    class VolNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 256),
                                      nn.GELU(), nn.Dropout(0.3), nn.Linear(256, len(LAB)))

        def forward(self, vols, mask):
            # vols [B, 3planes, D, H, W] (1 channel each); mask [B, 3]
            b, p = vols.shape[0], vols.shape[1]
            x = vols.reshape(b * p, 1, *vols.shape[2:]).repeat(1, 3, 1, 1, 1)  # 1ch -> 3ch for r3d_18
            feat = self.backbone(x).reshape(b, p, -1)                          # [B, 3, F]
            m = mask.unsqueeze(-1).float()
            fused = (feat * m).sum(1) / m.sum(1).clamp(min=1.0)                # masked mean over planes
            return self.head(fused)

    return VolNet()


def selftest():
    import torch
    log("SELFTEST: building model (weights=None) on CPU")
    model = build_model(pretrained=False)
    model.train()
    b, d, h, w = 2, 8, 32, 32
    vols = torch.randn(b, 3, d, h, w)
    mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
    y = torch.randint(0, 2, (b, len(LAB))).float()
    out = model(vols, mask)
    assert out.shape == (b, len(LAB)), f"bad output shape {out.shape}"
    assert torch.isfinite(out).all(), "non-finite logits"
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters())
    # a fully-missing-plane row must not produce NaN
    out2 = model(vols, torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32))
    assert torch.isfinite(out2).all(), "NaN when a study has no planes"
    n_params = sum(p.numel() for p in model.parameters())
    log(f"SELFTEST PASS: out {tuple(out.shape)}, loss {loss.item():.4f}, "
        f"grad_ok {grad_ok}, params {n_params/1e6:.1f}M, zero-plane row finite")
    WORK.mkdir(parents=True, exist_ok=True)
    json.dump({"selftest": "pass", "out_shape": list(out.shape), "loss": float(loss),
               "params_m": n_params / 1e6}, open(WORK / "vol3d_selftest.json", "w"), indent=1)
    print("VOL3D_SELFTEST_COMPLETE", flush=True)


# --- real training (GPU; runs Saturday). Kept compact; validated piecemeal by selftest + a cache smoke.
def load_cache():
    import glob
    base = None
    for root in ("/kaggle/input", "."):
        hits = glob.glob(f"{root}/**/vol_ids_shard*.npy", recursive=True)
        if hits:
            base = hits
            break
    if not base:
        raise FileNotFoundError("volume cache (vol_ids_shard*.npy) not attached")
    ids, masks, planes = [], [], {p: [] for p in PLANES}
    for idp in sorted(set(base)):
        d = Path(idp).parent
        s = idp.split("vol_ids")[-1].replace(".npy", "")
        ids.append(np.load(idp, allow_pickle=True))
        masks.append(np.load(d / f"vol_masks{s}.npy"))
        for p in PLANES:
            planes[p].append(np.load(d / f"vol_{p}{s}.npy", mmap_mode="r"))
    ids = np.concatenate(ids).astype(str)
    masks = np.concatenate(masks, 0)
    vols = {p: np.concatenate(planes[p], 0) for p in PLANES}
    return ids, masks, vols


def main():
    if SELFTEST:
        selftest()
        return
    import torch
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    import glob

    torch.manual_seed(SEED); np.random.seed(SEED)
    ids, masks, vols = load_cache()
    id_pos = {u: i for i, u in enumerate(ids)}
    log(f"cache: {len(ids)} studies, plane coverage {masks.mean(0).round(3).tolist()}")

    def find(name):
        h = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        if not h:
            raise FileNotFoundError(name)
        return h[0]
    pool = pd.read_csv(find("train_pool.csv")); pool["StudyInstanceUID"] = pool.StudyInstanceUID.astype(str)
    gold = pd.read_csv(find("gold_test.csv")); gold["StudyInstanceUID"] = gold.StudyInstanceUID.astype(str)

    class DS(Dataset):
        def __init__(self, df, soft, train=False):
            self.rows = [r for r in df.to_dict("records") if str(r["StudyInstanceUID"]) in id_pos]
            self.soft = soft
            self.train = bool(train and AUGMENT)
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]; k = id_pos[str(r["StudyInstanceUID"])]
            v = np.stack([np.asarray(vols[p][k], np.float32) / 255.0 for p in PLANES], 0)
            if self.train:
                v = augment_volume(v, np.random.default_rng(np.random.randint(0, 2 ** 32 - 1)))
            y = np.array([float(r[c]) for c in LAB], np.float32)
            if not self.soft:
                y = (y >= 0.5).astype(np.float32)
            return torch.from_numpy(v), torch.from_numpy(masks[k].astype(np.float32)), torch.from_numpy(y)

    rng = np.random.default_rng(SEED)
    pool_rows = [r for r in pool.to_dict("records") if str(r["StudyInstanceUID"]) in id_pos]
    if MAX_STUDIES:
        pool_rows = pool_rows[:MAX_STUDIES]                    # smoke / pre-flight cap
    use_ids = {str(r["StudyInstanceUID"]) for r in pool_rows}
    pool_use = pool[pool.StudyInstanceUID.isin(use_ids)]
    idx = rng.permutation(len(pool_rows)); n_val = max(1, int(len(idx) * VAL_FRACTION))
    val_ids = {str(pool_rows[i]["StudyInstanceUID"]) for i in idx[:n_val]}
    tr_df = pool_use[~pool_use.StudyInstanceUID.isin(val_ids)]
    oof_df = pool_use[pool_use.StudyInstanceUID.isin(val_ids)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=True).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    tl = DataLoader(DS(tr_df, True, train=True), batch_size=BATCH, shuffle=True,
                    num_workers=2, drop_last=True)
    gl = DataLoader(DS(gold, False), batch_size=BATCH, num_workers=2)
    ol = DataLoader(DS(oof_df, False), batch_size=BATCH, num_workers=2)

    def auc_over(loader):
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for v, m, y in loader:
                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    p = torch.sigmoid(model(v.to(device), m.to(device)))
                P.append(p.float().cpu().numpy()); Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(LAB))
                if 0 < Y[:, j].sum() < len(Y)]
        return float(np.mean(aucs)), P, Y

    import math
    total_steps = EPOCHS * max(1, len(tl)); warmup = max(1, int(WARMUP_FRAC * total_steps)); step = 0
    best = -1; hist = []
    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        for v, m, y in tl:
            lr = (LR * (step + 1) / warmup if step < warmup else
                  LR * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(v.to(device), m.to(device)), y.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); step += 1
        gold_auc, goldP, _ = auc_over(gl)
        hist.append({"epoch": ep + 1, "gold_macro_auc": gold_auc, "min": (time.time() - t0) / 60})
        log(json.dumps(hist[-1]))
        if gold_auc > best:
            best = gold_auc
            _, oofP, _ = auc_over(ol)
            np.save(WORK / "vol3d_oof_pred.npy", oofP)
            pd.Series([str(r["StudyInstanceUID"]) for r in DS(oof_df, False).rows]).to_csv(
                WORK / "vol3d_oof_ids.csv", index=False)
            # gold-58 predictions (for the blend test) + checkpoint (for test-time inference / submission)
            np.save(WORK / "vol3d_gold_pred.npy", goldP)
            pd.Series([str(r["StudyInstanceUID"]) for r in DS(gold, False).rows]).to_csv(
                WORK / "vol3d_gold_ids.csv", index=False)
            core = model.module if hasattr(model, "module") else model
            torch.save({"state_dict": core.state_dict(), "arch": "r3d_18_multiplane",
                        "planes": PLANES, "depth": None, "gold_macro_auc": best, "epoch": ep + 1},
                       WORK / "vol3d_best.pt")
    json.dump({"best_gold_macro_auc": best, "history": hist, "n_train": len(tr_df),
               "n_oof": len(oof_df)}, open(WORK / "vol3d_summary.json", "w"), indent=1)
    log(f"done best gold macro AUC {best:.4f}")
    print("VOL3D_TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
