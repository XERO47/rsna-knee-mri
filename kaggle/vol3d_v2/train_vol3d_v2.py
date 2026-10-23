# -*- coding: utf-8 -*-
"""vol3d v2 -- X3D-M multi-plane 3D ensemble member (decorrelation-first).

Changes from v1 (r3d_18):
  * backbone = X3D-M (pytorchvideo, Kinetics-pretrained) -- ~3.8M params, a genuinely
    different volumetric family (decorrelation-per-compute, per advisor).
  * HONEST validation: epoch/checkpoint selected by a POOL-C held-out split, NOT by gold.
    The 58 gold stay fully held out and are predicted by the selected model -> UNBIASED
    gold preds for the gating harness (residual corr + incremental dAUC). v1 selected on
    gold, which inflated its 0.848.
  * GOLD_MODE: 'heldout' (default; gold never trained on -> the screen) |
               'folds'   (5-fold cross-fit on the 58, concat held-out preds) |
               'final'   (train on pool-C + ALL 58 gold, upweighted -> submission model).

Self-test (VOL_SELFTEST=1): build X3D model + synthetic fwd/bwd on CPU, exit.
"""
import json, os, sys, time, glob, math
from pathlib import Path
import numpy as np

PLANES = ["axial", "coronal", "sagittal"]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEED = int(os.environ.get("VOL_SEED", "20260928"))
SELFTEST = int(os.environ.get("VOL_SELFTEST", "0"))
EPOCHS = int(os.environ.get("VOL_EPOCHS", "18"))
BATCH = int(os.environ.get("VOL_BATCH", "6"))      # T4-safe default (override via env on bigger GPUs)
LR = float(os.environ.get("VOL_LR", "3e-4"))
WARMUP_FRAC = float(os.environ.get("VOL_WARMUP_FRAC", "0.1"))
WEIGHT_DECAY = float(os.environ.get("VOL_WD", "0.05"))
GRAD_CLIP = float(os.environ.get("VOL_CLIP", "3.0"))
AUGMENT = int(os.environ.get("VOL_AUGMENT", "1"))
MAX_STUDIES = int(os.environ.get("VOL_MAX_STUDIES", "0"))
VAL_FRACTION = float(os.environ.get("VOL_VAL_FRACTION", "0.15"))
WORKERS = int(os.environ.get("VOL_WORKERS", "4"))
GOLD_MODE = os.environ.get("VOL_GOLD_MODE", "heldout")   # heldout | folds | final
GOLD_FOLDS = int(os.environ.get("VOL_GOLD_FOLDS", "5"))
GOLD_UPWEIGHT = float(os.environ.get("VOL_GOLD_UPWEIGHT", "4.0"))
WORK = Path(os.environ.get("VOL_WORK", "/kaggle/working" if os.path.isdir("/kaggle/working") else "/workspace/out"))
DATA = os.environ.get("VOL_DATA", "/kaggle/input" if os.path.isdir("/kaggle/input") else "/workspace/data")


def _ensure_pytorchvideo():
    try:
        import pytorchvideo  # noqa: F401
    except Exception:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "iopath", "fvcore"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pytorchvideo"], check=False)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def augment_volume(v, rng):
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
    import torch, torch.nn as nn
    _ensure_pytorchvideo()
    from pytorchvideo.models.hub import x3d_m
    net = x3d_m(pretrained=pretrained)
    head = net.blocks[-1]
    head.pool = nn.AdaptiveAvgPool3d((1, 1, 1))   # X3D head pool has a FIXED (16,7,7) kernel; make it size-agnostic
    head.proj = nn.Identity()
    if hasattr(head, "activation"):
        head.activation = nn.Identity()
    net.eval()
    with torch.no_grad():
        feat_dim = net(torch.randn(1, 3, 8, 64, 64)).shape[-1]

    class VolNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = net
            self.head = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 256),
                                      nn.GELU(), nn.Dropout(0.3), nn.Linear(256, len(LAB)))

        def forward(self, vols, mask):
            b, p = vols.shape[0], vols.shape[1]
            x = vols.reshape(b * p, 1, *vols.shape[2:]).repeat(1, 3, 1, 1, 1)  # 1ch->3ch
            feat = self.backbone(x).reshape(b, p, -1)
            m = mask.unsqueeze(-1).float()
            fused = (feat * m).sum(1) / m.sum(1).clamp(min=1.0)
            return self.head(fused)

    return VolNet(), feat_dim


def selftest():
    import torch
    log("SELFTEST: X3D-M build + synthetic fwd/bwd (CPU)")
    model, fd = build_model(pretrained=False)
    model.train()
    b, d, h, w = 2, 16, 96, 96
    vols = torch.randn(b, 3, d, h, w)
    mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
    y = torch.randint(0, 2, (b, len(LAB))).float()
    out = model(vols, mask)
    assert out.shape == (b, len(LAB)), out.shape
    assert torch.isfinite(out).all()
    torch.nn.functional.binary_cross_entropy_with_logits(out, y).backward()
    out2 = model(vols, torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32))
    assert torch.isfinite(out2).all(), "NaN on zero-plane row"
    n = sum(p.numel() for p in model.parameters())
    log(f"SELFTEST PASS: feat_dim {fd}, params {n/1e6:.2f}M, out {tuple(out.shape)}")
    print("VOL3D_V2_SELFTEST_COMPLETE", flush=True)


def load_cache():
    hits = glob.glob(f"{DATA}/**/vol_ids_shard*.npy", recursive=True)
    if not hits:
        raise FileNotFoundError(f"vol_ids_shard*.npy not under {DATA}")
    ids, masks, planes = [], [], {p: [] for p in PLANES}
    for idp in sorted(set(hits)):
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


def find(name):
    h = glob.glob(f"{DATA}/**/{name}", recursive=True)
    if not h:
        raise FileNotFoundError(name)
    return h[0]


def main():
    if SELFTEST:
        selftest(); return
    import torch
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score
    import pandas as pd

    torch.manual_seed(SEED); np.random.seed(SEED)
    ids, masks, vols = load_cache()
    id_pos = {u: i for i, u in enumerate(ids)}
    log(f"cache: {len(ids)} studies, plane coverage {masks.mean(0).round(3).tolist()}")
    pool = pd.read_csv(find("train_pool.csv")); pool["StudyInstanceUID"] = pool.StudyInstanceUID.astype(str)
    gold = pd.read_csv(find("gold_test.csv")); gold["StudyInstanceUID"] = gold.StudyInstanceUID.astype(str)
    gold = gold[gold.StudyInstanceUID.isin(id_pos)].reset_index(drop=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class DS(Dataset):
        def __init__(self, rows, soft, train=False, weights=None):
            self.rows = [r for r in rows if str(r["StudyInstanceUID"]) in id_pos]
            self.soft = soft; self.train = bool(train and AUGMENT); self.weights = weights
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]; k = id_pos[str(r["StudyInstanceUID"])]
            v = np.stack([np.asarray(vols[p][k], np.float32) / 255.0 for p in PLANES], 0)
            if self.train:
                v = augment_volume(v, np.random.default_rng(np.random.randint(0, 2**32 - 1)))
            y = np.array([float(r[c]) for c in LAB], np.float32)
            if not self.soft:
                y = (y >= 0.5).astype(np.float32)
            w = np.float32(r.get("_w", 1.0))
            return (torch.from_numpy(v), torch.from_numpy(masks[k].astype(np.float32)),
                    torch.from_numpy(y), torch.tensor(w))

    def auc_over(loader):
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for v, m, y, _ in loader:
                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    p = torch.sigmoid(model(v.to(device), m.to(device)))
                P.append(p.float().cpu().numpy()); Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        Yb = (Y >= 0.5).astype(int)   # roc_auc_score needs BINARY truth (sel split uses soft C)
        aucs = [roc_auc_score(Yb[:, j], P[:, j]) for j in range(len(LAB)) if 0 < Yb[:, j].sum() < len(Yb)]
        return float(np.mean(aucs)), P, Y

    def train_one(tr_rows, sel_rows, pred_rows, tag):
        """Train on tr_rows (soft C + optional gold), select epoch by sel_rows (pool-C) AUC,
        return predictions on pred_rows at the selected epoch."""
        global model
        model, _ = build_model(pretrained=True)
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
        tl = DataLoader(DS(tr_rows, True, train=True), batch_size=BATCH, shuffle=True,
                        num_workers=WORKERS, drop_last=True, pin_memory=True)
        sl = DataLoader(DS(sel_rows, True), batch_size=BATCH, num_workers=WORKERS)
        pl = DataLoader(DS(pred_rows, False), batch_size=BATCH, num_workers=WORKERS)
        total = EPOCHS * max(1, len(tl)); warm = max(1, int(WARMUP_FRAC * total)); step = 0
        best = -1; best_pred = None; hist = []
        for ep in range(EPOCHS):
            model.train(); t0 = time.time()
            for v, m, y, w in tl:
                lr = (LR * (step + 1) / warm if step < warm else
                      LR * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
                for g in opt.param_groups: g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    logits = model(v.to(device), m.to(device))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, y.to(device), reduction="none").mean(1)
                    loss = (loss * w.to(device)).mean()
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update(); step += 1
            sel_auc, _, _ = auc_over(sl)
            _, predP, _ = auc_over(pl)
            hist.append({"epoch": ep + 1, "sel_pool_auc": round(sel_auc, 4), "min": round((time.time()-t0)/60, 2)})
            log(f"[{tag}] " + json.dumps(hist[-1]))
            if sel_auc > best:
                best = sel_auc; best_pred = predP
        return best_pred, best, hist

    pool_rows = [r for r in pool.to_dict("records") if str(r["StudyInstanceUID"]) in id_pos]
    if MAX_STUDIES: pool_rows = pool_rows[:MAX_STUDIES]
    gold_rows = gold.to_dict("records")
    rng = np.random.default_rng(SEED)
    WORK.mkdir(parents=True, exist_ok=True)

    if GOLD_MODE == "heldout":
        idx = rng.permutation(len(pool_rows)); nv = max(1, int(len(idx) * VAL_FRACTION))
        sel_rows = [pool_rows[i] for i in idx[:nv]]
        tr_rows = [pool_rows[i] for i in idx[nv:]]
        goldP, selbest, hist = train_one(tr_rows, sel_rows, gold_rows, "heldout")
        goldY = np.array([[float(r[c] >= 0.5) for c in LAB] for r in gold_rows])
        aucs = [roc_auc_score(goldY[:, j], goldP[:, j]) for j in range(len(LAB)) if 0 < goldY[:, j].sum() < len(goldY)]
        gold_auc = float(np.mean(aucs))
        np.save(WORK / "vol3d_v2_gold_pred.npy", goldP)
        pd.Series([str(r["StudyInstanceUID"]) for r in gold_rows]).to_csv(WORK / "vol3d_v2_gold_ids.csv", index=False)
        json.dump({"mode": "heldout", "unbiased_gold_macro_auc": gold_auc, "sel_pool_auc": selbest,
                   "history": hist}, open(WORK / "vol3d_v2_summary.json", "w"), indent=1)
        log(f"HELDOUT unbiased gold macro AUC = {gold_auc:.4f}  (sel pool {selbest:.4f})")

    elif GOLD_MODE == "folds":
        gi = rng.permutation(len(gold_rows)); chunks = np.array_split(gi, GOLD_FOLDS)
        goldP = np.zeros((len(gold_rows), len(LAB)), np.float32)
        for k in range(GOLD_FOLDS):
            te = chunks[k]; trg = [gold_rows[i] for i in np.concatenate([chunks[j] for j in range(GOLD_FOLDS) if j != k])]
            for r in trg: r["_w"] = GOLD_UPWEIGHT
            idx = rng.permutation(len(pool_rows)); nv = max(1, int(len(idx) * VAL_FRACTION))
            sel_rows = [pool_rows[i] for i in idx[:nv]]
            tr_rows = [pool_rows[i] for i in idx[nv:]] + trg
            predP, _, _ = train_one(tr_rows, sel_rows, [gold_rows[i] for i in te], f"fold{k}")
            goldP[te] = predP
        goldY = np.array([[float(r[c] >= 0.5) for c in LAB] for r in gold_rows])
        aucs = [roc_auc_score(goldY[:, j], goldP[:, j]) for j in range(len(LAB)) if 0 < goldY[:, j].sum() < len(goldY)]
        np.save(WORK / "vol3d_v2_gold_pred.npy", goldP)
        pd.Series([str(r["StudyInstanceUID"]) for r in gold_rows]).to_csv(WORK / "vol3d_v2_gold_ids.csv", index=False)
        json.dump({"mode": "folds", "crossfit_gold_macro_auc": float(np.mean(aucs))},
                  open(WORK / "vol3d_v2_summary.json", "w"), indent=1)
        log(f"CROSSFIT gold macro AUC = {float(np.mean(aucs)):.4f}")

    print("VOL3D_V2_COMPLETE", flush=True)


model = None
if __name__ == "__main__":
    main()
