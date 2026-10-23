# -*- coding: utf-8 -*-
"""Slice-transformer over ALL 64 cache64 slots (your all-slices vision), on precomputed
frozen-backbone slot features. Each slot = a token; factored positional embedding encodes
plane (sag/cor/axial) + fluid-sensitive + in-plane position; a CLS token pools the study.
Variable validity handled by a key-padding mask (missing slots ignored). Trains on pool-C
with a held-out pool-val for epoch selection and the 58 gold held out for an UNBIASED AUC.

Cheap gate: does transformer-pooling over all slots beat the arms' attention-MIL (and
decorrelate)? If yes -> fine-tune the encoder end-to-end next.
"""
import json, os, time, math, glob
from pathlib import Path
import numpy as np

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOT_COUNTS = [18, 14, 12, 8, 12]          # Sag-fs, Sag-anat, Cor-fs, Cor-anat, Axial
SLOT_PLANE = [0, 0, 1, 1, 2]               # sagittal, coronal, axial
SLOT_FS    = [1, 0, 1, 0, 1]               # fluid-sensitive flag per slot-group
SEED = 20260928
EMB = os.environ.get("TX_EMB", "/workspace/out/slot_emb.npz")
DATA = os.environ.get("TX_DATA", "/workspace/data")
WORK = Path(os.environ.get("TX_WORK", "/workspace/out"))
EPOCHS = int(os.environ.get("TX_EPOCHS", "60"))
BS = int(os.environ.get("TX_BS", "32"))
LR = float(os.environ.get("TX_LR", "3e-4"))
WD = float(os.environ.get("TX_WD", "0.05"))
DMODEL = int(os.environ.get("TX_D", "256"))
DEPTH = int(os.environ.get("TX_DEPTH", "4"))
HEADS = int(os.environ.get("TX_HEADS", "8"))
DROP = float(os.environ.get("TX_DROP", "0.3"))
MASK_PRIOR, MASK_TOL = 0.25, 1e-6


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def slot_meta(depth):
    plane, fs, pos = [], [], []
    for g, c in enumerate(SLOT_COUNTS):
        plane += [SLOT_PLANE[g]] * c
        fs += [SLOT_FS[g]] * c
        pos += [(i + 0.5) / c for i in range(c)]     # normalized in-plane position
    assert len(plane) == depth, (len(plane), depth)
    return np.array(plane), np.array(fs), np.array(pos, np.float32)


def build():
    import torch, torch.nn as nn

    class SliceTx(nn.Module):
        def __init__(self, fdim, depth_slots):
            super().__init__()
            pl, fs, ps = slot_meta(depth_slots)
            self.register_buffer("plane_ids", torch.tensor(pl, dtype=torch.long))
            self.register_buffer("fs_ids", torch.tensor(fs, dtype=torch.long))
            self.register_buffer("pos_vals", torch.tensor(ps).unsqueeze(-1))
            self.proj = nn.Linear(fdim, DMODEL)
            self.plane_emb = nn.Embedding(3, DMODEL)
            self.fs_emb = nn.Embedding(2, DMODEL)
            self.pos_mlp = nn.Sequential(nn.Linear(1, DMODEL), nn.GELU(), nn.Linear(DMODEL, DMODEL))
            self.cls = nn.Parameter(torch.zeros(1, 1, DMODEL))
            nn.init.trunc_normal_(self.cls, std=0.02)
            layer = nn.TransformerEncoderLayer(DMODEL, HEADS, 4 * DMODEL, DROP,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.tx = nn.TransformerEncoder(layer, DEPTH)
            self.head = nn.Sequential(nn.LayerNorm(DMODEL), nn.Dropout(DROP), nn.Linear(DMODEL, len(LAB)))

        def forward(self, emb, mask):
            B = emb.shape[0]
            t = self.proj(emb)
            t = t + self.plane_emb(self.plane_ids) + self.fs_emb(self.fs_ids) + self.pos_mlp(self.pos_vals)
            t = torch.cat([self.cls.expand(B, -1, -1), t], 1)                 # (B, 1+S, d)
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=emb.device),
                             ~mask.bool()], 1)                                # True = ignore
            t = self.tx(t, src_key_padding_mask=kpm)
            return self.head(t[:, 0])
    return SliceTx


def find(name):
    h = glob.glob(f"{DATA}/**/{name}", recursive=True)
    if not h:
        raise FileNotFoundError(name)
    return h[0]


def main():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    torch.manual_seed(SEED); np.random.seed(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    z = np.load(EMB, allow_pickle=True)
    emb = z["emb"]; masks = z["masks"]; ids = [str(u) for u in z["ids"]]; depth = emb.shape[1]
    id2i = {u: i for i, u in enumerate(ids)}
    log(f"emb {emb.shape} feat_dim {int(z['feat_dim'])} depth {depth}")
    pool = pd.read_csv(find("train_pool.csv")); pool["StudyInstanceUID"] = pool.StudyInstanceUID.astype(str)
    gold = pd.read_csv(find("gold_test.csv")); gold["StudyInstanceUID"] = gold.StudyInstanceUID.astype(str)
    pool = pool[pool.StudyInstanceUID.isin(id2i)]; gold = gold[gold.StudyInstanceUID.isin(id2i)]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensors(df, soft):
        uids = df.StudyInstanceUID.tolist(); idx = [id2i[u] for u in uids]
        X = torch.from_numpy(emb[idx].astype(np.float32)); M = torch.from_numpy(masks[idx].astype(np.float32))
        Y = df[LAB].to_numpy(np.float32)
        W = np.ones_like(Y) if soft else np.ones_like(Y)
        if soft:
            W = (np.abs(Y - MASK_PRIOR) >= MASK_TOL).astype(np.float32)   # mask 'report-absent' prior
        return TensorDataset(X, M, torch.from_numpy(Y), torch.from_numpy(W)), uids

    rng = np.random.default_rng(SEED)
    pidx = rng.permutation(len(pool)); nv = max(1, int(0.12 * len(pool)))
    val_df = pool.iloc[pidx[:nv]]; tr_df = pool.iloc[pidx[nv:]]
    tr_ds, _ = tensors(tr_df, True); val_ds, _ = tensors(val_df, True); gold_ds, gold_uids = tensors(gold, False)

    SliceTx = build()
    model = SliceTx(emb.shape[2], depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    tl = DataLoader(tr_ds, batch_size=BS, shuffle=True, drop_last=True)

    def ev(ds):
        dl = DataLoader(ds, batch_size=128); model.eval(); P = []; Y = []
        with torch.no_grad():
            for X, M, y, w in dl:
                P.append(torch.sigmoid(model(X.to(dev), M.to(dev))).cpu().numpy()); Y.append(y.numpy())
        P = np.concatenate(P); Y = (np.concatenate(Y) >= 0.5).astype(int)
        a = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(LAB)) if 0 < Y[:, j].sum() < len(Y)]
        return float(np.mean(a)), P

    total = EPOCHS * max(1, len(tl)); warm = max(1, int(0.1 * total)); step = 0
    best = -1.0; hist = []
    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        for X, M, y, w in tl:
            lr = LR * ((step + 1) / warm if step < warm else
                       0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            logit = model(X.to(dev), M.to(dev)); wdev = w.to(dev)
            loss = (bce(logit, y.to(dev)) * wdev).sum() / (wdev.sum() + 1e-8)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); step += 1
        va, _ = ev(val_ds); ga, gP = ev(gold_ds)
        hist.append({"epoch": ep + 1, "val_pool_auc": round(va, 4), "gold_auc": round(ga, 4),
                     "min": round((time.time() - t0) / 60, 2)})
        log("[tx] " + json.dumps(hist[-1]))
        if va > best:
            best = va
            np.save(WORK / "tx_gold_pred.npy", gP)
            pd.Series(gold_uids).to_csv(WORK / "tx_gold_ids.csv", index=False)
            torch.save({"model": model.state_dict(), "val_pool_auc": best, "gold_auc": ga, "epoch": ep + 1},
                       WORK / "tx_best.pt")
    json.dump({"best_val_pool_auc": best, "final_gold_auc": hist[-1]["gold_auc"], "history": hist,
               "n_train": len(tr_df), "config": {"d": DMODEL, "depth": DEPTH, "heads": HEADS}},
              open(WORK / "tx_summary.json", "w"), indent=1)
    log(f"DONE best val-pool {best:.4f}")
    print("SLICE_TX_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
