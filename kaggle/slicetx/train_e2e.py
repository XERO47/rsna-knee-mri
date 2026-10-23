# -*- coding: utf-8 -*-
"""End-to-end all-slices slice-transformer: a fine-tuned pretrained ConvNeXt encoder over
ALL 64 cache64 slots -> factored positional embeddings (plane + fluid-sensitive + in-plane
position) -> transformer with a CLS pool -> 12 heads. Trained jointly on pool-C (label-masked),
pool-val epoch selection, 58 gold held out (unbiased). Reuses the cache64 machinery from
train_enc.py (Corpus / load_index / load_labels / build_backbone / augment)."""
import os, json, time, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
import train_enc as E                       # safe: its main() is __main__-guarded

LAB = E.LAB
SLOT_COUNTS = E.SLOT_COUNTS                  # [18,14,12,8,12] -> 64
SLOT_PLANE = [0, 0, 1, 1, 2]
SLOT_FS = [1, 0, 1, 0, 1]
RES = E.RES
SEED = 20260928
EPOCHS = int(os.environ.get("E2E_EPOCHS", "20"))
BS = int(os.environ.get("E2E_BS", "6"))
BB_LR = float(os.environ.get("E2E_BB_LR", "2e-5"))     # fine-tune pretrained encoder: small LR
TX_LR = float(os.environ.get("E2E_TX_LR", "3e-4"))     # transformer from scratch: bigger LR
WD = float(os.environ.get("E2E_WD", "0.05"))
DMODEL = int(os.environ.get("E2E_D", "384"))
TDEPTH = int(os.environ.get("E2E_TDEPTH", "4"))
HEADS = int(os.environ.get("E2E_HEADS", "8"))
DROP = float(os.environ.get("E2E_DROP", "0.3"))
WORKERS = int(os.environ.get("E2E_WORKERS", "12"))
WORK = Path(os.environ.get("ARM10_WORK", "/workspace/out"))


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def slot_meta(depth):
    plane, fs, pos = [], [], []
    for g, c in enumerate(SLOT_COUNTS):
        plane += [SLOT_PLANE[g]] * c; fs += [SLOT_FS[g]] * c
        pos += [(i + 0.5) / c for i in range(c)]
    assert len(plane) == depth, (len(plane), depth)
    return np.array(plane), np.array(fs), np.array(pos, np.float32)


class AllSlots(Dataset):
    def __init__(self, uids, id2row, masks, labels, lab_mask=None, train=False):
        self.uids = uids; self.id2row = id2row; self.masks = masks
        self.labels = labels; self.lab_mask = lab_mask; self.train = train; self.corpus = None

    def __len__(self): return len(self.uids)

    def __getitem__(self, i):
        if self.corpus is None:
            self.corpus = E.Corpus()
        uid = self.uids[i]; row = self.id2row[uid]; depth = self.masks.shape[1]
        sl = np.asarray(self.corpus.study(row, list(range(depth))), np.float32) / 255.0  # (D,H,W)
        tri = np.stack([sl[[max(0, c - 1), c, min(depth - 1, c + 1)]] for c in range(depth)])  # (D,3,H,W)
        x = torch.from_numpy(tri)
        if x.shape[-1] != RES:
            x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
        if self.train:
            x = E.augment(x.unsqueeze(0))[0]                      # MRI-safe aug (per-window)
        x = (x - E.IMAGENET_MEAN) / E.IMAGENET_STD
        y = torch.from_numpy(self.labels[uid])
        w = torch.ones(len(LAB)) if self.lab_mask is None else torch.from_numpy((~self.lab_mask[uid]).astype(np.float32))
        return x, torch.from_numpy(self.masks[row].astype(np.float32)), y, w


class E2E(nn.Module):
    def __init__(self, depth_slots):
        super().__init__()
        self.backbone = E.build_backbone(E.ARCH, E.PRETRAINED)
        try:
            self.backbone.set_grad_checkpointing(True)
        except Exception:
            pass
        f = self.backbone.num_features
        pl, fs, ps = slot_meta(depth_slots)
        self.register_buffer("plane_ids", torch.tensor(pl, dtype=torch.long))
        self.register_buffer("fs_ids", torch.tensor(fs, dtype=torch.long))
        self.register_buffer("pos_vals", torch.tensor(ps).unsqueeze(-1))
        self.norm = nn.LayerNorm(f)
        self.proj = nn.Linear(f, DMODEL)
        self.plane_emb = nn.Embedding(3, DMODEL)
        self.fs_emb = nn.Embedding(2, DMODEL)
        self.pos_mlp = nn.Sequential(nn.Linear(1, DMODEL), nn.GELU(), nn.Linear(DMODEL, DMODEL))
        self.cls = nn.Parameter(torch.zeros(1, 1, DMODEL)); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(DMODEL, HEADS, 4 * DMODEL, DROP, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tx = nn.TransformerEncoder(layer, TDEPTH)
        self.head = nn.Sequential(nn.LayerNorm(DMODEL), nn.Dropout(DROP), nn.Linear(DMODEL, len(LAB)))

    def forward(self, x, mask):
        B, S = x.shape[:2]
        h = self.norm(self.backbone(x.flatten(0, 1)).view(B, S, -1))         # (B,S,f)
        t = self.proj(h) + self.plane_emb(self.plane_ids) + self.fs_emb(self.fs_ids) + self.pos_mlp(self.pos_vals)
        t = torch.cat([self.cls.expand(B, -1, -1), t], 1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask.bool()], 1)
        t = self.tx(t, src_key_padding_mask=kpm)
        return self.head(t[:, 0])


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    ids, masks = E.load_index(); id2row = {u: i for i, u in enumerate(ids)}
    train_uids, gold_uids, labels = E.load_labels(ids); depth = masks.shape[1]
    mx = int(os.environ.get("E2E_MAX", "0"))
    if mx:
        train_uids = train_uids[:mx]                 # smoke cap
    lab_mask = E.build_label_mask(labels, train_uids)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={dev} depth {depth} | pool {len(train_uids)} gold {len(gold_uids)} | ARCH={E.ARCH} "
        f"PRETRAINED={E.PRETRAINED} BS={BS} EPOCHS={EPOCHS} D={DMODEL} TDEPTH={TDEPTH} RES={RES}")
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(train_uids)); nv = max(1, int(0.12 * len(train_uids)))
    val_uids = [train_uids[i] for i in perm[:nv].tolist()]; tr_uids = [train_uids[i] for i in perm[nv:].tolist()]

    model = E2E(depth).to(dev)
    resume = os.environ.get("E2E_RESUME", "")
    resume_best = -1.0
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        resume_best = float(ck.get("val_pool_auc", -1))
        log(f"RESUMED from {resume}: val {ck.get('val_pool_auc')} gold {ck.get('gold_auc')} epoch {ck.get('epoch')}")
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": BB_LR}, {"params": rest, "lr": TX_LR}], weight_decay=WD)
    base_lr = [BB_LR, TX_LR]
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    tl = DataLoader(AllSlots(tr_uids, id2row, masks, labels, lab_mask, train=True), batch_size=BS,
                    shuffle=True, num_workers=WORKERS, drop_last=True, pin_memory=True)

    def ev(uids):
        dl = DataLoader(AllSlots(uids, id2row, masks, labels), batch_size=max(2, BS), num_workers=WORKERS)
        model.eval(); P = []
        with torch.no_grad():
            for x, m, y, w in dl:
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                    P.append(torch.sigmoid(model(x.to(dev), m.to(dev)).float()).cpu().numpy())
        P = np.concatenate(P); Y = (np.stack([labels[u] for u in uids]) >= 0.5).astype(int)
        a = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(LAB)) if 0 < Y[:, j].sum() < len(Y)]
        return float(np.mean(a)), P

    total = EPOCHS * max(1, len(tl)); warm = max(1, int(0.1 * total)); step = 0; best = resume_best; hist = []
    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        for x, m, y, w in tl:
            sc = ((step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            for gi, g in enumerate(opt.param_groups): g["lr"] = base_lr[gi] * sc
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(x.to(dev), m.to(dev)); wd = w.to(dev)
                loss = (bce(logit, y.to(dev)) * wd).sum() / (wd.sum() + 1e-8)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(opt); scaler.update(); step += 1
        va, _ = ev(val_uids); ga, gP = ev(gold_uids)
        hist.append({"epoch": ep + 1, "val_pool_auc": round(va, 4), "gold_auc": round(ga, 4), "min": round((time.time() - t0) / 60, 2)})
        log("[e2e] " + json.dumps(hist[-1]))
        if va > best:
            best = va
            np.save(WORK / "e2e_gold_pred.npy", gP)
            pd.Series(gold_uids).to_csv(WORK / "e2e_gold_ids.csv", index=False)
            torch.save({"model": model.state_dict(), "val_pool_auc": best, "gold_auc": ga, "epoch": ep + 1}, WORK / "e2e_best.pt")
    json.dump({"best_val_pool_auc": best, "final_gold_auc": hist[-1]["gold_auc"], "history": hist}, open(WORK / "e2e_summary.json", "w"), indent=1)
    log(f"DONE best val-pool {best:.4f}")
    print("E2E_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
