"""End-to-end fine-tune over the full-image all-slices cache.

Same idea as the embed+transformer, but the encoder is NO LONGER frozen: each step feeds
N real 448px slices per study through a fine-tunable encoder -> per-slice features ->
slice-transformer -> 12 findings, trained jointly on pool-C (gold held out). This is what
gets task-tuned features (the frozen embed path capped ~0.73; fine-tuned arms hit ~0.835).

To fit memory/time, each study contributes N sampled slices per epoch (random subset =
augmentation), spread across its series; within-series triplets give through-plane context.
Reads the full cache via load_full.FullCache (raw pixels, ~160GB on a big-disk pod).
"""
import json, os, time, math, glob
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
import timm
from load_full import FullCache

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEED = 20260928
ROOT = os.environ.get("FULL_ROOT", "/workspace/full")
DATA = os.environ.get("FULL_DATA", "/workspace/data")
WORK = Path(os.environ.get("FULL_WORK", "/workspace/out"))
ARCH = os.environ.get("E2E_ARCH", "vit_small_patch14_dinov2.lvd142m")  # 'that encoder' (SSL) fine-tuned; convnext_small.* also works
PRETRAINED = os.environ.get("E2E_PRETRAINED", "1") == "1"
RES = int(os.environ.get("E2E_RES", "336"))
NSLICE = int(os.environ.get("E2E_NSLICE", "32"))        # slices/study per step (sampled)
EPOCHS = int(os.environ.get("E2E_EPOCHS", "15"))
BS = int(os.environ.get("E2E_BS", "8"))
BB_LR = float(os.environ.get("E2E_BB_LR", "2e-5"))      # encoder: small LR (fine-tune)
TX_LR = float(os.environ.get("E2E_TX_LR", "2e-4"))      # transformer/head: bigger LR
WD = float(os.environ.get("E2E_WD", "0.05"))
DMODEL = int(os.environ.get("E2E_D", "384"))
DEPTH = int(os.environ.get("E2E_DEPTH", "4"))
HEADS = int(os.environ.get("E2E_HEADS", "8"))
DROP = float(os.environ.get("E2E_DROP", "0.3"))
NW = int(os.environ.get("E2E_NW", "8"))
TAG = os.environ.get("E2E_TAG", "")
MAX_SERIES = 16
MASK_PRIOR, MASK_TOL = 0.25, 1e-6
IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
ISTD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find(name):
    h = glob.glob(f"{DATA}/**/{name}", recursive=True)
    if not h:
        raise FileNotFoundError(name)
    return h[0]


class StudyDS(Dataset):
    """Returns N sampled slices/study as triplets + per-slice meta + a valid-mask."""
    def __init__(self, uids, labels, soft, train):
        self.uids, self.labels, self.soft, self.train = uids, labels, soft, train
        self.fc = None

    def __len__(self): return len(self.uids)

    def __getitem__(self, i):
        if self.fc is None:
            self.fc = FullCache(ROOT)                    # open per-worker (memmaps)
        u = self.uids[i]; s, row = self.fc.uid2loc[u]
        off, cnt = (int(x) for x in self.fc.index[s][row])
        meta = np.asarray(self.fc.meta[s][off:off + cnt])            # (K,4) tiny
        ser = meta[:, 2].astype(int)
        rng = np.random
        if cnt > NSLICE:
            targ = np.sort(rng.choice(cnt, NSLICE, replace=False) if self.train
                           else np.linspace(0, cnt - 1, NSLICE).round().astype(int))
        else:
            targ = np.arange(cnt)
        prev = np.where((targ > 0) & (ser[np.clip(targ - 1, 0, cnt - 1)] == ser[targ]), targ - 1, targ)
        nxt = np.where((targ < cnt - 1) & (ser[np.clip(targ + 1, 0, cnt - 1)] == ser[targ]), targ + 1, targ)
        slc = self.fc.slices[s]
        tri = np.stack([np.asarray(slc[off + prev]), np.asarray(slc[off + targ]),
                        np.asarray(slc[off + nxt])], 1).astype(np.float32) / 255.0   # (n,3,H,W)
        xt = torch.from_numpy(tri)
        if xt.shape[-1] != RES:
            xt = F.interpolate(xt, size=(RES, RES), mode="bilinear", align_corners=False)
        x = torch.zeros(NSLICE, 3, RES, RES)
        mt = np.zeros((NSLICE, 4), np.float32); valid = torch.zeros(NSLICE, dtype=torch.bool)
        n = len(targ); x[:n] = xt; mt[:n] = meta[targ]; valid[:n] = True
        y = self.labels[u].astype(np.float32)
        w = (np.abs(y - MASK_PRIOR) >= MASK_TOL).astype(np.float32) if self.soft else np.ones_like(y)
        return x, torch.from_numpy(mt), ~valid, torch.from_numpy(y), torch.from_numpy(w)


class E2E(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_vit = "dinov2" in ARCH or ARCH.startswith("vit")
        if self.is_vit:
            self.enc = timm.create_model(ARCH, pretrained=PRETRAINED, num_classes=0, img_size=RES)
            self.nprefix = int(getattr(self.enc, "num_prefix_tokens", 1))
            feat = self.enc.num_features * 2            # CLS (+) mean-patch
        else:
            self.enc = timm.create_model(ARCH, pretrained=PRETRAINED, num_classes=0, global_pool="avg")
            feat = self.enc.num_features
        try: self.enc.set_grad_checkpointing(True)
        except Exception: pass
        self.proj = nn.Linear(feat, DMODEL)
        self.plane_emb = nn.Embedding(3, DMODEL); self.fs_emb = nn.Embedding(2, DMODEL)
        self.series_emb = nn.Embedding(MAX_SERIES, DMODEL)
        self.pos_mlp = nn.Sequential(nn.Linear(1, DMODEL), nn.GELU(), nn.Linear(DMODEL, DMODEL))
        self.cls = nn.Parameter(torch.zeros(1, 1, DMODEL)); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(DMODEL, HEADS, 4 * DMODEL, DROP, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tx = nn.TransformerEncoder(layer, DEPTH)
        self.head = nn.Sequential(nn.LayerNorm(DMODEL), nn.Dropout(DROP), nn.Linear(DMODEL, len(LAB)))

    def encode(self, x):                                 # (M,3,RES,RES) -> (M,feat)
        if self.is_vit:
            t = self.enc.forward_features(x)
            return torch.cat([t[:, 0], t[:, self.nprefix:].mean(1)], -1)
        return self.enc(x)

    def forward(self, x, meta, pad):
        B, S = x.shape[:2]
        f = self.encode(x.flatten(0, 1)).view(B, S, -1)
        plane = meta[:, :, 0].long().clamp(0, 2); fs = meta[:, :, 1].long().clamp(0, 1)
        ser = meta[:, :, 2].long().clamp(0, MAX_SERIES - 1); pos = meta[:, :, 3:4]
        t = self.proj(f) + self.plane_emb(plane) + self.fs_emb(fs) + self.series_emb(ser) + self.pos_mlp(pos)
        t = torch.cat([self.cls.expand(B, -1, -1), t], 1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], 1)
        return self.head(self.tx(t, src_key_padding_mask=kpm)[:, 0])


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc0 = FullCache(ROOT); have = set(fc0.uids())
    pool = pd.read_csv(find("train_pool.csv")); pool["StudyInstanceUID"] = pool.StudyInstanceUID.astype(str)
    gold = pd.read_csv(find("gold_test.csv")); gold["StudyInstanceUID"] = gold.StudyInstanceUID.astype(str)
    pool = pool[pool.StudyInstanceUID.isin(have)]; gold = gold[gold.StudyInstanceUID.isin(have)]
    labels = {str(r["StudyInstanceUID"]): r[LAB].to_numpy(np.float32)
              for _, r in pd.concat([pool, gold]).iterrows()}
    rng = np.random.default_rng(SEED); pu = pool.StudyInstanceUID.tolist()
    perm = rng.permutation(len(pu)); nv = max(1, int(0.12 * len(pu)))
    val_uids = [pu[i] for i in perm[:nv]]; tr_uids = [pu[i] for i in perm[nv:]]
    gold_uids = gold.StudyInstanceUID.tolist()
    pfx = f"full_e2e_{TAG}" if TAG else "full_e2e"
    log(f"ARCH {ARCH} RES {RES} NSLICE {NSLICE} | pool {len(pu)} tr {len(tr_uids)} val {len(val_uids)} gold {len(gold_uids)} | dev {dev}")

    model = E2E().to(dev)
    bb = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": BB_LR}, {"params": rest, "lr": TX_LR}], weight_decay=WD)
    base_lr = [BB_LR, TX_LR]
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mean, std = IMEAN.to(dev), ISTD.to(dev)
    tl = DataLoader(StudyDS(tr_uids, labels, True, True), BS, shuffle=True, drop_last=True,
                    num_workers=NW, pin_memory=True, persistent_workers=NW > 0)

    def ev(uids):
        dl = DataLoader(StudyDS(uids, labels, False, False), max(2, BS), num_workers=NW)
        model.eval(); P = []
        with torch.no_grad():
            for x, m, pad, y, w in dl:
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                    xin = (x.to(dev) - mean.view(1, 1, 3, 1, 1)) / std.view(1, 1, 3, 1, 1)
                    P.append(torch.sigmoid(model(xin, m.to(dev), pad.to(dev)).float()).cpu().numpy())
        P = np.concatenate(P); Y = (np.stack([labels[u] for u in uids]) >= 0.5).astype(int)
        a = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(LAB)) if 0 < Y[:, j].sum() < len(Y)]
        return float(np.mean(a)), P

    pd.Series(gold_uids).to_csv(WORK / f"{pfx}_gold_ids.csv", index=False)
    total = EPOCHS * max(1, len(tl)); warm = max(1, int(0.05 * total)); step = 0
    best_val = -1.0; gold_at_best = -1.0; best_gold = -1.0; hist = []
    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        for x, m, pad, y, w in tl:
            sc = ((step + 1) / warm if step < warm else
                  0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            for gi, g in enumerate(opt.param_groups): g["lr"] = base_lr[gi] * sc
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                xin = (x.to(dev) - mean.view(1, 1, 3, 1, 1)) / std.view(1, 1, 3, 1, 1)
                logit = model(xin, m.to(dev), pad.to(dev)); wd = w.to(dev)
                loss = (bce(logit, y.to(dev)) * wd).sum() / (wd.sum() + 1e-8)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(opt); scaler.update(); step += 1
        va, _ = ev(val_uids); ga, gP = ev(gold_uids)
        hist.append({"epoch": ep + 1, "val_pool_auc": round(va, 4), "gold_auc": round(ga, 4),
                     "min": round((time.time() - t0) / 60, 2)})
        log("[full_e2e] " + json.dumps(hist[-1]))
        ck = {"model": model.state_dict(), "val_pool_auc": va, "gold_auc": ga, "epoch": ep + 1, "arch": ARCH}
        if va > best_val:
            best_val = va; gold_at_best = ga
            np.save(WORK / f"{pfx}_gold_pred.npy", gP); torch.save(ck, WORK / f"{pfx}_best.pt")
        if ga > best_gold:
            best_gold = ga; np.save(WORK / f"{pfx}_gold_pred_oracle.npy", gP); torch.save(ck, WORK / f"{pfx}_bestgold.pt")
        torch.save(ck, WORK / f"{pfx}_final.pt")
    json.dump({"best_val_pool_auc": round(best_val, 4), "gold_at_best_val_UNBIASED": round(gold_at_best, 4),
               "best_gold_ORACLE_biased": round(best_gold, 4), "final_gold_auc": hist[-1]["gold_auc"],
               "passes_0.92_unbiased": gold_at_best >= 0.92, "arch": ARCH, "nslice": NSLICE, "res": RES,
               "history": hist}, open(WORK / f"{pfx}_summary.json", "w"), indent=1)
    log(f"[{TAG or 'e2e'}] gold@best-val {gold_at_best:.4f} | best-gold {best_gold:.4f} | val {best_val:.4f}")
    print("FULL_E2E_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
