"""Variable-length slice-transformer over the full-cache embeddings.

Every slice of every series is a token (no 64-slot bottleneck). The positional code is
built from the PER-SLICE metadata the cache carries -- plane (sag/cor/ax), fluid-sensitive,
which series it came from, and its physical position in that series -- not a fixed slot
table. A CLS token pools the study; a key-padding mask handles the variable slice count.

Trains on pool-C (report labels, 'absent'-prior masked), selects epochs on a held-out
pool-val, and scores the 58 image-gold held out (unbiased). Saves gold preds so the
residual correlation vs the 0.904 ensemble can be checked -- the whole point is whether
feeding ALL slices at 448 finally decorrelates from the cache64 models.
"""
import json, os, time, math, glob
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEED = 20260928
EMB = os.environ.get("FULL_EMB", "/workspace/full_emb")
DATA = os.environ.get("FULL_DATA", "/workspace/data")
WORK = Path(os.environ.get("FULL_WORK", "/workspace/out"))
EPOCHS = int(os.environ.get("FULL_EPOCHS", "40"))
BS = int(os.environ.get("FULL_BS", "16"))
LR = float(os.environ.get("FULL_LR", "3e-4"))
WD = float(os.environ.get("FULL_WD", "0.05"))
DMODEL = int(os.environ.get("FULL_D", "384"))
DEPTH = int(os.environ.get("FULL_DEPTH", "6"))
HEADS = int(os.environ.get("FULL_HEADS", "8"))
DROP = float(os.environ.get("FULL_DROP", "0.3"))
MAXLEN = int(os.environ.get("FULL_MAXLEN", "320"))     # safety cap on slices/study (p-max ~300)
NW = int(os.environ.get("FULL_NW", "6"))               # DataLoader workers (0 for local dry-run)
POOL = os.environ.get("FULL_POOL", "cls")              # cls | attn | mean  (study-level aggregation)
TAG = os.environ.get("FULL_TAG", "")                   # prefix for output files (sweep runs)
MAX_SERIES = 16
MASK_PRIOR, MASK_TOL = 0.25, 1e-6


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find(name):
    h = glob.glob(f"{DATA}/**/{name}", recursive=True)
    if not h:
        raise FileNotFoundError(name)
    return h[0]


class Studies(Dataset):
    def __init__(self, uids, loc, emb, meta, labels, soft):
        self.uids, self.loc, self.emb, self.meta = uids, loc, emb, meta
        self.labels, self.soft = labels, soft

    def __len__(self): return len(self.uids)

    def __getitem__(self, i):
        u = self.uids[i]; off, cnt = self.loc[u]
        if cnt > MAXLEN:                                  # even subsample if a study is huge
            sel = np.linspace(0, cnt - 1, MAXLEN).round().astype(int)
            e = self.emb[off + sel]; m = self.meta[off + sel]
        else:
            e = self.emb[off:off + cnt]; m = self.meta[off:off + cnt]
        y = self.labels[u].astype(np.float32)
        w = (np.abs(y - MASK_PRIOR) >= MASK_TOL).astype(np.float32) if self.soft else np.ones_like(y)
        return (torch.from_numpy(e.astype(np.float32)), torch.from_numpy(m.astype(np.float32)),
                torch.from_numpy(y), torch.from_numpy(w))


def collate(batch):
    es, ms, ys, ws = zip(*batch)
    K = max(e.shape[0] for e in es); F = es[0].shape[1]
    B = len(batch)
    emb = torch.zeros(B, K, F); meta = torch.zeros(B, K, 4); pad = torch.ones(B, K, dtype=torch.bool)
    for b, (e, m) in enumerate(zip(es, ms)):
        k = e.shape[0]; emb[b, :k] = e; meta[b, :k] = m; pad[b, :k] = False
    return emb, meta, pad, torch.stack(ys), torch.stack(ws)


class FullTx(nn.Module):
    def __init__(self, fdim):
        super().__init__()
        self.proj = nn.Linear(fdim, DMODEL)
        self.plane_emb = nn.Embedding(3, DMODEL)
        self.fs_emb = nn.Embedding(2, DMODEL)
        self.series_emb = nn.Embedding(MAX_SERIES, DMODEL)
        self.pos_mlp = nn.Sequential(nn.Linear(1, DMODEL), nn.GELU(), nn.Linear(DMODEL, DMODEL))
        self.cls = nn.Parameter(torch.zeros(1, 1, DMODEL)); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(DMODEL, HEADS, 4 * DMODEL, DROP, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tx = nn.TransformerEncoder(layer, DEPTH)
        if POOL == "attn":                                   # learned-query attention pooling
            self.pool_q = nn.Parameter(torch.zeros(1, 1, DMODEL)); nn.init.trunc_normal_(self.pool_q, std=0.02)
            self.pool_attn = nn.MultiheadAttention(DMODEL, HEADS, dropout=DROP, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(DMODEL), nn.Dropout(DROP), nn.Linear(DMODEL, len(LAB)))

    def forward(self, emb, meta, pad):
        B = emb.shape[0]
        plane = meta[:, :, 0].long().clamp(0, 2); fs = meta[:, :, 1].long().clamp(0, 1)
        ser = meta[:, :, 2].long().clamp(0, MAX_SERIES - 1); pos = meta[:, :, 3:4]
        t = self.proj(emb) + self.plane_emb(plane) + self.fs_emb(fs) + self.series_emb(ser) + self.pos_mlp(pos)
        if POOL == "cls":
            t = torch.cat([self.cls.expand(B, -1, -1), t], 1)
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=emb.device), pad], 1)
            return self.head(self.tx(t, src_key_padding_mask=kpm)[:, 0])
        t = self.tx(t, src_key_padding_mask=pad)             # no CLS; pool the token outputs
        if POOL == "attn":
            pooled, _ = self.pool_attn(self.pool_q.expand(B, -1, -1), t, t, key_padding_mask=pad)
            return self.head(pooled[:, 0])
        m = (~pad).float().unsqueeze(-1)                     # mean: masked average over valid tokens
        return self.head((t * m).sum(1) / m.sum(1).clamp(min=1))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    emb = np.load(f"{EMB}/full_emb.npy", mmap_mode="r"); meta = np.load(f"{EMB}/full_meta.npy", mmap_mode="r")
    idx = np.load(f"{EMB}/full_idx.npy"); ids = [str(u) for u in np.load(f"{EMB}/full_ids.npy")]
    loc = {u: (int(idx[i, 0]), int(idx[i, 1])) for i, u in enumerate(ids)}
    fdim = emb.shape[1]
    log(f"emb {emb.shape} fdim {fdim} | studies {len(ids)}")

    pool = pd.read_csv(find("train_pool.csv")); pool["StudyInstanceUID"] = pool.StudyInstanceUID.astype(str)
    gold = pd.read_csv(find("gold_test.csv")); gold["StudyInstanceUID"] = gold.StudyInstanceUID.astype(str)
    pool = pool[pool.StudyInstanceUID.isin(loc)]; gold = gold[gold.StudyInstanceUID.isin(loc)]
    labels = {str(row["StudyInstanceUID"]): row[LAB].to_numpy(np.float32)
              for _, row in pd.concat([pool, gold]).iterrows()}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(SEED)
    pu = pool.StudyInstanceUID.tolist(); perm = rng.permutation(len(pu)); nv = max(1, int(0.12 * len(pu)))
    val_uids = [pu[i] for i in perm[:nv]]; tr_uids = [pu[i] for i in perm[nv:]]
    gold_uids = gold.StudyInstanceUID.tolist()
    log(f"pool {len(pu)} -> train {len(tr_uids)} val {len(val_uids)} | gold {len(gold_uids)} | dev {dev}")

    emb_f = np.asarray(emb); meta_f = np.asarray(meta)      # materialize (~1-2 GB) for fast indexing
    mk = lambda u, soft: Studies(u, loc, emb_f, meta_f, labels, soft)
    tl = DataLoader(mk(tr_uids, True), BS, shuffle=True, drop_last=True, num_workers=NW,
                    collate_fn=collate, pin_memory=True)
    model = FullTx(fdim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def ev(uids):
        dl = DataLoader(mk(uids, False), 32, num_workers=NW, collate_fn=collate)
        model.eval(); P = []
        with torch.no_grad():
            for e, m, p, y, w in dl:
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                    P.append(torch.sigmoid(model(e.to(dev), m.to(dev), p.to(dev)).float()).cpu().numpy())
        P = np.concatenate(P); Y = (np.stack([labels[u] for u in uids]) >= 0.5).astype(int)
        a = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(LAB)) if 0 < Y[:, j].sum() < len(Y)]
        return float(np.mean(a)), P

    pfx = f"full_tx_{TAG}" if TAG else "full_tx"
    pd.Series(gold_uids).to_csv(WORK / f"{pfx}_gold_ids.csv", index=False)
    total = EPOCHS * max(1, len(tl)); warm = max(1, int(0.1 * total)); step = 0
    best_val = -1.0; gold_at_best_val = -1.0; best_gold = -1.0; hist = []
    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        for e, m, p, y, w in tl:
            lr = LR * ((step + 1) / warm if step < warm else
                       0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(e.to(dev), m.to(dev), p.to(dev)); wd = w.to(dev)
                loss = (bce(logit, y.to(dev)) * wd).sum() / (wd.sum() + 1e-8)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); step += 1
        va, _ = ev(val_uids); ga, gP = ev(gold_uids)
        hist.append({"epoch": ep + 1, "val_pool_auc": round(va, 4), "gold_auc": round(ga, 4),
                     "min": round((time.time() - t0) / 60, 2)})
        log("[full_tx] " + json.dumps(hist[-1]))
        ck = {"model": model.state_dict(), "val_pool_auc": va, "gold_auc": ga, "epoch": ep + 1,
              "config": {"pool": POOL, "d": DMODEL, "depth": DEPTH, "heads": HEADS, "maxlen": MAXLEN}}
        # (1) unbiased pick: best on held-out pool-val -> report its gold as the honest number
        if va > best_val:
            best_val = va; gold_at_best_val = ga
            np.save(WORK / f"{pfx}_gold_pred.npy", gP)
            torch.save(ck, WORK / f"{pfx}_best.pt")
        # (2) oracle pick: best gold directly (BIASED, for ceiling/reference only) -- the aim is >=0.92
        if ga > best_gold:
            best_gold = ga
            np.save(WORK / f"{pfx}_gold_pred_oracle.npy", gP)
            torch.save(ck, WORK / f"{pfx}_bestgold.pt")
        # (3) always keep the latest model regardless of results (user: save irrespective)
        torch.save(ck, WORK / f"{pfx}_final.pt")
    json.dump({"best_val_pool_auc": round(best_val, 4),
               "gold_at_best_val_UNBIASED": round(gold_at_best_val, 4),
               "best_gold_ORACLE_biased": round(best_gold, 4),
               "final_gold_auc": hist[-1]["gold_auc"], "target_gold": 0.92,
               "passes_0.92_unbiased": gold_at_best_val >= 0.92,
               "history": hist, "n_pool": len(pu), "n_gold": len(gold_uids),
               "config": {"pool": POOL, "d": DMODEL, "depth": DEPTH, "heads": HEADS, "maxlen": MAXLEN,
                          "epochs": EPOCHS, "bs": BS, "lr": LR}},
              open(WORK / f"{pfx}_summary.json", "w"), indent=1)
    log(f"[{TAG or 'base'}] gold@best-val (unbiased) {gold_at_best_val:.4f} | best-gold {best_gold:.4f} | val {best_val:.4f}")
    print("FULL_TX_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
