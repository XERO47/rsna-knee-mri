"""arm7: now that perception is fixed, retest the two things arm4 got blamed for.

arm4 (transformer + latents + finding queries, ~160 slices) lost badly --
0.6679 vs arm3's 0.6791. My diagnosis at the time: it upgraded the *reasoning*
layer while the *perception* layer stayed blind (from-scratch CNN at 160px),
so richer aggregation had nothing good to aggregate. arm5/arm6 then fixed
perception with pretrained weights and jumped to ~0.847 using only simple
per-finding attention pooling over 32 slices.

That makes arm4's verdict provisional, not final. Its core idea -- letting
slices read each other -- deserves a rerun on good features.

REVISED under a ~4.6h GPU quota ceiling: run ONLY the interaction arm, at
32 slices/case matched to arm6. That makes arm7b vs arm6 a clean
single-variable test -- same backbone, same slice count, same pipeline, same
seed; the ONLY difference is cross-slice self-attention. It also costs ~1h
instead of ~4h. The separate "do more slices help" question is deferred; it is
only worth quota if interaction proves out.

The model projects 2048->512 before pooling in both configurations, so the
cross-slice path adds *only* the transformer, never a dimension change --
keeping the comparison against arm6 attributable to attention alone.

Deliberately lighter than arm4's aggregation: 2 encoder layers, no latent
array, no separate finding-query block. arm4 was 32.7M params and plateaued
while train loss kept falling (overfitting 4,349 noisy-labelled cases). The
question here is whether slice interaction helps at all, not whether maximum
capacity helps -- so this is the minimum architecture that answers it.

Backbone: ImageNet ResNet-50 from the CC0 dataset (no NC/SA clauses, so no
Section 2.8 exposure).
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]
N_FINDINGS = 12
BACKBONE_DIM = 2048
D_MODEL = 512             # both arms project here, so arm7b adds only attention
N_ATTN_LAYERS = 2
N_HEADS = 8
DROPOUT = 0.1
N_SLOT_TYPES = 8

SLICES_PER_CASE = 32      # MATCHED to arm6 -- see note below
IMG_SIZE = 160
BATCH_CASES = 2
ACCUM_STEPS = 8           # effective batch 16 cases
EPOCHS = 12               # arm5 hit 0.845 by ep8; 12 captures the useful range
LR = 3e-4
BACKBONE_LR_MULT = 0.1
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
WARMUP_FRAC = 0.1
TIME_BUDGET_S_PER_ARM = 1.6 * 3600
SEED = 42

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

torch.manual_seed(SEED)
np.random.seed(SEED)


def find_root(input_root: Path, marker: str, skip_dirs=("train_series", "test_series")):
    if not input_root.exists():
        return None
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    return None


INPUT_ROOT = Path("/kaggle/input")
LABELS_ROOT = find_root(INPUT_ROOT, "gold_test.csv")
WEIGHTS_ROOT = find_root(INPUT_ROOT, "resnet50-0676ba61.pth")
SHARDS = [(s, r) for s, r in ((s, find_root(INPUT_ROOT, f"index_shard{s}.json"))
                              for s in (0, 1)) if r]
WORK = Path("/kaggle/working")
print(f"LABELS_ROOT={LABELS_ROOT}")
print(f"WEIGHTS_ROOT={WEIGHTS_ROOT}")
print(f"all-slice shards: {len(SHARDS)}")


class CaseCache:
    """Ragged all-slice cache; also returns per-slice slot + normalised depth,
    which arm7b's embeddings consume (arm5/arm6 ignored this metadata)."""

    def __init__(self, shards):
        self.uid_to_loc = {}
        self.arrays, self.indexes = [], []
        for pos, (shard_no, root) in enumerate(shards):
            idx = json.load(open(root / f"index_shard{shard_no}.json"))
            arr = np.load(root / f"slices_shard{shard_no}.npy", mmap_mode="r")
            self.arrays.append(arr)
            self.indexes.append(idx)
            for ci, uid in enumerate(idx["uids"]):
                self.uid_to_loc[uid] = (pos, ci)
            print(f"  shard {shard_no}: {arr.shape}, {len(idx['uids'])} cases")

    def __contains__(self, uid):
        return uid in self.uid_to_loc

    def get(self, uid):
        pos, ci = self.uid_to_loc[uid]
        idx, arr = self.indexes[pos], self.arrays[pos]
        lo, hi = idx["offsets"][ci], idx["offsets"][ci + 1]
        return (np.asarray(arr[lo:hi], dtype=np.uint8),
                np.array(idx["slot_of_slice"][lo:hi], np.int64),
                np.array(idx["depth_frac_of_slice"][lo:hi], np.float32))


class CaseDataset(Dataset):
    def __init__(self, studies_df, cache, train: bool):
        keep = [u for u in studies_df.StudyInstanceUID if u in cache]
        self.studies = studies_df[studies_df.StudyInstanceUID.isin(keep)].reset_index(drop=True)
        self.cache, self.train = cache, train
        dropped = len(studies_df) - len(self.studies)
        if dropped:
            print(f"  {dropped} studies missing from cache, skipped")

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        row = self.studies.iloc[i]
        slices, slots, depth = self.cache.get(row["StudyInstanceUID"])
        n = len(slices)
        if n >= SLICES_PER_CASE:
            if self.train:
                pick = np.sort(np.random.choice(n, SLICES_PER_CASE, replace=False))
            else:
                pick = np.linspace(0, n - 1, SLICES_PER_CASE).round().astype(int)
            mask = np.ones(SLICES_PER_CASE, np.float32)
        else:
            pick = np.concatenate([np.arange(n), np.full(SLICES_PER_CASE - n, n - 1)])
            mask = np.concatenate([np.ones(n, np.float32),
                                   np.zeros(SLICES_PER_CASE - n, np.float32)])
        return {
            "bag": torch.from_numpy(slices[pick].astype(np.float32) / 255.0).unsqueeze(1),
            "slots": torch.from_numpy(slots[pick]),
            "depth": torch.from_numpy(depth[pick]),
            "mask": torch.from_numpy(mask),
            "y": torch.tensor([row[c] for c in LABELS], dtype=torch.float32),
        }


def load_imagenet_backbone():
    """Full resnet50 state_dict -> load into the FULL net, then strip.

    Loading a full state_dict straight into a stripped Sequential matches
    nothing (conv1.weight vs 0.weight) and silently trains from scratch --
    which would fake a 'pretraining does not help' result.
    """
    net = resnet50(weights=None)
    sd = torch.load(WEIGHTS_ROOT / "resnet50-0676ba61.pth", map_location="cpu",
                    weights_only=False)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"ImageNet init: {len(missing)} missing, {len(unexpected)} unexpected "
          f"(expect 0/0 into the full resnet50)")
    return nn.Sequential(*list(net.children())[:-2])


class Model(nn.Module):
    def __init__(self, cross_slice: bool):
        super().__init__()
        self.cross_slice = cross_slice
        self.backbone = load_imagenet_backbone()
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(BACKBONE_DIM, D_MODEL)

        if cross_slice:
            self.slot_emb = nn.Embedding(N_SLOT_TYPES, D_MODEL)
            self.depth_proj = nn.Linear(1, D_MODEL)
            self.tok_norm = nn.LayerNorm(D_MODEL)
            layer = nn.TransformerEncoderLayer(
                D_MODEL, N_HEADS, dim_feedforward=D_MODEL * 4, dropout=DROPOUT,
                batch_first=True, norm_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, N_ATTN_LAYERS)

        self.att = nn.Sequential(nn.Linear(D_MODEL, 256), nn.Tanh(),
                                 nn.Linear(256, N_FINDINGS))
        self.heads = nn.Linear(D_MODEL, N_FINDINGS)

    def forward(self, bag, slots, depth, mask):
        B, K, C, H, W = bag.shape
        x = bag.reshape(B * K, C, H, W).repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        feats = self.pool2d(self.backbone(x)).flatten(1).reshape(B, K, -1)
        tok = self.proj(feats)

        if self.cross_slice:
            tok = tok + self.slot_emb(slots) + self.depth_proj(depth.unsqueeze(-1))
            tok = self.tok_norm(tok)
            tok = self.encoder(tok, src_key_padding_mask=(mask == 0))

        att = self.att(tok).masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        w = torch.nan_to_num(torch.softmax(att, dim=1))
        pooled = torch.einsum("bkn,bkd->bnd", w, tok)
        return (pooled * self.heads.weight.unsqueeze(0)).sum(-1) + self.heads.bias


def macro_auc(y_true, y_pred):
    per = []
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        per.append(roc_auc_score(yj, y_pred[:, j]) if len(np.unique(yj)) > 1 else np.nan)
    return float(np.nanmean(per)), per


def run_arm(name, cross_slice, train_dl, val_dl, device):
    print("\n" + "=" * 78)
    print(f"ARM {name}  (cross_slice={cross_slice})")
    print("=" * 78)
    model = Model(cross_slice=cross_slice).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    bb_ids = {id(p) for p in model.backbone.parameters()}
    opt = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": LR * BACKBONE_LR_MULT},
        {"params": [p for p in model.parameters() if id(p) not in bb_ids], "lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()

    steps_per_epoch = max(len(train_dl) // ACCUM_STEPS, 1)
    total_steps = EPOCHS * steps_per_epoch
    warmup = max(int(WARMUP_FRAC * total_steps), 1)

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(total_steps - warmup, 1)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def evaluate():
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for b in val_dl:
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    out = model(b["bag"].to(device), b["slots"].to(device),
                                b["depth"].to(device), b["mask"].to(device))
                preds.append(torch.sigmoid(out.float()).cpu().numpy())
                ys.append(b["y"].numpy())
        return macro_auc(np.concatenate(ys), np.concatenate(preds))

    t0, best, best_per, hist, gstep = time.time(), -1.0, None, [], 0
    for epoch in range(EPOCHS):
        if time.time() - t0 > TIME_BUDGET_S_PER_ARM:
            print(f"[{name}] time budget reached before epoch {epoch}")
            break
        model.train()
        tot, nb, ep_t0, last = 0.0, 0, time.time(), time.time()
        opt.zero_grad(set_to_none=True)
        for bi, b in enumerate(train_dl):
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(b["bag"].to(device), b["slots"].to(device),
                            b["depth"].to(device), b["mask"].to(device))
                y = b["y"].to(device)
                tgt = y * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
                loss = lossf(out.float(), tgt) / ACCUM_STEPS
            scaler.scale(loss).backward()
            tot += float(loss.detach()) * ACCUM_STEPS
            nb += 1
            if (bi + 1) % ACCUM_STEPS == 0:
                for gi, g in enumerate(opt.param_groups):
                    g["lr"] = base_lrs[gi] * lr_scale(gstep)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                gstep += 1
            if time.time() - last > 30:
                el = time.time() - ep_t0
                print(f"  [{name}] ep {epoch} batch {nb}/{len(train_dl)} "
                      f"loss {tot/nb:.4f} lr {opt.param_groups[-1]['lr']:.2e} "
                      f"{el:.0f}s eta {(len(train_dl)-nb)*el/nb/60:.1f}m", flush=True)
                last = time.time()
            if time.time() - t0 > TIME_BUDGET_S_PER_ARM:
                break

        macro, per = evaluate()
        star = ""
        if macro > best:
            best, best_per, star = macro, per, "  <- best"
            torch.save(model.state_dict(), WORK / f"arm7{name}_best.pt")
        hist.append({"epoch": epoch, "loss": tot / max(nb, 1), "gold_macro_auc": macro,
                     "minutes": (time.time() - ep_t0) / 60})
        print(f"[{name}] epoch {epoch}  loss {tot/max(nb,1):.4f}  "
              f"gold macro-AUC {macro:.4f}{star}  ({(time.time()-ep_t0)/60:.1f} min)",
              flush=True)

    return {"arm": name, "cross_slice": cross_slice, "best_macro_auc": best,
            "per_finding": dict(zip(LABELS, best_per)) if best_per else None,
            "history": hist}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if not SHARDS:
        raise FileNotFoundError("all-slice cache shards not attached")
    if WEIGHTS_ROOT is None:
        raise FileNotFoundError("resnet50-0676ba61.pth not attached")

    cache = CaseCache(SHARDS)
    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv")
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")
    print(f"gold_test: {len(gold)}; train_pool: {len(pool)}; "
          f"{SLICES_PER_CASE} slices/case")

    train_dl = DataLoader(CaseDataset(pool, cache, True), batch_size=BATCH_CASES,
                          shuffle=True, num_workers=2, drop_last=True)
    val_dl = DataLoader(CaseDataset(gold, cache, False), batch_size=BATCH_CASES,
                        shuffle=False, num_workers=2)

    results = []
    for name, cross in [("b", True)]:
        results.append(run_arm(name, cross, train_dl, val_dl, device))
        with open(WORK / "arm7_ab_summary.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print("ARM7 RESULTS")
    print("=" * 78)
    for r in results:
        print(f"  arm7{r['arm']} (cross_slice={r['cross_slice']}): "
              f"{r['best_macro_auc']:.4f}")
    print("\nReference: arm6 ImageNet @32 slices, arm5 RadImageNet @32 = 0.8486")
    print("  arm7a vs arm6  -> did 32->64 slices help?")
    print("  arm7b vs arm7a -> did slice interaction help, given good features?")
    if len(results) == 2 and all(r["per_finding"] for r in results):
        print(f"\n{'finding':20s} {'arm7a':>9s} {'arm7b':>9s} {'delta':>9s}")
        for lb in LABELS:
            a, b = results[0]["per_finding"][lb], results[1]["per_finding"][lb]
            print(f"{lb:20s} {a:9.4f} {b:9.4f} {b-a:+9.4f}")


if __name__ == "__main__":
    main()
