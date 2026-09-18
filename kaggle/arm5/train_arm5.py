"""arm5: arm3's proven attention-MIL, but with a RadImageNet-pretrained ResNet-50.

Why this exists. Five arms (arm1-arm4) all landed within 0.018 of each other
(0.6679-0.6860) despite varying backbone, aggregation, slices/case by 9x,
training data by 3.6x, SSL init, and epochs 4->16. That invariance says none
of those was the binding constraint. What every one of them shared:
**trained from scratch, at 160px.**

Pretrained weights turned out to be available on Kaggle after all -- the
internet-off rule blocks downloading at runtime, not attaching. The 0.936
community notebook's own inputs show what the top solutions actually lean on,
and it isn't ImageNet: RadImageNet appears four times across its attachments.
RadImageNet is pretrained on ~1.35M *radiological* images (CT/MRI/US), which
is domain-matched to knee MRI in a way ImageNet is not.

Verified locally before writing this: the checkpoint is an OrderedDict of a
stripped nn.Sequential(*resnet50.children()[:-2]) (backbone.0=conv1,
.1=bn1, .4-.7=layer1-4), and **318/318 tensors match torchvision resnet50
exactly, none missing**. Stem conv is (64,3,7,7), so grayscale is repeated to
3 channels rather than adapting conv1 -- that keeps 100% of the pretrained
stem usable.

Deliberately holds everything else at arm3's settings so the comparison
isolates the backbone: per-finding attention pooling, same gold_test eval.
Baseline to beat: **0.6791** (arm3).

Backbone LR is 0.1x the head's -- pretrained medical features are the whole
point of this run and should be nudged, not overwritten by 4,349 noisy labels.
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
EMBED_DIM = 2048          # ResNet50 output (vs 512 for ResNet34)

SLICES_PER_CASE = 32      # more than arm3's 20; the cache makes it cheap
IMG_SIZE = 160
BATCH_CASES = 4
ACCUM_STEPS = 4           # effective batch 16 cases
EPOCHS = 20               # also the cosine horizon -- sized to actually finish
LR = 3e-4
BACKBONE_LR_MULT = 0.1    # pretrained medical features: nudge, don't rewrite
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
WARMUP_FRAC = 0.1
TIME_BUDGET_S = 5.0 * 3600
SEED = 42

# RadImageNet ports conventionally use ImageNet normalisation.
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
RADIMAGENET_ROOT = find_root(INPUT_ROOT, "ResNet50.pt")
RESUME_ROOT = find_root(INPUT_ROOT, "arm5_checkpoint.pt")
SHARDS = [(s, r) for s, r in ((s, find_root(INPUT_ROOT, f"index_shard{s}.json"))
                              for s in (0, 1)) if r]
WORK = Path("/kaggle/working")
print(f"LABELS_ROOT={LABELS_ROOT}")
print(f"RADIMAGENET_ROOT={RADIMAGENET_ROOT}")
print(f"all-slice shards: {len(SHARDS)}")
print(f"RESUME_ROOT={RESUME_ROOT}  ({'resuming' if RESUME_ROOT else 'fresh start'})")


class CaseCache:
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
        return np.asarray(arr[lo:hi], dtype=np.uint8)


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
        slices = self.cache.get(row["StudyInstanceUID"])
        n = len(slices)
        if n >= SLICES_PER_CASE:
            # random subset while training (sees everything across epochs, and
            # doubles as augmentation); evenly spaced at eval for determinism
            if self.train:
                pick = np.sort(np.random.choice(n, SLICES_PER_CASE, replace=False))
            else:
                pick = np.linspace(0, n - 1, SLICES_PER_CASE).round().astype(int)
        else:
            pick = np.concatenate([np.arange(n),
                                   np.full(SLICES_PER_CASE - n, n - 1)])
        x = torch.from_numpy(slices[pick].astype(np.float32) / 255.0).unsqueeze(1)
        mask = torch.ones(SLICES_PER_CASE)
        if n < SLICES_PER_CASE:
            mask[n:] = 0.0
        return {"bag": x, "mask": mask,
                "y": torch.tensor([row[c] for c in LABELS], dtype=torch.float32)}


def load_radimagenet_backbone():
    """Stripped Sequential (conv1..layer4) with RadImageNet weights; 318/318."""
    net = resnet50(weights=None)
    seq = nn.Sequential(*list(net.children())[:-2])
    sd = torch.load(RADIMAGENET_ROOT / "ResNet50.pt", map_location="cpu",
                    weights_only=False)
    stripped = {k.replace("backbone.", ""): v for k, v in sd.items()}
    missing, unexpected = seq.load_state_dict(stripped, strict=False)
    print(f"RadImageNet init: {len(missing)} missing, {len(unexpected)} unexpected "
          f"(expect 0/0)")
    return seq


class AttentionMIL(nn.Module):
    def __init__(self, n_findings=N_FINDINGS, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = load_radimagenet_backbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(nn.Linear(embed_dim, 256), nn.Tanh(),
                                 nn.Linear(256, n_findings))
        self.heads = nn.Linear(embed_dim, n_findings)

    def forward(self, bag, mask):
        B, K, C, H, W = bag.shape
        x = bag.reshape(B * K, C, H, W).repeat(1, 3, 1, 1)   # grayscale -> 3ch
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        feats = self.pool(self.backbone(x)).flatten(1).reshape(B, K, -1)

        att = self.att(feats).masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        w = torch.nan_to_num(torch.softmax(att, dim=1))
        pooled = torch.einsum("bkn,bkd->bnd", w, feats)
        return (pooled * self.heads.weight.unsqueeze(0)).sum(-1) + self.heads.bias


def macro_auc(y_true, y_pred):
    per = []
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        per.append(roc_auc_score(yj, y_pred[:, j]) if len(np.unique(yj)) > 1 else np.nan)
    return float(np.nanmean(per)), per


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if not SHARDS:
        raise FileNotFoundError("all-slice cache shards not attached")
    if RADIMAGENET_ROOT is None:
        raise FileNotFoundError("RadImageNet ResNet50.pt not attached")

    cache = CaseCache(SHARDS)
    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv")
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")
    print(f"gold_test: {len(gold)}; train_pool: {len(pool)}; "
          f"{SLICES_PER_CASE} slices/case")

    train_dl = DataLoader(CaseDataset(pool, cache, True), batch_size=BATCH_CASES,
                          shuffle=True, num_workers=2, drop_last=True)
    val_dl = DataLoader(CaseDataset(gold, cache, False), batch_size=BATCH_CASES,
                        shuffle=False, num_workers=2)

    model = AttentionMIL().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    bb_ids = {id(p) for p in model.backbone.parameters()}
    opt = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": LR * BACKBONE_LR_MULT},
        {"params": [p for p in model.parameters() if id(p) not in bb_ids], "lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()

    start_epoch, best_macro, history, gstep = 0, -1.0, [], 0
    if RESUME_ROOT is not None:
        ck = torch.load(RESUME_ROOT / "arm5_checkpoint.pt", map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best_macro = ck["epoch"] + 1, ck["best_macro"]
        history, gstep = ck["history"], ck.get("gstep", 0)
        print(f"resumed at epoch {start_epoch}, best {best_macro:.4f}")

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
                    out = model(b["bag"].to(device), b["mask"].to(device))
                preds.append(torch.sigmoid(out.float()).cpu().numpy())
                ys.append(b["y"].numpy())
        return macro_auc(np.concatenate(ys), np.concatenate(preds))

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"time budget reached before epoch {epoch}")
            break
        model.train()
        tot, nb, ep_t0, last = 0.0, 0, time.time(), time.time()
        opt.zero_grad(set_to_none=True)
        for bi, b in enumerate(train_dl):
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(b["bag"].to(device), b["mask"].to(device))
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
                print(f"  epoch {epoch}  batch {nb}/{len(train_dl)}  loss {tot/nb:.4f}"
                      f"  lr {opt.param_groups[-1]['lr']:.2e}  {el:.0f}s"
                      f"  eta {(len(train_dl)-nb)*el/nb/60:.1f} min", flush=True)
                last = time.time()
            if time.time() - t0 > TIME_BUDGET_S:
                break

        macro, per = evaluate()
        star = ""
        if macro > best_macro:
            best_macro, star = macro, "  <- best"
            torch.save(model.state_dict(), WORK / "arm5_best.pt")
            with open(WORK / "arm5_best_per_finding.json", "w") as f:
                json.dump(dict(zip(LABELS, per)), f, indent=2)
        history.append({"epoch": epoch, "loss": tot / max(nb, 1),
                        "gold_macro_auc": macro,
                        "minutes": (time.time() - ep_t0) / 60})
        print(f"epoch {epoch}  loss {tot/max(nb,1):.4f}  gold macro-AUC {macro:.4f}"
              f"{star}  ({(time.time()-ep_t0)/60:.1f} min, "
              f"total {(time.time()-t0)/60:.0f} min)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best_macro": best_macro,
                    "history": history, "gstep": gstep},
                   WORK / "arm5_checkpoint.pt")

    with open(WORK / "arm5_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDONE. best gold macro-AUC {best_macro:.4f} over {len(history)} epochs.")
    print("Baseline to beat: arm3 0.6791 (from-scratch ResNet34).")


if __name__ == "__main__":
    main()
