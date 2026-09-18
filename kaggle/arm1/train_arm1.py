"""RSNA Knee Abnormality Detection -- arm1, a first trained baseline. (v2)

Runs entirely on Kaggle infrastructure: the competition DICOM series are
mounted read-only at /kaggle/input/rsna-knee-abnormality-detection/, and the
report-derived training labels (built locally, see scripts/prepare_dataset.py
in the repo) are attached as the "rsna-knee-prelim-labels" dataset.

Problem recap
--------------
train.csv has real 0/1 labels for only 58 of 4,407 studies. Everything else
carries a free-text report and nothing more. Those 58 ("gold") studies are
held out here as the only evaluation set -- never trained on. The other
4,349 ("pool") studies train against soft pseudo-labels mined from their
reports by a community LLM pipeline (see README "Label sourcing"; validated
at 0.893 macro-AUC against gold before being trusted here).

Each study is a variable number of DICOM series (different planes: sagittal
/ coronal / axial, different sequences). This script builds a FIXED bag per
study: up to 6 evenly-sampled slices from one representative series per
plane (18 slots total), zero-padded and masked when a plane is missing, so
studies batch despite having different numbers of series and slices. Within
a plane, a fluid-sensitive series is preferred (fluid-sensitive sequences
show effusion/edema; the others show anatomy/cartilage better -- the twelve
findings split across both, per the community writeups reviewed).

A shared 2D CNN (torchvision resnet18, trained from scratch -- internet is
off, matching the competition's actual submission constraint, so no
ImageNet init here) embeds every slice. An attention layer with SEPARATE
weights per finding pools the 18 slot embeddings into one vector per
finding, because which slices matter differs by finding (an ACL tear shows
on a couple of sagittal slices; osteoarthritis is spread across many
coronal ones). Twelve independent linear heads score the twelve findings.

This is deliberately a *first* model: modest image size, a subset of the
pool for speed, trained from scratch, no test-time augmentation. It exists
to prove the pipeline end-to-end and produce a real macro-AUC number on the
gold set, not to be competitive. Scale up (full pool, more epochs, ImageNet
init via an attached weights dataset, finer per-sequence slots) once this
runs clean.
"""
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]
PLANES = ["Sagittal", "Coronal", "Axial"]
SLICES_PER_PLANE = 6
IMG_SIZE = 160
POOL_SUBSET = 1200          # studies sampled from the 4,349 pool per run -- speed lever for arm1
BATCH_SIZE = 4
EPOCHS = 6
LR = 3e-4
TIME_BUDGET_S = 3.0 * 3600  # stop early rather than get killed mid-epoch
SEED = 42

def find_root(input_root: Path, marker: str, skip_dirs=("train_series", "test_series")) -> Path:
    """Locate the directory holding `marker`, without assuming Kaggle's exact
    /kaggle/input nesting (it has changed before: flat vs competitions/ and
    datasets/<owner>/ prefixes). Skips the huge per-slice DICOM folders so a
    top-level CSV is found in milliseconds rather than after a deep walk."""
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{marker!r} not found anywhere under {input_root}")


INPUT_ROOT = Path("/kaggle/input")
COMP_ROOT = find_root(INPUT_ROOT, "train_series.csv")
LABELS_ROOT = find_root(INPUT_ROOT, "gold_test.csv")
WORK = Path("/kaggle/working")
print(f"COMP_ROOT resolved to {COMP_ROOT}")
print(f"LABELS_ROOT resolved to {LABELS_ROOT}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def pick_series_per_plane(series_df: pd.DataFrame) -> dict:
    """One series per plane, preferring fluid-sensitive when there's a choice."""
    chosen = {}
    for plane in PLANES:
        cand = series_df[series_df["Anatomical_Plane"] == plane]
        if cand.empty:
            continue
        cand = cand.sort_values("Fluid_Sensitive", ascending=False)
        chosen[plane] = cand.iloc[0]["SeriesInstanceUID"]
    return chosen


def sorted_series_files(series_dir: Path) -> list:
    """DICOM filenames are instance UIDs -- arbitrary strings, not slice order.
    Sort by InstanceNumber (header-only read, no pixel decode) so "evenly
    spaced across the stack" means spatial position, not filename order."""
    files = list(series_dir.glob("*.dcm"))
    tagged = []
    for f in files:
        try:
            n = pydicom.dcmread(str(f), stop_before_pixels=True).InstanceNumber
        except Exception:
            n = 0
        tagged.append((int(n), f))
    tagged.sort(key=lambda t: t[0])
    return [f for _, f in tagged]


def load_slice(path: Path, size: int) -> np.ndarray:
    dcm = pydicom.dcmread(str(path))
    arr = dcm.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    arr -= arr.min()
    denom = arr.max()
    if denom > 0:
        arr /= denom
    if arr.shape != (size, size):
        import cv2
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr


class KneeBagDataset(Dataset):
    """One study -> an (18, 1, H, W) bag + an 18-slot validity mask + a 12-way target."""

    def __init__(self, studies_df: pd.DataFrame, series_df: pd.DataFrame,
                 series_root: Path, targets: bool):
        self.studies = studies_df.reset_index(drop=True)
        self.series_by_study = {
            uid: g for uid, g in series_df.groupby("StudyInstanceUID")
        }
        self.series_root = series_root
        self.targets = targets

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        row = self.studies.iloc[idx]
        uid = row["StudyInstanceUID"]
        bag = np.zeros((len(PLANES) * SLICES_PER_PLANE, IMG_SIZE, IMG_SIZE), np.float32)
        mask = np.zeros(len(PLANES) * SLICES_PER_PLANE, np.float32)

        series_df = self.series_by_study.get(uid)
        if series_df is not None:
            chosen = pick_series_per_plane(series_df)
            for p_idx, plane in enumerate(PLANES):
                series_uid = chosen.get(plane)
                if series_uid is None:
                    continue
                series_dir = self.series_root / uid / series_uid
                files = sorted_series_files(series_dir)
                n = len(files)
                if n == 0:
                    continue
                # evenly across 6-94% of the stack: the outer slices carry the
                # collateral ligaments and lateral meniscus, per community notes
                fracs = np.linspace(0.06, 0.94, SLICES_PER_PLANE)
                idxs = np.clip((fracs * (n - 1)).round().astype(int), 0, n - 1)
                for slot, fi in enumerate(idxs):
                    try:
                        bag[p_idx * SLICES_PER_PLANE + slot] = load_slice(files[fi], IMG_SIZE)
                        mask[p_idx * SLICES_PER_PLANE + slot] = 1.0
                    except Exception:
                        pass  # slot stays zero/invalid; the mask keeps it out of pooling

        sample = {
            "bag": torch.from_numpy(bag).unsqueeze(1),   # (18, 1, H, W)
            "mask": torch.from_numpy(mask),              # (18,)
            "uid": uid,
        }
        if self.targets:
            sample["y"] = torch.tensor([row[c] for c in LABELS], dtype=torch.float32)
        return sample


class AttentionMIL(nn.Module):
    """Shared CNN embed -> per-finding attention pool -> 12 independent heads."""

    def __init__(self, n_findings=12, embed_dim=512):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.att = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, n_findings)
        )
        self.heads = nn.Linear(embed_dim, n_findings)

    def forward(self, bag, mask):
        B, K, C, H, W = bag.shape
        feats = self.backbone(bag.reshape(B * K, C, H, W)).reshape(B, K, -1)

        att_logits = self.att(feats)                       # (B, K, n_findings)
        att_logits = att_logits.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        weights = torch.softmax(att_logits, dim=1)          # (B, K, n_findings)
        weights = torch.nan_to_num(weights)                 # an all-masked bag -> zeros, not NaN

        pooled = torch.einsum("bkn,bkd->bnd", weights, feats)  # (B, n_findings, D)
        logits = (pooled * self.heads.weight.unsqueeze(0)).sum(-1) + self.heads.bias
        return logits


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray):
    per = []
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        if len(np.unique(yj)) > 1:
            per.append(roc_auc_score(yj, y_pred[:, j]))
        else:
            per.append(np.nan)
    return float(np.nanmean(per)), per


def main():
    print("=" * 78)
    print("arm1: loading labels and building splits")
    print("=" * 78)
    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv")
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")
    if len(pool) > POOL_SUBSET:
        pool = pool.sample(n=POOL_SUBSET, random_state=SEED).reset_index(drop=True)
    print(f"gold_test: {len(gold)} studies (held out, real labels)")
    print(f"train pool subset: {len(pool)} of {pd.read_csv(LABELS_ROOT / 'train_pool.csv').shape[0]} (arm1 speed lever)")

    series_df = pd.read_csv(COMP_ROOT / "train_series.csv")
    series_root = COMP_ROOT / "train_series"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU -- this will be slow. Check Settings -> Accelerator.")

    train_ds = KneeBagDataset(pool, series_df, series_root, targets=True)
    val_ds = KneeBagDataset(gold, series_df, series_root, targets=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = AttentionMIL().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    def evaluate():
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for batch in val_dl:
                bag = batch["bag"].to(device)
                mask = batch["mask"].to(device)
                logits = model(bag, mask)
                preds.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(batch["y"].numpy())
        preds, ys = np.concatenate(preds), np.concatenate(ys)
        macro, per = macro_auc(ys, preds)
        return macro, per, preds

    t0 = time.time()
    print("\n" + "=" * 78)
    print("training")
    print("=" * 78)
    best_macro = -1.0
    for epoch in range(EPOCHS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"time budget reached before epoch {epoch}; stopping")
            break
        model.train()
        tot_loss, n_batches = 0.0, 0
        ep_t0 = time.time()
        for batch in train_dl:
            bag = batch["bag"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(bag, mask)
            loss = lossf(logits, y)
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach())
            n_batches += 1
            if time.time() - t0 > TIME_BUDGET_S:
                break
        macro, per, _ = evaluate()
        star = ""
        if macro > best_macro:
            best_macro = macro
            star = "  <- best"
            torch.save(model.state_dict(), WORK / "arm1_best.pt")
        print(f"epoch {epoch}  loss {tot_loss / max(n_batches,1):.4f}  "
              f"gold macro-AUC {macro:.4f}{star}  ({(time.time()-ep_t0)/60:.1f} min)",
              flush=True)

    print("\n" + "=" * 78)
    print(f"ARM1 DONE -- best gold macro-AUC: {best_macro:.4f}")
    print("=" * 78)
    macro, per, preds = evaluate()
    print("final per-finding AUC on gold_test:")
    for name, a in zip(LABELS, per):
        print(f"  {name:20s} {a:.4f}" if not math.isnan(a) else f"  {name:20s} n/a")
    np.save(WORK / "arm1_gold_preds.npy", preds)
    torch.save(model.state_dict(), WORK / "arm1_final.pt")
    with open(WORK / "arm1_summary.json", "w") as f:
        json.dump({"best_macro_auc": best_macro, "final_macro_auc": macro,
                   "per_finding": dict(zip(LABELS, per)),
                   "pool_subset": POOL_SUBSET, "epochs_run": epoch + 1}, f, indent=2)


if __name__ == "__main__":
    main()
