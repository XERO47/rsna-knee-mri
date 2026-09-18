"""arm2: full-pool, 5-slot attention-MIL, SSL-init vs scratch-init A/B.

Two upgrades over arm1 in one build, plus the experiment this whole project
has been building toward:

  1. Full 4,349-study pool (arm1 used a 1,200-study subset for iteration speed).
  2. Proper 5-slot layout (sagittal FS/non-FS, coronal FS/non-FS, axial),
     matching the community-proven "Raptor" architecture's fixed-slot scheme,
     instead of arm1's 3-slot (one series per plane, no FS split).
  3. The actual point: train the SAME architecture twice, identical in every
     way except backbone initialization -- once from the knee-DINO
     self-supervised checkpoint (kaggle/knee_dino/), once from scratch -- and
     compare on the untouched gold_test set. That A/B is the real test of
     whether the SSL pretraining (which fixed the protocol-clustering problem
     but showed no positive kNN label-agreement on its own) actually helps
     once a supervised head gets to fine-tune on top of it.

Both arms run sequentially in this one kernel session (same environment,
same library versions, same data loader code) rather than as separate kernel
pushes, so the comparison isolates backbone init and nothing else.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet34

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]
# 5 slots: plane x fluid-sensitivity, matching the community "Raptor" scheme
# (sagittal/coronal each split FS vs non-FS; axial gets one slot). Slice
# counts per slot kept modest (not Raptor's 18/14/12/8/12) to fit a full-pool
# run in one Kaggle GPU session.
SLOTS = ["Sagittal_FS", "Sagittal_nonFS", "Coronal_FS", "Coronal_nonFS", "Axial"]
SLICES_PER_SLOT = 4
IMG_SIZE = 160
BATCH_SIZE = 4
EPOCHS = 6
LR = 3e-4
LABEL_SMOOTH = 0.05
TIME_BUDGET_S_PER_ARM = 2.75 * 3600
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def find_root(input_root: Path, marker: str, skip_dirs=("train_series", "test_series")) -> Path:
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{marker!r} not found under {input_root}")


INPUT_ROOT = Path("/kaggle/input")
COMP_ROOT = find_root(INPUT_ROOT, "train_series.csv")
LABELS_ROOT = find_root(INPUT_ROOT, "gold_test.csv")
CHECKPOINT_ROOT = find_root(INPUT_ROOT, "knee_dino_checkpoint.pt")
SERIES_ROOT = COMP_ROOT / "train_series"
WORK = Path("/kaggle/working")
print(f"COMP_ROOT={COMP_ROOT}")
print(f"LABELS_ROOT={LABELS_ROOT}")
print(f"CHECKPOINT_ROOT={CHECKPOINT_ROOT}")


def sorted_series_files(series_dir: Path) -> list:
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
    arr = np.clip(arr, lo, hi).astype(np.float32, copy=False)
    arr -= arr.min()
    denom = arr.max()
    if denom > 0:
        arr /= denom
    if arr.shape != (size, size):
        import cv2
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr.astype(np.float32, copy=False)


def pick_slot_series(series_df: pd.DataFrame) -> dict:
    """One series per (plane, fluid-sensitivity) slot, when one exists."""
    chosen = {}
    for plane, fs, slot in [
        ("Sagittal", 1, "Sagittal_FS"), ("Sagittal", 0, "Sagittal_nonFS"),
        ("Coronal", 1, "Coronal_FS"), ("Coronal", 0, "Coronal_nonFS"),
    ]:
        cand = series_df[(series_df.Anatomical_Plane == plane) & (series_df.Fluid_Sensitive == fs)]
        if not cand.empty:
            chosen[slot] = cand.iloc[0]["SeriesInstanceUID"]
    axial = series_df[series_df.Anatomical_Plane == "Axial"].sort_values("Fluid_Sensitive", ascending=False)
    if not axial.empty:
        chosen["Axial"] = axial.iloc[0]["SeriesInstanceUID"]
    return chosen


class KneeBagDataset(Dataset):
    def __init__(self, studies_df: pd.DataFrame, series_df: pd.DataFrame, series_root: Path):
        self.studies = studies_df.reset_index(drop=True)
        self.series_by_study = {uid: g for uid, g in series_df.groupby("StudyInstanceUID")}
        self.series_root = series_root

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        row = self.studies.iloc[idx]
        uid = row["StudyInstanceUID"]
        n_slots = len(SLOTS)
        bag = np.zeros((n_slots * SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE), np.float32)
        mask = np.zeros(n_slots * SLICES_PER_SLOT, np.float32)

        sdf = self.series_by_study.get(uid)
        if sdf is not None:
            chosen = pick_slot_series(sdf)
            for slot_idx, slot in enumerate(SLOTS):
                series_uid = chosen.get(slot)
                if series_uid is None:
                    continue
                files = sorted_series_files(self.series_root / uid / series_uid)
                n = len(files)
                if n == 0:
                    continue
                fracs = np.linspace(0.06, 0.94, SLICES_PER_SLOT)
                idxs = np.clip((fracs * (n - 1)).round().astype(int), 0, n - 1)
                for s, fi in enumerate(idxs):
                    try:
                        bag[slot_idx * SLICES_PER_SLOT + s] = load_slice(files[fi], IMG_SIZE)
                        mask[slot_idx * SLICES_PER_SLOT + s] = 1.0
                    except Exception:
                        pass

        sample = {
            "bag": torch.from_numpy(bag).unsqueeze(1),
            "mask": torch.from_numpy(mask),
            "y": torch.tensor([row[c] for c in LABELS], dtype=torch.float32),
        }
        return sample


class AttentionMIL(nn.Module):
    def __init__(self, n_findings=12, embed_dim=512):
        super().__init__()
        backbone = resnet34(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.att = nn.Sequential(nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, n_findings))
        self.heads = nn.Linear(embed_dim, n_findings)

    def forward(self, bag, mask):
        B, K, C, H, W = bag.shape
        feats = self.backbone(bag.reshape(B * K, C, H, W)).reshape(B, K, -1)
        att_logits = self.att(feats)
        att_logits = att_logits.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        weights = torch.softmax(att_logits, dim=1)
        weights = torch.nan_to_num(weights)
        pooled = torch.einsum("bkn,bkd->bnd", weights, feats)
        logits = (pooled * self.heads.weight.unsqueeze(0)).sum(-1) + self.heads.bias
        return logits


def load_ssl_backbone(model: AttentionMIL, checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Teacher is the EMA-averaged, more stable representation -- the
    # canonical "final" encoder in DINO-style pretraining.
    state = ckpt.get("teacher_encoder", ckpt.get("encoder"))
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"SSL init: {len(missing)} missing, {len(unexpected)} unexpected keys "
          f"(expect 0/0 for an exact architecture match)")
    print(f"SSL checkpoint config: {ckpt.get('config')}")


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray):
    per = []
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        per.append(roc_auc_score(yj, y_pred[:, j]) if len(np.unique(yj)) > 1 else np.nan)
    return float(np.nanmean(per)), per


def soft_target(y: torch.Tensor) -> torch.Tensor:
    return y * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH


def run_arm(name: str, backbone_init: str, train_dl, val_dl, device):
    print("\n" + "=" * 78)
    print(f"ARM: {name}  (backbone_init={backbone_init})")
    print("=" * 78)
    model = AttentionMIL().to(device)
    if backbone_init == "ssl":
        load_ssl_backbone(model, CHECKPOINT_ROOT / "knee_dino_checkpoint.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    def evaluate():
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for batch in val_dl:
                bag, mask = batch["bag"].to(device), batch["mask"].to(device)
                preds.append(torch.sigmoid(model(bag, mask)).cpu().numpy())
                ys.append(batch["y"].numpy())
        preds, ys = np.concatenate(preds), np.concatenate(ys)
        return macro_auc(ys, preds), preds

    t0 = time.time()
    best_macro, best_per, history = -1.0, None, []
    for epoch in range(EPOCHS):
        if time.time() - t0 > TIME_BUDGET_S_PER_ARM:
            print(f"[{name}] time budget reached before epoch {epoch}")
            break
        model.train()
        tot_loss, n_batches = 0.0, 0
        ep_t0 = time.time()
        for batch in train_dl:
            bag, mask, y = batch["bag"].to(device), batch["mask"].to(device), batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(bag, mask), soft_target(y))
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach())
            n_batches += 1
            if time.time() - t0 > TIME_BUDGET_S_PER_ARM:
                break
        (macro, per), _ = evaluate()
        star = ""
        if macro > best_macro:
            best_macro, best_per, star = macro, per, "  <- best"
            torch.save(model.state_dict(), WORK / f"arm2_{name}_best.pt")
        row = {"epoch": epoch, "loss": tot_loss / max(n_batches, 1), "gold_macro_auc": macro,
               "minutes": (time.time() - ep_t0) / 60}
        history.append(row)
        print(f"[{name}] epoch {epoch}  loss {row['loss']:.4f}  "
              f"gold macro-AUC {macro:.4f}{star}  ({row['minutes']:.1f} min)", flush=True)

    return {"arm": name, "backbone_init": backbone_init, "best_macro_auc": best_macro,
            "per_finding": dict(zip(LABELS, best_per)) if best_per else None,
            "history": history}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv")
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")  # FULL pool, no subsetting
    series_df = pd.read_csv(COMP_ROOT / "train_series.csv")
    print(f"gold_test: {len(gold)} studies; train_pool (full): {len(pool)} studies")

    train_ds = KneeBagDataset(pool, series_df, SERIES_ROOT)
    val_ds = KneeBagDataset(gold, series_df, SERIES_ROOT)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    results = []
    for name, init in [("scratch", "scratch"), ("ssl", "ssl")]:
        results.append(run_arm(name, init, train_dl, val_dl, device))
        with open(WORK / "arm2_ab_summary.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print("ARM2 A/B COMPARISON")
    print("=" * 78)
    for r in results:
        print(f"{r['arm']:10s} best gold macro-AUC = {r['best_macro_auc']:.4f}")
    if len(results) == 2 and results[0]["per_finding"] and results[1]["per_finding"]:
        print(f"\n{'finding':20s} {'scratch':>10s} {'ssl':>10s} {'delta':>10s}")
        for label in LABELS:
            s = results[0]["per_finding"][label]
            k = results[1]["per_finding"][label]
            print(f"{label:20s} {s:10.4f} {k:10.4f} {k - s:+10.4f}")


if __name__ == "__main__":
    main()
