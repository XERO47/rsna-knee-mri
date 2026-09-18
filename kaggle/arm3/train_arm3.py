"""arm3: scratch-init (tonight's A/B winner), full pool, 5-slot, resumable.

arm2's A/B (2026-09-07) gave scratch-init a small but consistent edge over
SSL-init on both gold_test (0.6769 vs 0.6702) and real LB (0.715 vs 0.706),
but both arms were cut off at only 4 epochs by a 2h45m-per-arm budget inside
one 6h Kaggle session. This arm drops the SSL comparison (answered, for now)
and gives the winning scratch-init configuration a full session's budget --
plus checkpoint/resume support, since one 6h session is a hard Kaggle cap
(confirmed via the accelerator quota API: totalTimeAllowed=21600s) and real
convergence will likely take more than one session, same constraint the
community's own "Session 1 / Session 2..." notebooks solved.

To resume: attach the previous session's output (arm3_checkpoint.pt) as a
dataset input and this picks it up automatically via find_root(); with
nothing attached it starts fresh.
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
SLOTS = ["Sagittal_FS", "Sagittal_nonFS", "Coronal_FS", "Coronal_nonFS", "Axial"]
SLICES_PER_SLOT = 4
IMG_SIZE = 160
BATCH_SIZE = 4
EPOCHS = 40  # high cap; time budget is the real stopping condition
LR = 3e-4
LABEL_SMOOTH = 0.05
TIME_BUDGET_S = 5.5 * 3600  # session cap is 21600s (6h) -- leave 30min margin
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def find_root(input_root: Path, marker: str, skip_dirs=("train_series", "test_series")) -> Path | None:
    if not input_root.exists():
        return None
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    return None


def require_root(input_root: Path, marker: str) -> Path:
    root = find_root(input_root, marker)
    if root is None:
        raise FileNotFoundError(f"{marker!r} not found under {input_root}")
    return root


INPUT_ROOT = Path("/kaggle/input")
COMP_ROOT = require_root(INPUT_ROOT, "train_series.csv")
LABELS_ROOT = require_root(INPUT_ROOT, "gold_test.csv")
RESUME_ROOT = find_root(INPUT_ROOT, "arm3_checkpoint.pt")  # None if not attached
SERIES_ROOT = COMP_ROOT / "train_series"
WORK = Path("/kaggle/working")
print(f"COMP_ROOT={COMP_ROOT}")
print(f"LABELS_ROOT={LABELS_ROOT}")
print(f"RESUME_ROOT={RESUME_ROOT}  ({'resuming' if RESUME_ROOT else 'fresh start'})")


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

        return {
            "bag": torch.from_numpy(bag).unsqueeze(1),
            "mask": torch.from_numpy(mask),
            "y": torch.tensor([row[c] for c in LABELS], dtype=torch.float32),
        }


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


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray):
    per = []
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        per.append(roc_auc_score(yj, y_pred[:, j]) if len(np.unique(yj)) > 1 else np.nan)
    return float(np.nanmean(per)), per


def soft_target(y: torch.Tensor) -> torch.Tensor:
    return y * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv")
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")
    series_df = pd.read_csv(COMP_ROOT / "train_series.csv")
    print(f"gold_test: {len(gold)} studies; train_pool (full): {len(pool)} studies")

    train_ds = KneeBagDataset(pool, series_df, SERIES_ROOT)
    val_ds = KneeBagDataset(gold, series_df, SERIES_ROOT)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = AttentionMIL().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    start_epoch, best_macro, history = 0, -1.0, []
    if RESUME_ROOT is not None:
        ckpt = torch.load(RESUME_ROOT / "arm3_checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        best_macro = ckpt["best_macro"]
        history = ckpt["history"]
        print(f"resumed at epoch {start_epoch}, best gold macro-AUC so far {best_macro:.4f}")

    def evaluate():
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for batch in val_dl:
                bag, mask = batch["bag"].to(device), batch["mask"].to(device)
                preds.append(torch.sigmoid(model(bag, mask)).cpu().numpy())
                ys.append(batch["y"].numpy())
        preds, ys = np.concatenate(preds), np.concatenate(ys)
        return macro_auc(ys, preds)

    def save_checkpoint(epoch):
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
            "best_macro": best_macro, "history": history,
        }, WORK / "arm3_checkpoint.pt")

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"time budget reached before epoch {epoch}")
            break
        model.train()
        tot_loss, n_batches = 0.0, 0
        ep_t0 = time.time()
        last_print = ep_t0
        n_train_batches = len(train_dl)
        for batch in train_dl:
            bag, mask, y = batch["bag"].to(device), batch["mask"].to(device), batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(bag, mask), soft_target(y))
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach())
            n_batches += 1
            # Time-based, not every-N-batches: keeps the same cadence whether a
            # batch takes 0.1s or 3s, so a silent multi-minute gap in the log
            # never becomes ambiguous with a genuine hang.
            if time.time() - last_print > 30:
                elapsed = time.time() - ep_t0
                eta_min = (n_train_batches - n_batches) * elapsed / n_batches / 60
                print(f"  epoch {epoch}  batch {n_batches}/{n_train_batches}  "
                      f"loss {tot_loss / n_batches:.4f}  {elapsed:.0f}s elapsed  "
                      f"eta {eta_min:.1f} min", flush=True)
                last_print = time.time()
            if time.time() - t0 > TIME_BUDGET_S:
                break
        macro, per = evaluate()
        star = ""
        if macro > best_macro:
            best_macro, star = macro, "  <- best"
            torch.save(model.state_dict(), WORK / "arm3_best.pt")
            with open(WORK / "arm3_best_per_finding.json", "w") as f:
                json.dump(dict(zip(LABELS, per)), f, indent=2)
        row = {"epoch": epoch, "loss": tot_loss / max(n_batches, 1), "gold_macro_auc": macro,
               "minutes": (time.time() - ep_t0) / 60}
        history.append(row)
        print(f"epoch {epoch}  loss {row['loss']:.4f}  gold macro-AUC {macro:.4f}{star}  "
              f"({row['minutes']:.1f} min, total {(time.time()-t0)/60:.0f} min)", flush=True)
        save_checkpoint(epoch)  # every epoch: cheap relative to an epoch's cost, and this
                                 # is exactly the artifact the next session resumes from

    with open(WORK / "arm3_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDONE this session. best gold macro-AUC so far: {best_macro:.4f} "
          f"over {len(history)} total epochs.")
    print("If not converged, pull arm3_checkpoint.pt, re-upload it, attach to "
          "the next session, and re-run -- it resumes automatically.")


if __name__ == "__main__":
    main()
