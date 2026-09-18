"""Inference kernel for RSNA Knee Abnormality Detection.

Loads a trained arm2 checkpoint (backbone + attention-MIL head, identical
architecture to kaggle/arm2/train_arm2.py -- classes copied verbatim here
rather than imported, since Kaggle script kernels are single self-contained
files and a preprocessing/architecture mismatch between training and
inference would silently produce garbage predictions), runs it over the
competition's actual test set, and writes submission.csv.

MODEL_DATASET below must point at whichever checkpoint dataset holds the
winning arm2 result (ssl-init or scratch-init, whichever wins on gold_test)
-- update it once arm2 finishes, before pushing this kernel.

This kernel's own gold_test/test.csv run is just for sanity-checking the
code path; the actual competition score comes from Kaggle's own rerun
against the hidden test set once submitted via:
  kaggle competitions submit rsna-knee-abnormality-detection \
    -k zeroday19/rsna-knee-infer-submit -v <version> \
    -f submission.csv -m "<message>"
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
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

# arm2's A/B (2026-09-07) was inconclusive: scratch 0.6769 vs ssl 0.6702 gold
# macro-AUC, a delta inside gold-set noise, both cut short at 4/6 epochs by
# the time budget. Submitting both to the real LB to settle it -- toggle this
# between the two checkpoint filenames in zeroday19/rsna-knee-arm2-checkpoints
# and push a new kernel version per submission.
MODEL_MARKER_FILE = "arm2_ssl_best.pt"


def find_root(input_root: Path, marker: str, skip_dirs=("train_series", "test_series")) -> Path:
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{marker!r} not found under {input_root}")


INPUT_ROOT = Path("/kaggle/input")
COMP_ROOT = find_root(INPUT_ROOT, "test.csv")
MODEL_ROOT = find_root(INPUT_ROOT, MODEL_MARKER_FILE)
WORK = Path("/kaggle/working")
print(f"COMP_ROOT={COMP_ROOT}")
print(f"MODEL_ROOT={MODEL_ROOT}")


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


class TestBagDataset(Dataset):
    def __init__(self, studies_df: pd.DataFrame, series_df: pd.DataFrame, series_root: Path):
        self.studies = studies_df.reset_index(drop=True)
        self.series_by_study = {uid: g for uid, g in series_df.groupby("StudyInstanceUID")}
        self.series_root = series_root

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        uid = self.studies.iloc[idx]["StudyInstanceUID"]
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
            "uid": uid,
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    test = pd.read_csv(COMP_ROOT / "test.csv")
    series_df = pd.read_csv(COMP_ROOT / "test_series.csv")
    series_root = COMP_ROOT / "test_series"
    print(f"test studies: {len(test)}")

    model = AttentionMIL().to(device)
    state = torch.load(MODEL_ROOT / MODEL_MARKER_FILE, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    ds = TestBagDataset(test, series_df, series_root)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    uids, preds = [], []
    with torch.no_grad():
        for batch in dl:
            bag, mask = batch["bag"].to(device), batch["mask"].to(device)
            probs = torch.sigmoid(model(bag, mask)).cpu().numpy()
            preds.append(probs)
            uids.extend(batch["uid"])

    preds = np.concatenate(preds)
    sub = pd.DataFrame({"StudyInstanceUID": uids})
    for j, label in enumerate(LABELS):
        sub[label] = preds[:, j]

    # Match sample_submission.csv row order/coverage exactly.
    sample = pd.read_csv(COMP_ROOT / "sample_submission.csv")
    sub = sample[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    missing = sub[LABELS].isna().any(axis=1).sum()
    if missing:
        print(f"WARNING: {missing} studies had no prediction, filling with 0.5")
        sub[LABELS] = sub[LABELS].fillna(0.5)

    sub.to_csv(WORK / "submission.csv", index=False)
    print(f"wrote submission.csv: {sub.shape}")
    print(sub.head())


if __name__ == "__main__":
    main()
