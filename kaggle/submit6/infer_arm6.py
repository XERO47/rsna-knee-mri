"""Inference for arm6 (ImageNet ResNet-50, 32 slices) -> submission.csv.

THE TRAP THIS FILE EXISTS TO AVOID: the 20 GB all-slice cache covers
train_series only. The test set has no cache, so inference must decode DICOM
live -- and must reproduce the cache's construction *exactly*, or the model
sees inputs unlike anything it trained on and the score silently collapses.

Everything below mirrors kaggle/prep_v2/build_all_slice_cache.py step for step:

  1. Series order  = the order rows appear in test_series.csv for that study
                     (the cache used `sdf.iterrows()` over the same grouping).
  2. Slice order   = InstanceNumber, header-only read. NOT filename order --
                     filenames are instance UIDs and sort arbitrarily.
  3. Preprocessing = percentile[1,99] clip -> min-max to [0,1] -> cv2 resize
                     160 INTER_AREA.
  4. uint8 round-trip. The cache stored uint8(arr*255) and training read it
     back as float/255. Training inputs therefore passed through uint8
     quantisation, so inference quantises too. Skipping this would feed the
     model slightly different pixel values than it ever saw.
  5. 320-slice cap applied before selection, matching MAX_SLICES_PER_CASE.
  6. Eval-time selection = np.linspace(0, n-1, 32) over the CONCATENATED
     ordered slice list, matching CaseDataset's non-train branch.

Speed: we only need 32 slices, so full pixel decode happens 32 times per case.
Ordering only needs InstanceNumber, which is a header-only read
(stop_before_pixels=True) -- cheap. That keeps the hidden-test rerun fast
even though a case averages ~177 slices.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]
N_FINDINGS = 12
BACKBONE_DIM = 2048
SLICES_PER_CASE = 32          # must match arm6 training
IMG_SIZE = 160                # must match arm6 training
MAX_SLICES_PER_CASE = 320     # must match the cache builder
BATCH_CASES = 2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find_root(input_root: Path, marker: str, skip_dirs=("train_series", "test_series")):
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{marker!r} not found under {input_root}")


INPUT_ROOT = Path("/kaggle/input")
COMP_ROOT = find_root(INPUT_ROOT, "test.csv")
MODEL_ROOT = find_root(INPUT_ROOT, "arm6_best.pt")
WORK = Path("/kaggle/working")
print(f"COMP_ROOT={COMP_ROOT}")
print(f"MODEL_ROOT={MODEL_ROOT}")


def ordered_files(series_dir: Path):
    """InstanceNumber order, header-only reads (no pixel decode)."""
    tagged = []
    for f in series_dir.glob("*.dcm"):
        try:
            n = int(pydicom.dcmread(str(f), stop_before_pixels=True).InstanceNumber)
        except Exception:
            n = 0
        tagged.append((n, f))
    tagged.sort(key=lambda t: t[0])
    return [f for _, f in tagged]


def load_slice(path: Path) -> np.ndarray:
    """Identical to the cache builder, including the uint8 round-trip."""
    dcm = pydicom.dcmread(str(path))
    arr = dcm.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi).astype(np.float32, copy=False)
    arr -= arr.min()
    denom = arr.max()
    if denom > 0:
        arr /= denom
    if arr.shape != (IMG_SIZE, IMG_SIZE):
        import cv2
        arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    q = np.clip(arr * 255.0, 0, 255).astype(np.uint8)     # cache quantisation
    return q.astype(np.float32) / 255.0                    # training read-back


class TestDataset(Dataset):
    def __init__(self, studies_df, series_df, series_root):
        self.studies = studies_df.reset_index(drop=True)
        self.series_df = series_df
        self.series_root = series_root

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        uid = self.studies.iloc[i]["StudyInstanceUID"]
        sdf = self.series_df[self.series_df.StudyInstanceUID == uid]

        # Build the ordered file list exactly as the cache concatenated it.
        all_files = []
        for _, srow in sdf.iterrows():
            all_files.extend(ordered_files(self.series_root / uid /
                                           srow["SeriesInstanceUID"]))
        n = len(all_files)

        bag = np.zeros((SLICES_PER_CASE, IMG_SIZE, IMG_SIZE), np.float32)
        mask = np.zeros(SLICES_PER_CASE, np.float32)
        if n == 0:
            print(f"WARNING: no slices for {uid}")
            return {"bag": torch.from_numpy(bag).unsqueeze(1),
                    "mask": torch.from_numpy(mask), "uid": uid}

        # cache-side 320 cap, then eval-side even selection
        if n > MAX_SLICES_PER_CASE:
            keep = np.linspace(0, n - 1, MAX_SLICES_PER_CASE).round().astype(int)
            all_files = [all_files[k] for k in keep]
            n = len(all_files)

        if n >= SLICES_PER_CASE:
            pick = np.linspace(0, n - 1, SLICES_PER_CASE).round().astype(int)
        else:
            pick = np.concatenate([np.arange(n),
                                   np.full(SLICES_PER_CASE - n, n - 1)])

        for s, fi in enumerate(pick):
            try:
                bag[s] = load_slice(all_files[fi])
                mask[s] = 1.0
            except Exception as e:
                print(f"slice error {uid} idx{fi}: {e}")
        if n < SLICES_PER_CASE:
            mask[n:] = 0.0

        return {"bag": torch.from_numpy(bag).unsqueeze(1),
                "mask": torch.from_numpy(mask), "uid": uid}


class AttentionMIL(nn.Module):
    """Must match arm6's module layout exactly so state_dict keys line up."""

    def __init__(self, n_findings=N_FINDINGS, embed_dim=BACKBONE_DIM):
        super().__init__()
        net = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(nn.Linear(embed_dim, 256), nn.Tanh(),
                                 nn.Linear(256, n_findings))
        self.heads = nn.Linear(embed_dim, n_findings)

    def forward(self, bag, mask):
        B, K, C, H, W = bag.shape
        x = bag.reshape(B * K, C, H, W).repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        feats = self.pool(self.backbone(x)).flatten(1).reshape(B, K, -1)
        att = self.att(feats).masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        w = torch.nan_to_num(torch.softmax(att, dim=1))
        pooled = torch.einsum("bkn,bkd->bnd", w, feats)
        return (pooled * self.heads.weight.unsqueeze(0)).sum(-1) + self.heads.bias


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    test = pd.read_csv(COMP_ROOT / "test.csv")
    series_df = pd.read_csv(COMP_ROOT / "test_series.csv")
    print(f"test studies: {len(test)}, test series rows: {len(series_df)}")

    model = AttentionMIL().to(device)
    state = torch.load(MODEL_ROOT / "arm6_best.pt", map_location=device,
                       weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"checkpoint load: {len(missing)} missing, {len(unexpected)} unexpected "
          f"(expect 0/0 -- anything else means the architecture drifted)")
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: {missing[:5]} {unexpected[:5]}")
    model.eval()

    dl = DataLoader(TestDataset(test, series_df, COMP_ROOT / "test_series"),
                    batch_size=BATCH_CASES, shuffle=False, num_workers=2)

    uids, preds = [], []
    with torch.no_grad():
        for bi, b in enumerate(dl):
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(b["bag"].to(device), b["mask"].to(device))
            preds.append(torch.sigmoid(out.float()).cpu().numpy())
            uids.extend(b["uid"])
            if (bi + 1) % 25 == 0:
                print(f"  {len(uids)}/{len(test)} cases", flush=True)

    preds = np.concatenate(preds)
    sub = pd.DataFrame({"StudyInstanceUID": uids})
    for j, lb in enumerate(LABELS):
        sub[lb] = preds[:, j]

    sample = pd.read_csv(COMP_ROOT / "sample_submission.csv")
    sub = sample[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    n_missing = int(sub[LABELS].isna().any(axis=1).sum())
    if n_missing:
        print(f"WARNING: {n_missing} studies had no prediction, filling 0.5")
        sub[LABELS] = sub[LABELS].fillna(0.5)

    sub.to_csv(WORK / "submission.csv", index=False)
    print(f"wrote submission.csv {sub.shape}")
    print(sub.head())


if __name__ == "__main__":
    main()
