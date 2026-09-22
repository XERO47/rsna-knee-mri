"""Standalone Kaggle trainer for Arm 16 (Arm 13 + local ROI residuals).

Run this entire file directly as a notebook cell (copy/paste its contents, or
`%run` it). All configuration lives in the CONFIGURATION block below as plain
variables — edit them there instead of setting environment variables.

The script needs the cache64 files (vols/ids/masks shard 0 and 1) and the
validated gold_test.csv/train_pool.csv label files in attached Kaggle inputs.
The high-probability recipe requires every ROI bag. Attach either the original shard files named
``roi_<slug>_shard{0,1}.npy`` with matching masks and IDs, or the merged
dataset files ``roi_<slug>.npy``, ``roi_<slug>_masks.npy``, and ``roi_ids.npy``.
Each ROI array is uint8 [N,4,3,336,336], each mask is binary uint8 [N,4].
Merged IDs must exactly equal global shard 0 IDs followed by shard 1 IDs.

The default recipe deliberately requires an Arm 13 CoAtNet-384 checkpoint.  It
first trains only the new ROI heads, then jointly fine-tunes the global path at
much smaller learning rates.  Batch-size two plus gradient accumulation and
two sampled ROI windows per target keep the shared CoAtNet within Kaggle GPU
memory; evaluation always uses every valid ROI window.

W&B logs the warm-start baseline, global-only and combined metrics, per-target
deltas, gradient/LR diagnostics, and uploads complete resumable checkpoints.
The best validation checkpoint and an uninterrupted-run top-three weight
average are also evaluated with all 62 global windows before final selection.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


# ------------------------------ configuration ------------------------------
LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)
ROI_TARGETS = ("MCL", "Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis")
ROI_INDEX = {name: LABELS.index(name) for name in ROI_TARGETS}
RES = 384
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
PRETRAINED = False
EPOCHS = 10
K_TRAIN = 12
K_EVAL = 24
K_FINAL = 62
ROI_K_TRAIN = 2
BATCH_SIZE = 2
ACCUM_STEPS = 2
WORKERS = 2
BACKBONE_LR = 1e-5
GLOBAL_HEAD_LR = 3e-5
ROI_HEAD_LR = 3e-4
FREEZE_GLOBAL_EPOCHS = 2
GLOBAL_LOSS_WEIGHT = 0.25
WEIGHT_DECAY = 0.05
CLIP_NORM = 3.0
SEED = 42
OUTPUT_DIR = Path("/kaggle/working/arm16")
RESUME = ""
INIT_CHECKPOINT = "arm13_best.pt"  # arm13's SWA average step OOM'd; the best epoch checkpoint is the usable SWA-equivalent
ALLOW_SCRATCH = False
REQUIRE_ROI = True
TOP_K = 3
UPLOAD_EVERY = 2
WANDB_PROJECT = "rsna-knee-arm16"
WANDB_ENTITY = "rsna-knee-cap-abnormality" or None
WANDB_RUN_NAME = "arm16-arm13-roi-seed42"
WANDB_API_KEY = ""  # set to authenticate non-interactively; leave blank for anonymous login
WANDB_MODE = ""  # e.g. "offline" or "disabled"; leave blank for normal online logging
ARM16_INPUT = "/kaggle/input"  # root directory to search for attached datasets
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@lru_cache(maxsize=None)
def find_file(name: str) -> Path | None:
    roots = [Path(ARM16_INPUT), Path("/kaggle/working")]
    for root in roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.exists():
            return direct
        for directory, subdirectories, filenames in os.walk(root):
            subdirectories[:] = [
                item
                for item in subdirectories
                if item not in {"train_series", "test_series", ".git", "__pycache__"}
            ]
            if name in filenames:
                return Path(directory) / name
    return None


def require_file(name: str) -> Path:
    path = find_file(name)
    if path is None:
        raise FileNotFoundError(
            f"Could not find {name}. Attach the cache64/label Kaggle dataset or set "
            "ARM16_INPUT variable to its mounted directory."
        )
    return path


class Corpus:
    def __init__(self):
        self.a = np.load(require_file("vols_shard0.npy"), mmap_mode="r")
        self.b = np.load(require_file("vols_shard1.npy"), mmap_mode="r")
        self.depth = int(self.a.shape[1])
        if self.a.ndim != 4 or self.b.ndim != 4 or self.a.shape[1:] != self.b.shape[1:]:
            raise ValueError(f"invalid cache64 shapes: {self.a.shape}, {self.b.shape}")

    def study(self, row: int, indices: list[int]) -> np.ndarray:
        source = self.a if row < len(self.a) else self.b
        offset = row if row < len(self.a) else row - len(self.a)
        return np.asarray(source[offset, indices])


def load_index() -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    shard_ids = (
        np.load(require_file("ids_shard0.npy"), allow_pickle=True).astype(str),
        np.load(require_file("ids_shard1.npy"), allow_pickle=True).astype(str),
    )
    ids = np.concatenate(shard_ids)
    masks = np.concatenate([
        np.load(require_file("masks_shard0.npy")),
        np.load(require_file("masks_shard1.npy")),
    ])
    if len(ids) != len(masks) or len(ids) != len(set(ids)):
        raise ValueError("cache64 ids/masks are missing, duplicated, or misaligned")
    return ids, masks, shard_ids


def load_labels(ids: np.ndarray) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    gold_path, pool_path = require_file("gold_test.csv"), require_file("train_pool.csv")
    gold, pool = pd.read_csv(gold_path), pd.read_csv(pool_path)
    required_columns = {"StudyInstanceUID", *LABELS}
    for name, frame in (("gold_test.csv", gold), ("train_pool.csv", pool)):
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{name} is missing columns: {missing_columns}")
        if frame.StudyInstanceUID.astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate StudyInstanceUID rows")
    idset = set(ids.astype(str))
    gold_ids = [str(x) for x in gold.StudyInstanceUID if str(x) in idset]
    if len(gold_ids) != len(gold):
        raise ValueError(
            f"only {len(gold_ids)}/{len(gold)} Gold studies exist in cache64"
        )
    gold_set = set(gold.StudyInstanceUID.astype(str))
    train_ids = [
        str(x)
        for x in pool.StudyInstanceUID
        if str(x) in idset and str(x) not in gold_set
    ]
    labels: dict[str, np.ndarray] = {}
    for frame in (pool, gold):
        for uid, values in zip(
            frame.StudyInstanceUID.astype(str),
            frame[list(LABELS)].to_numpy(np.float32),
        ):
            labels[str(uid)] = np.asarray(values, dtype=np.float32)
    missing = [
        uid
        for uid in train_ids + gold_ids
        if uid not in labels or not np.isfinite(labels[uid]).all()
    ]
    if missing:
        raise ValueError(f"missing/invalid labels for {len(missing)} studies, e.g. {missing[0]}")
    if not train_ids:
        raise ValueError("train_pool.csv did not produce any non-Gold training studies")
    return train_ids, gold_ids, labels


class OptionalROICache:
    """Loads optional per-target ROI arrays, preserving cache row order."""

    def __init__(self, shard_ids: tuple[np.ndarray, np.ndarray]):
        self.n0 = len(shard_ids[0])
        self.merged = find_file("roi_ids.npy") is not None or any(
            find_file(f"roi_{slug(target)}.npy") is not None
            for target in ROI_TARGETS
        )
        self.data: dict[str, tuple[object, object]] = {}
        any_roi_data = any(
            find_file(f"roi_{slug(target)}_shard{shard}.npy") is not None
            for target in ROI_TARGETS
            for shard in (0, 1)
        ) or self.merged
        if any_roi_data or REQUIRE_ROI:
            if self.merged:
                roi_ids_path = find_file("roi_ids.npy")
                if roi_ids_path is None:
                    raise FileNotFoundError("merged ROI data is present but roi_ids.npy is missing")
                expected_ids = np.concatenate(shard_ids)
                roi_ids = np.load(roi_ids_path, allow_pickle=True).astype(str)
                if not np.array_equal(roi_ids, expected_ids):
                    raise ValueError(
                        "roi_ids.npy does not exactly match global shard 0 IDs followed "
                        "by shard 1 IDs; refusing a potentially misaligned cache"
                    )
            else:
                for shard in (0, 1):
                    roi_ids_path = find_file(f"roi_ids_shard{shard}.npy")
                    if roi_ids_path is None:
                        raise FileNotFoundError(
                            f"ROI data is present but roi_ids_shard{shard}.npy is missing"
                        )
                    roi_ids = np.load(roi_ids_path, allow_pickle=True).astype(str)
                    if not np.array_equal(roi_ids, shard_ids[shard]):
                        raise ValueError(
                            f"roi_ids_shard{shard}.npy does not exactly match "
                            f"ids_shard{shard}.npy; refusing a potentially misaligned cache"
                        )
        for target in ROI_TARGETS:
            s = slug(target)
            if self.merged:
                data_path = find_file(f"roi_{s}.npy")
                mask_path = find_file(f"roi_{s}_masks.npy")
                if data_path is None or mask_path is None:
                    if REQUIRE_ROI:
                        raise FileNotFoundError(
                            f"merged ROI cache is required for {target}; missing "
                            f"roi_{s}.npy or roi_{s}_masks.npy"
                        )
                    print(f"[Arm16] merged ROI cache not found for {target}; global-only logits.")
                    continue
                array = np.load(data_path, mmap_mode="r")
                mask = np.load(mask_path, mmap_mode="r")
                if array.shape != (self.n0 + len(shard_ids[1]), 4, 3, 336, 336) or array.dtype != np.uint8:
                    raise ValueError(f"merged ROI {target} must be uint8 [N,4,3,336,336]")
                if mask.shape != (len(array), 4) or mask.dtype != np.uint8 or not np.isin(np.unique(mask), [0, 1]).all():
                    raise ValueError(f"merged ROI {target} mask must be binary uint8 [N,4]")
                self.data[target] = (array, mask.astype(bool))
                continue
            files = [find_file(f"roi_{s}_shard{i}.npy") for i in (0, 1)]
            mask_files = [find_file(f"roi_{s}_masks_shard{i}.npy") for i in (0, 1)]
            if any(x is None for x in files + mask_files):
                if REQUIRE_ROI:
                    expected = [
                        f"roi_{s}_shard{i}.npy" for i in (0, 1)
                    ] + [
                        f"roi_{s}_masks_shard{i}.npy" for i in (0, 1)
                    ]
                    missing = [
                        name
                        for name, path in zip(expected, files + mask_files)
                        if path is None
                    ]
                    raise FileNotFoundError(
                        f"ROI cache is required for {target}; missing {missing}"
                    )
                print(
                    f"[Arm16] ROI cache not found for {target}; "
                    "this target uses global-only logits."
                )
                continue
            arrays = [np.load(x, mmap_mode="r") for x in files if x is not None]
            masks = [np.load(x, mmap_mode="r") for x in mask_files if x is not None]
            row_counts_match = (
                arrays[0].shape[0] == len(shard_ids[0])
                and arrays[1].shape[0] == len(shard_ids[1])
            )
            if not row_counts_match:
                raise ValueError(f"ROI {target} row count does not match cache64 ids")
            expected_tail = (4, 3, 336, 336)
            invalid_array = any(
                array.shape[1:] != expected_tail or array.dtype != np.uint8
                for array in arrays
            )
            if invalid_array:
                raise ValueError(
                    f"ROI {target} must be uint8 [N,4,3,336,336]; "
                    f"received {[array.shape for array in arrays]}"
                )
            if any(
                mask.shape != (len(shard_ids[index]), 4)
                or mask.dtype != np.uint8
                or not np.isin(np.unique(mask), [0, 1]).all()
                for index, mask in enumerate(masks)
            ):
                raise ValueError(f"ROI {target} masks must be binary uint8 [N,4]")
            self.data[target] = (
                [arrays[0], arrays[1]],
                [masks[0].astype(bool), masks[1].astype(bool)],
            )

    def get(self, target: str, row: int) -> tuple[np.ndarray, np.ndarray] | None:
        if target not in self.data:
            return None
        arrays, masks = self.data[target]
        if self.merged:
            return np.asarray(arrays[row]), np.asarray(masks[row])
        shard = 0 if row < self.n0 else 1
        offset = row if shard == 0 else row - self.n0
        return np.asarray(arrays[shard][offset]), np.asarray(masks[shard][offset])


class StudyDataset(Dataset):
    def __init__(self, uids, id2row, masks, labels, k, train, roi_cache):
        self.uids, self.id2row, self.masks = list(uids), id2row, masks
        self.labels, self.k, self.train, self.roi_cache = labels, k, train, roi_cache
        self.corpus: Corpus | None = None

    def __len__(self):
        return len(self.uids)

    def _centers(self, row: int) -> list[int]:
        valid = np.flatnonzero(self.masks[row] > 0)
        if len(valid) < 3:
            valid = np.arange(min(3, self.masks.shape[1]))
        lo, hi = int(valid.min()), int(valid.max())
        centers = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, self.masks.shape[1] - 2))]
        if self.train:
            pool = centers * (self.k // len(centers) + 1)
            random.shuffle(pool)
            return pool[:self.k]
        return [centers[i] for i in np.linspace(0, len(centers) - 1, self.k).round().astype(int)]

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != RES:
            x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
        return (x - IMAGENET_MEAN) / IMAGENET_STD

    def _select_roi_windows(
        self,
        array: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample real ROI windows without duplicating anatomy.

        Training uses a fixed-size subset for predictable GPU memory.  If a
        study has fewer valid windows than requested, unused slots remain
        masked rather than repeating a slice.  Evaluation retains all four.
        """

        if not self.train or ROI_K_TRAIN >= len(mask):
            return array, mask
        valid = np.flatnonzero(mask)
        invalid = np.flatnonzero(~mask)
        if len(valid) > ROI_K_TRAIN:
            chosen = np.asarray(
                random.sample(valid.tolist(), ROI_K_TRAIN),
                dtype=np.int64,
            )
        else:
            needed = ROI_K_TRAIN - len(valid)
            chosen = np.concatenate([valid, invalid[:needed]]).astype(np.int64)
        return array[chosen], mask[chosen]

    def __getitem__(self, index):
        if self.corpus is None:
            self.corpus = Corpus()
        uid = self.uids[index]
        row = self.id2row[uid]
        centers = [max(1, min(c, self.corpus.depth - 2)) for c in self._centers(row)]
        needed = sorted({s for c in centers for s in (c - 1, c, c + 1)})
        position = {s: j for j, s in enumerate(needed)}
        stack = self.corpus.study(row, needed)
        triplets = np.stack(
            [stack[[position[c - 1], position[c], position[c + 1]]] for c in centers]
        )
        global_x = self._normalize(torch.from_numpy(triplets).float().div_(255.0))
        global_mask = torch.ones(len(centers), dtype=torch.bool)
        roi_x, roi_mask = {}, {}
        for target in ROI_TARGETS:
            item = self.roi_cache.get(target, row)
            if item is None:
                continue
            arr, mask = item
            arr, mask = self._select_roi_windows(arr, mask)
            if arr.shape[1] == 1:
                arr = np.repeat(arr, 3, axis=1)
            x = self._normalize(torch.from_numpy(arr).float().div_(255.0))
            roi_x[target], roi_mask[target] = x, torch.from_numpy(mask.astype(bool))
        return global_x, global_mask, roi_x, roi_mask, torch.from_numpy(self.labels[uid])


def collate(batch):
    global_x = torch.stack([x[0] for x in batch])
    global_mask = torch.stack([x[1] for x in batch])
    roi_x, roi_mask = {}, {}
    for target in ROI_TARGETS:
        if all(target in x[2] for x in batch):
            roi_x[target] = torch.stack([x[2][target] for x in batch])
            roi_mask[target] = torch.stack([x[3][target] for x in batch])
    labels = torch.stack([x[4] for x in batch])
    return global_x, global_mask, roi_x, roi_mask, labels


# ---------------------------------- model ----------------------------------
def build_backbone():
    if ARCH == "tiny_test":
        net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        net.num_features = 16
        return net
    import timm
    net = timm.create_model(
        ARCH,
        pretrained=PRETRAINED,
        num_classes=0,
        in_chans=3,
        global_pool="avg",
    )
    if hasattr(net, "set_grad_checkpointing"):
        net.set_grad_checkpointing(True)
    return net


def masked_softmax(scores, mask):
    mask = mask.to(device=scores.device, dtype=torch.bool)
    masked_scores = scores.float().masked_fill(
        ~mask,
        torch.finfo(torch.float32).min,
    )
    weights = torch.softmax(masked_scores, dim=1).to(scores.dtype)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(1, keepdim=True).clamp_min(1e-12)


class QueryMIL(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsW, self.clsb = nn.Parameter(torch.zeros(12, f)), nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(self, h, mask):
        a = masked_softmax(self.att(h), mask.unsqueeze(-1))
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


class ROIPooler(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.norm = nn.LayerNorm(f)
        self.value, self.gate = nn.Linear(f, 256), nn.Linear(f, 256)
        self.score, self.readout = nn.Linear(256, 1), nn.Linear(f, 1)
        # The warm-started model must initially be exactly Arm 13.  A zero
        # readout makes every new residual zero while still allowing the ROI
        # head to begin learning on its first optimization step.
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, h, mask):
        mask = mask.to(device=h.device, dtype=torch.bool)
        h = self.norm(h)
        scores = self.score(torch.tanh(self.value(h)) * torch.sigmoid(self.gate(h))).squeeze(-1)
        a = masked_softmax(scores, mask)
        residual = self.readout(torch.einsum("bk,bkf->bf", a, h)).squeeze(-1)
        return residual * mask.any(1).to(residual.dtype)


class ArmC(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = build_backbone()
        f = int(self.backbone.num_features)
        self.norm, self.global_head = nn.LayerNorm(f), QueryMIL(f)
        self.roi_poolers = nn.ModuleDict({t: ROIPooler(f) for t in ROI_TARGETS})
        self.beta = nn.ParameterDict({t: nn.Parameter(torch.tensor(-2.0)) for t in ROI_TARGETS})

    def encode(self, x, mask):
        b, k = mask.shape
        flat, valid = x.flatten(0, 1), mask.to(x.device).flatten()
        h = x.new_zeros((b * k, int(self.backbone.num_features)))
        if valid.any():
            h[valid] = self.backbone(flat[valid]).to(h.dtype)
        return h.view(b, k, -1)

    def forward(
        self,
        global_x,
        global_mask,
        roi_x=None,
        roi_mask=None,
        return_global=False,
    ):
        h = self.norm(self.encode(global_x, global_mask))
        global_logits = self.global_head(h, global_mask)
        logits = global_logits.clone()
        roi_x, roi_mask = roi_x or {}, roi_mask or {}
        for target in ROI_TARGETS:
            if target not in roi_x:
                continue
            roi_features = self.encode(roi_x[target], roi_mask[target])
            residual = self.roi_poolers[target](roi_features, roi_mask[target])
            target_index = ROI_INDEX[target]
            logits[:, target_index] = (
                logits[:, target_index]
                + torch.sigmoid(self.beta[target]) * residual
            )
        if return_global:
            return logits, global_logits
        return logits


# -------------------------------- training ---------------------------------
def macro_auc(y, p):
    values = {}
    for j, name in enumerate(LABELS):
        truth = (y[:, j] > 0.5).astype(np.int64)
        if len(np.unique(truth)) > 1:
            values[name] = float(roc_auc_score(truth, p[:, j]))
    missing = sorted(set(LABELS) - set(values))
    if missing:
        raise ValueError(f"Gold split cannot compute AUC for targets: {missing}")
    return float(np.mean(list(values.values()))), values


def autocast_context(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")


def make_scaler(device):
    try:
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, global_predictions, truths = [], [], []
    for gx, gm, rx, rm, y in loader:
        with autocast_context(device):
            roi_x = {k: v.to(device) for k, v in rx.items()}
            roi_mask = {k: v.to(device) for k, v in rm.items()}
            logits, global_logits = model(
                gx.to(device),
                gm.to(device),
                roi_x,
                roi_mask,
                return_global=True,
            )
            p = torch.sigmoid(logits).float()
            global_p = torch.sigmoid(global_logits).float()
        predictions.append(p.cpu().numpy())
        global_predictions.append(global_p.cpu().numpy())
        truths.append(y.numpy())
    p, y = np.concatenate(predictions), np.concatenate(truths)
    global_p = np.concatenate(global_predictions)
    combined_auc, combined_by_target = macro_auc(y, p)
    global_auc, global_by_target = macro_auc(y, global_p)
    return {
        "combined_auc": combined_auc,
        "combined_by_target": combined_by_target,
        "global_auc": global_auc,
        "global_by_target": global_by_target,
    }


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, history, config, metrics):
    return {
        "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
        "config": config, "metrics": metrics, "labels": LABELS, "roi_targets": ROI_TARGETS,
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_global_initialization(model: nn.Module, checkpoint_path: str) -> dict[str, Any]:
    """Load and validate the proven Arm 13 global path exactly."""

    if not checkpoint_path:
        return {}
    path = Path(checkpoint_path)
    if not path.exists():
        discovered = find_file(checkpoint_path)
        if discovered is None:
            raise FileNotFoundError(
                "INIT_CHECKPOINT does not exist and was not found in "
                f"Kaggle inputs: {checkpoint_path}"
            )
        path = discovered
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_arch = checkpoint.get("arch") if isinstance(checkpoint, dict) else None
    checkpoint_res = checkpoint.get("res") if isinstance(checkpoint, dict) else None
    if checkpoint_arch is not None and checkpoint_arch != ARCH:
        raise ValueError(
            f"warm-start architecture mismatch: checkpoint={checkpoint_arch}, run={ARCH}"
        )
    if checkpoint_res is not None and int(checkpoint_res) != RES:
        raise ValueError(
            f"warm-start resolution mismatch: checkpoint={checkpoint_res}, run={RES}"
        )
    state = checkpoint.get("model", checkpoint)
    mapped = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("backbone.") or key.startswith("norm."):
            mapped[key] = value
        elif key.startswith("att."):
            mapped["global_head." + key] = value
        elif key in {"clsW", "clsb"}:
            mapped["global_head." + key] = value
    if not mapped:
        raise ValueError(f"No Arm 13-compatible global weights found in {path}")
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    unexpected = [x for x in unexpected if not x.startswith(("roi_poolers.", "beta."))]
    required = [x for x in missing if x.startswith(("backbone.", "norm.", "global_head."))]
    if unexpected or required:
        raise RuntimeError(
            "global checkpoint mismatch; "
            f"missing={required[:5]}, unexpected={unexpected[:5]}"
        )
    print(f"[Arm16] warm-started {len(mapped)} global tensors from {path}")
    return {
        "path": str(path),
        "arch": checkpoint_arch,
        "res": checkpoint_res,
        "loaded_tensors": len(mapped),
    }


def log_checkpoint(run, payload, epoch, metrics, final=False, extra_files=()):
    name = f"arm16_checkpoint_ep{epoch:03d}.pt"
    path = OUTPUT_DIR / name
    torch.save(payload, path)
    artifact = wandb.Artifact(
        name=f"arm16-{run.id}", type="model",
        metadata={
            "epoch": epoch,
            "final": final,
            **{
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            },
        },
    )
    artifact.add_file(str(path), name=name)
    for extra in extra_files:
        extra = Path(extra)
        artifact.add_file(str(extra), name=extra.name)
    run.log_artifact(artifact, aliases=["latest", "final" if final else f"epoch-{epoch}"])
    print(f"[Arm16] uploaded W&B artifact {name}")


def set_training_phase(model: ArmC, epoch: int) -> str:
    """Freeze the proven global model while new ROI heads find a signal."""

    roi_only = epoch < FREEZE_GLOBAL_EPOCHS
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not roi_only or name.startswith(("roi_poolers.", "beta."))
    if roi_only:
        model.backbone.eval()
        model.norm.eval()
        model.global_head.eval()
        return "roi_only"
    return "joint_finetune"


def restore_rng_state(state: dict[str, Any]) -> None:
    if state.get("rng_python") is not None:
        random.setstate(state["rng_python"])
    if state.get("rng_numpy") is not None:
        np.random.set_state(state["rng_numpy"])
    if state.get("rng_torch") is not None:
        torch.set_rng_state(state["rng_torch"])
    if torch.cuda.is_available() and state.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(state["rng_cuda"])


def save_top_candidate(model: nn.Module, epoch: int, auc: float, history) -> list[Path]:
    """Keep model-only files for the best validation epochs in this run."""

    ranked = sorted(history, key=lambda item: item["gold/macro_auc"], reverse=True)[:TOP_K]
    keep_epochs = {int(item["epoch"]) for item in ranked}
    candidate = OUTPUT_DIR / f"arm16_candidate_ep{epoch:03d}.pt"
    if epoch in keep_epochs:
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "gold_macro_auc": auc},
            candidate,
        )
    for path in OUTPUT_DIR.glob("arm16_candidate_ep*.pt"):
        match = re.search(r"ep(\d+)\.pt$", path.name)
        if match and int(match.group(1)) not in keep_epochs:
            path.unlink()
    return [
        OUTPUT_DIR / f"arm16_candidate_ep{int(item['epoch']):03d}.pt"
        for item in ranked
        if (OUTPUT_DIR / f"arm16_candidate_ep{int(item['epoch']):03d}.pt").exists()
    ]


def average_candidates(paths: list[Path]) -> dict[str, torch.Tensor]:
    if not paths:
        raise ValueError("no candidate checkpoints are available for averaging")
    states = [
        torch.load(path, map_location="cpu", weights_only=False)["model"]
        for path in paths
    ]
    averaged = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            value = first.float().clone()
            for state in states[1:]:
                value.add_(state[key].float())
            averaged[key] = value.div_(len(states)).to(first.dtype)
        else:
            averaged[key] = first.clone()
    return averaged


def init_wandb(config):
    """Authenticate without an interactive prompt in a Kaggle cell."""
    if WANDB_MODE:
        os.environ["WANDB_MODE"] = WANDB_MODE
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY, relogin=True, quiet=True)
    elif WANDB_MODE.lower() not in {"offline", "disabled"}:
        # Anonymous runs still support metric and artifact uploads. A configured
        # WANDB_API_KEY is recommended when the run must appear in a team project.
        wandb.login(anonymous="allow", quiet=True)
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config=config,
        job_type="train",
    )


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ACCUM_STEPS < 1 or ROI_K_TRAIN < 1 or TOP_K < 1 or UPLOAD_EVERY < 1:
        raise ValueError("accumulation, ROI sampling, top-k, and upload cadence must be positive")
    if FREEZE_GLOBAL_EPOCHS < 0 or FREEZE_GLOBAL_EPOCHS >= EPOCHS:
        raise ValueError("FREEZE_GLOBAL_EPOCHS must be in [0, EPOCHS)")
    if RESUME and INIT_CHECKPOINT:
        print("[Arm16] RESUME takes precedence over INIT_CHECKPOINT")
    if not RESUME and not INIT_CHECKPOINT and not ALLOW_SCRATCH:
        raise ValueError(
            "The high-probability Arm 16 recipe requires INIT_CHECKPOINT "
            "pointing to the Arm 13 best/SWA checkpoint. Set ALLOW_SCRATCH=True "
            "only for an explicitly separate scratch ablation."
        )
    if not RESUME and not INIT_CHECKPOINT and ALLOW_SCRATCH and not PRETRAINED:
        raise ValueError("a scratch ablation must set PRETRAINED=True")
    ids, masks, shard_ids = load_index()
    train_ids, gold_ids, labels = load_labels(ids)
    roi_cache = OptionalROICache(shard_ids)
    missing_roi = sorted(set(ROI_TARGETS) - set(roi_cache.data))
    if missing_roi:
        raise ValueError(f"required ROI targets are unavailable: {missing_roi}")
    id2row = {uid: i for i, uid in enumerate(ids)}
    train_ds = StudyDataset(
        train_ids,
        id2row,
        masks,
        labels,
        K_TRAIN,
        True,
        roi_cache,
    )
    gold_ds = StudyDataset(
        gold_ids,
        id2row,
        masks,
        labels,
        K_EVAL,
        False,
        roi_cache,
    )
    final_gold_ds = StudyDataset(
        gold_ids,
        id2row,
        masks,
        labels,
        K_FINAL,
        False,
        roi_cache,
    )
    loader_kwargs = {
        "num_workers": WORKERS,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": WORKERS > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    gold_loader = DataLoader(
        gold_ds,
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    )
    final_gold_loader = DataLoader(
        final_gold_ds,
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    )
    model = ArmC().to(device)
    initialization = {}
    if INIT_CHECKPOINT and not RESUME:
        initialization = load_global_initialization(model, INIT_CHECKPOINT)
    training_model = (
        nn.DataParallel(model)
        if device.type == "cuda" and torch.cuda.device_count() > 1
        else model
    )
    backbone_params = [
        param
        for name, param in model.named_parameters()
        if name.startswith("backbone.")
    ]
    global_head_params = [
        param
        for name, param in model.named_parameters()
        if name.startswith(("norm.", "global_head."))
    ]
    roi_params = [
        param
        for name, param in model.named_parameters()
        if name.startswith(("roi_poolers.", "beta."))
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": global_head_params, "lr": GLOBAL_HEAD_LR},
            {"params": roi_params, "lr": ROI_HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[BACKBONE_LR, GLOBAL_HEAD_LR, ROI_HEAD_LR],
        total_steps=EPOCHS * steps_per_epoch,
        pct_start=max(0.2, min(0.4, (FREEZE_GLOBAL_EPOCHS + 1) / EPOCHS)),
    )
    scaler = make_scaler(device)
    prevalence = np.clip(
        np.stack([labels[uid] for uid in train_ids]).mean(0),
        0.03,
        0.7,
    )
    pos_weight = torch.tensor(
        np.clip(np.sqrt((1 - prevalence) / prevalence), 1, 3),
        dtype=torch.float32,
        device=device,
    )
    config = {
        "arch": ARCH,
        "resolution": RES,
        "pretrained": PRETRAINED,
        "epochs": EPOCHS,
        "global_windows_train": K_TRAIN,
        "global_windows_eval": K_EVAL,
        "global_windows_final": K_FINAL,
        "roi_windows_train_per_target": ROI_K_TRAIN,
        "roi_windows_eval_per_target": 4,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * ACCUM_STEPS,
        "visible_gpus": torch.cuda.device_count(),
        "workers": WORKERS,
        "backbone_lr": BACKBONE_LR,
        "global_head_lr": GLOBAL_HEAD_LR,
        "roi_head_lr": ROI_HEAD_LR,
        "freeze_global_epochs": FREEZE_GLOBAL_EPOCHS,
        "global_loss_weight": GLOBAL_LOSS_WEIGHT,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "required_roi": REQUIRE_ROI,
        "initialization": initialization,
    }
    run = init_wandb(config)
    start_epoch, history = 0, []
    resume_path = Path(RESUME) if RESUME else None
    if resume_path and not resume_path.exists():
        resume_path = find_file(RESUME)
    if RESUME and resume_path is None:
        raise FileNotFoundError(f"RESUME was not found: {RESUME}")
    if resume_path is not None:
        state = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        saved_config = state.get("config", {})
        resume_contract = (
            "arch",
            "resolution",
            "epochs",
            "global_windows_train",
            "roi_windows_train_per_target",
            "batch_size",
            "accumulation_steps",
            "backbone_lr",
            "global_head_lr",
            "roi_head_lr",
            "freeze_global_epochs",
        )
        mismatches = {
            key: (saved_config.get(key), config[key])
            for key in resume_contract
            if saved_config.get(key) != config[key]
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        history = state.get("history", [])
        restore_rng_state(state)
        print(f"[Arm16] resumed from epoch {start_epoch - 1}")
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("gold/*", step_metric="epoch")
    run.define_metric("system/*", step_metric="epoch")
    run.define_metric("model/*", step_metric="epoch")
    if start_epoch == 0:
        baseline = evaluate(training_model, gold_loader, device)
        if INIT_CHECKPOINT and not math.isclose(
            baseline["combined_auc"],
            baseline["global_auc"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("new ROI branches changed the Arm 13 initialization")
        baseline_metrics = {
            "epoch": -1,
            "gold/macro_auc": baseline["combined_auc"],
            "gold/global_macro_auc": baseline["global_auc"],
        }
        baseline_metrics.update(
            {
                f"gold/{slug(name)}_auc": value
                for name, value in baseline["combined_by_target"].items()
            }
        )
        run.log(baseline_metrics)
        print(f"[Arm16] exact Arm 13 baseline macro AUC: {baseline['global_auc']:.6f}")
    roi_columns = torch.tensor(
        [ROI_INDEX[name] for name in ROI_TARGETS],
        dtype=torch.long,
        device=device,
    )
    for epoch in range(start_epoch, EPOCHS):
        started = time.time()
        training_model.train()
        phase = set_training_phase(model, epoch)
        running, count, optimizer_steps = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (gx, gm, rx, rm, y) in enumerate(train_loader):
            gx = gx.to(device, non_blocking=True)
            gm = gm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rx = {
                key: value.to(device, non_blocking=True)
                for key, value in rx.items()
            }
            rm = {
                key: value.to(device, non_blocking=True)
                for key, value in rm.items()
            }
            with autocast_context(device):
                logits, global_logits = training_model(
                    gx,
                    gm,
                    rx,
                    rm,
                    return_global=True,
                )
                if phase == "roi_only":
                    loss = F.binary_cross_entropy_with_logits(
                        logits.index_select(1, roi_columns).float(),
                        y.index_select(1, roi_columns).float(),
                        pos_weight=pos_weight.index_select(0, roi_columns),
                    )
                else:
                    combined_loss = F.binary_cross_entropy_with_logits(
                        logits.float(),
                        y.float(),
                        pos_weight=pos_weight,
                    )
                    global_roi_loss = F.binary_cross_entropy_with_logits(
                        global_logits.index_select(1, roi_columns).float(),
                        y.index_select(1, roi_columns).float(),
                        pos_weight=pos_weight.index_select(0, roi_columns),
                    )
                    loss = combined_loss + GLOBAL_LOSS_WEIGHT * global_roi_loss
            group_start = (batch_index // ACCUM_STEPS) * ACCUM_STEPS
            accumulation_divisor = min(
                ACCUM_STEPS,
                len(train_loader) - group_start,
            )
            scaler.scale(loss / accumulation_divisor).backward()
            should_step = (
                (batch_index + 1) % ACCUM_STEPS == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
            running += float(loss.detach())
            count += 1
        evaluation = evaluate(training_model, gold_loader, device)
        metrics = {
            "epoch": epoch,
            "train/loss": running / max(count, 1),
            "train/phase_roi_only": float(phase == "roi_only"),
            "gold/macro_auc": evaluation["combined_auc"],
            "gold/global_macro_auc": evaluation["global_auc"],
            "gold/roi_gain_macro_auc": (
                evaluation["combined_auc"] - evaluation["global_auc"]
            ),
            "system/epoch_minutes": (time.time() - started) / 60,
            "system/optimizer_steps": optimizer_steps,
            "system/lr_backbone": optimizer.param_groups[0]["lr"],
            "system/lr_global_head": optimizer.param_groups[1]["lr"],
            "system/lr_roi_head": optimizer.param_groups[2]["lr"],
        }
        metrics.update(
            {
                f"gold/{slug(key)}_auc": value
                for key, value in evaluation["combined_by_target"].items()
            }
        )
        metrics.update(
            {
                f"gold/{slug(key)}_roi_gain": (
                    value - evaluation["global_by_target"][key]
                )
                for key, value in evaluation["combined_by_target"].items()
                if key in ROI_TARGETS
            }
        )
        metrics.update(
            {
                f"model/gate_{slug(key)}": float(
                    torch.sigmoid(model.beta[key]).detach().cpu()
                )
                for key in ROI_TARGETS
            }
        )
        run.log(metrics)
        history.append(metrics)
        candidate_paths = save_top_candidate(
            model,
            epoch,
            evaluation["combined_auc"],
            history,
        )
        print(
            json.dumps(
                {
                    key: round(float(value), 5)
                    if isinstance(value, (float, int))
                    else value
                    for key, value in metrics.items()
                }
            )
        )
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            history,
            config,
            metrics,
        )
        torch.save(payload, OUTPUT_DIR / "arm16_last.pt")
        if (epoch + 1) % UPLOAD_EVERY == 0:
            extras = candidate_paths[:1]
            log_checkpoint(run, payload, epoch, metrics, extra_files=extras)
    final_epoch = max(start_epoch, EPOCHS) - 1
    final_metrics = history[-1] if history else {"epoch": final_epoch}
    resumable_payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        scaler,
        final_epoch,
        history,
        config,
        final_metrics,
    )
    candidate_paths = save_top_candidate(
        model,
        final_epoch,
        float(final_metrics.get("gold/macro_auc", float("nan"))),
        history,
    )
    if not candidate_paths:
        raise RuntimeError("no top validation checkpoint survived for final selection")
    best_state = torch.load(
        candidate_paths[0],
        map_location="cpu",
        weights_only=False,
    )["model"]
    model.load_state_dict(best_state)
    best_evaluation = evaluate(training_model, gold_loader, device)
    selected_name = "best_single"
    selected_state = best_state
    if len(candidate_paths) > 1:
        averaged_state = average_candidates(candidate_paths)
        model.load_state_dict(averaged_state)
        averaged_evaluation = evaluate(training_model, gold_loader, device)
        if averaged_evaluation["combined_auc"] >= best_evaluation["combined_auc"]:
            selected_name = f"top_{len(candidate_paths)}_average"
            selected_state = averaged_state
        else:
            model.load_state_dict(best_state)
    final_evaluation = evaluate(training_model, final_gold_loader, device)
    selected_path = OUTPUT_DIR / "arm16_selected_model.pt"
    torch.save(
        {
            "model": selected_state,
            "arch": ARCH,
            "res": RES,
            "labels": LABELS,
            "roi_targets": ROI_TARGETS,
            "selection": selected_name,
            "candidate_epochs": [
                int(re.search(r"ep(\d+)\.pt$", path.name).group(1))
                for path in candidate_paths
            ],
            "gold_final": final_evaluation,
            "config": config,
        },
        selected_path,
    )
    final_log = {
        "epoch": EPOCHS,
        "gold/final_macro_auc_k62": final_evaluation["combined_auc"],
        "gold/final_global_macro_auc_k62": final_evaluation["global_auc"],
    }
    final_log.update(
        {
            f"gold/final_{slug(key)}_auc_k62": value
            for key, value in final_evaluation["combined_by_target"].items()
        }
    )
    run.log(final_log)
    log_checkpoint(
        run,
        resumable_payload,
        final_epoch,
        {**final_metrics, **final_log},
        final=True,
        extra_files=[selected_path, *candidate_paths],
    )
    run.summary.update(
        {
            "best_gold_macro_auc": max(
                (item["gold/macro_auc"] for item in history),
                default=float("nan"),
            ),
            "selected_model": selected_name,
            "final_gold_macro_auc_k62": final_evaluation["combined_auc"],
            "completed_epoch": final_epoch,
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
