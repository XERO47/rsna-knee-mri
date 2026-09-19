"""Standalone Kaggle trainer for Arm C (global MIL + local ROI residuals).

Run this entire file in one Kaggle notebook cell, for example:

    exec(open('/kaggle/working/train_arm16_kaggle.py').read())

The script needs the cache64 files (vols/ids/masks shard 0 and 1) and the
validated gold_test.csv/train_pool.csv label files in attached Kaggle inputs.
ROI bags are optional at startup so the script remains runnable with the
existing cache64 dataset. To enable the finalized local branch, attach files
named ``roi_<slug>_shard{0,1}.npy`` and matching mask files. Each ROI array is
uint8 [N,4,3,336,336], each mask is binary uint8 [N,4], and
``roi_ids_shard{0,1}.npy`` must exactly equal the corresponding global IDs.

W&B logs epoch metrics and uploads a complete, resumable checkpoint artifact at
every even epoch and after the final epoch. Checkpoints contain model,
optimizer, OneCycle scheduler, AMP scaler, history, RNG states, and config.
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
RES = int(os.environ.get("ARM16_RES", "336"))
ARCH = os.environ.get("ARM16_ARCH", "convnext_small.in12k_ft_in1k_384")
PRETRAINED = os.environ.get("ARM16_PRETRAINED", "1") == "1"
EPOCHS = int(os.environ.get("ARM16_EPOCHS", "12"))
K_TRAIN = int(os.environ.get("ARM16_K_TRAIN", "12"))
K_EVAL = int(os.environ.get("ARM16_K_EVAL", "24"))
BATCH_SIZE = int(os.environ.get("ARM16_BATCH_SIZE", "4"))
WORKERS = int(os.environ.get("ARM16_WORKERS", "2"))
BACKBONE_LR = float(os.environ.get("ARM16_BACKBONE_LR", "1e-4"))
HEAD_LR = float(os.environ.get("ARM16_HEAD_LR", "3e-4"))
WEIGHT_DECAY = float(os.environ.get("ARM16_WEIGHT_DECAY", "0.05"))
CLIP_NORM = float(os.environ.get("ARM16_CLIP_NORM", "3.0"))
SEED = int(os.environ.get("ARM16_SEED", "42"))
OUTPUT_DIR = Path(os.environ.get("ARM16_OUTPUT", "/kaggle/working/arm16"))
RESUME = os.environ.get("ARM16_RESUME", "")
INIT_CHECKPOINT = os.environ.get("ARM16_INIT_CHECKPOINT", "")
REQUIRE_ROI = os.environ.get("ARM16_REQUIRE_ROI", "0") == "1"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rsna-knee-arm16")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "") or None
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "arm16-global-local")
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
    roots = [Path(os.environ.get("ARM16_INPUT", "/kaggle/input")), Path("/kaggle/working")]
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
            "ARM16_INPUT to its mounted directory."
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
    idset = set(ids.astype(str))
    gold_ids = [str(x) for x in gold.StudyInstanceUID if str(x) in idset]
    gold_set = set(gold_ids)
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
    if len(gold_ids) == 0:
        raise ValueError("gold_test.csv did not overlap cache ids")
    return train_ids, gold_ids, labels


class OptionalROICache:
    """Loads optional per-target ROI arrays, preserving cache row order."""

    def __init__(self, shard_ids: tuple[np.ndarray, np.ndarray]):
        self.n0 = len(shard_ids[0])
        self.data: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        any_roi_data = any(
            find_file(f"roi_{slug(target)}_shard{shard}.npy") is not None
            for target in ROI_TARGETS
            for shard in (0, 1)
        )
        if any_roi_data or REQUIRE_ROI:
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
            if arr.shape[1] == 1:
                arr = np.repeat(arr, 3, axis=2)
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
    return timm.create_model(
        ARCH,
        pretrained=PRETRAINED,
        num_classes=0,
        in_chans=3,
        global_pool="avg",
    )


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

    def forward(self, global_x, global_mask, roi_x=None, roi_mask=None):
        h = self.norm(self.encode(global_x, global_mask))
        logits = self.global_head(h, global_mask).clone()
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
        return logits


# -------------------------------- training ---------------------------------
def macro_auc(y, p):
    values = {}
    for j, name in enumerate(LABELS):
        truth = (y[:, j] > 0.5).astype(np.int64)
        if len(np.unique(truth)) > 1:
            values[name] = float(roc_auc_score(truth, p[:, j]))
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
    predictions, truths = [], []
    for gx, gm, rx, rm, y in loader:
        with autocast_context(device):
            roi_x = {k: v.to(device) for k, v in rx.items()}
            roi_mask = {k: v.to(device) for k, v in rm.items()}
            p = torch.sigmoid(
                model(gx.to(device), gm.to(device), roi_x, roi_mask)
            ).float()
        predictions.append(p.cpu().numpy())
        truths.append(y.numpy())
    p, y = np.concatenate(predictions), np.concatenate(truths)
    return macro_auc(y, p)


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, history, config, metrics):
    return {
        "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
        "config": config, "metrics": metrics, "labels": LABELS, "roi_targets": ROI_TARGETS,
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_global_initialization(model: nn.Module, checkpoint_path: str) -> None:
    """Load the proven Arm 12/15 global weights when a checkpoint is supplied."""
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"ARM16_INIT_CHECKPOINT does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
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
        raise ValueError(f"No Arm 12/15 global weights found in {path}")
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    unexpected = [x for x in unexpected if not x.startswith(("roi_poolers.", "beta."))]
    required = [x for x in missing if x.startswith(("backbone.", "norm.", "global_head."))]
    if unexpected or required:
        raise RuntimeError(
            "global checkpoint mismatch; "
            f"missing={required[:5]}, unexpected={unexpected[:5]}"
        )
    print(f"[Arm16] warm-started global path from {path}")


def log_checkpoint(run, payload, epoch, metrics, final=False):
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
    run.log_artifact(artifact, aliases=["latest", "final" if final else f"epoch-{epoch}"])
    print(f"[Arm16] uploaded W&B artifact {name}")


def init_wandb(config):
    """Authenticate without an interactive prompt in a Kaggle cell."""
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True, quiet=True)
    elif os.environ.get("WANDB_MODE", "").lower() not in {"offline", "disabled"}:
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
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    ids, masks, shard_ids = load_index()
    train_ids, gold_ids, labels = load_labels(ids)
    roi_cache = OptionalROICache(shard_ids)
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
        batch_size=2,
        shuffle=False,
        **loader_kwargs,
    )
    model = ArmC().to(device)
    backbone_params = [
        param
        for name, param in model.named_parameters()
        if name.startswith("backbone.")
    ]
    head_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[BACKBONE_LR, HEAD_LR],
        total_steps=EPOCHS * len(train_loader),
        pct_start=0.1,
    )
    scaler = make_scaler(device)
    load_global_initialization(model, INIT_CHECKPOINT)
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
        key: value
        for key, value in globals().items()
        if key.startswith("ARM16_")
        or key
        in {
            "ARCH",
            "PRETRAINED",
            "EPOCHS",
            "K_TRAIN",
            "K_EVAL",
            "BATCH_SIZE",
            "WORKERS",
        }
    }
    run = init_wandb(config)
    start_epoch, history = 0, []
    resume_path = Path(RESUME) if RESUME else None
    if resume_path and resume_path.exists():
        state = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        history = state.get("history", [])
        print(f"[Arm16] resumed from epoch {start_epoch - 1}")
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("gold/*", step_metric="epoch")
    run.define_metric("system/*", step_metric="epoch")
    for epoch in range(start_epoch, EPOCHS):
        started = time.time()
        model.train()
        running, count = 0.0, 0
        for gx, gm, rx, rm, y in train_loader:
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
                logits = model(gx, gm, rx, rm)
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(),
                    y.float(),
                    pos_weight=pos_weight,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
            count += 1
        gold_auc, gold_by_target = evaluate(model, gold_loader, device)
        metrics = {
            "epoch": epoch,
            "train/loss": running / max(count, 1),
            "gold/macro_auc": gold_auc,
            "system/epoch_minutes": (time.time() - started) / 60,
            "system/lr_backbone": optimizer.param_groups[0]["lr"],
            "system/lr_head": optimizer.param_groups[1]["lr"],
        }
        metrics.update(
            {f"gold/{slug(key)}_auc": value for key, value in gold_by_target.items()}
        )
        metrics.update(
            {
                f"model/gate_{slug(key)}": float(
                    torch.sigmoid(model.beta[key]).detach().cpu()
                )
                for key in ROI_TARGETS
            }
        )
        run.log(metrics, step=epoch)
        history.append(metrics)
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
        if (epoch + 1) % 2 == 0:
            log_checkpoint(run, payload, epoch, metrics)
    final_epoch = max(start_epoch, EPOCHS) - 1
    final_metrics = history[-1] if history else {"epoch": final_epoch}
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        scaler,
        final_epoch,
        history,
        config,
        final_metrics,
    )
    log_checkpoint(run, payload, final_epoch, final_metrics, final=True)
    run.summary.update(
        {
            "best_gold_macro_auc": max(
                (item["gold/macro_auc"] for item in history),
                default=float("nan"),
            ),
            "completed_epoch": final_epoch,
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
