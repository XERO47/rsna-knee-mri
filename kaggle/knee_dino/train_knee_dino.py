"""Case-level self-supervised pretraining for RSNA knee MRI.

The positive unit is a study/case, not an individual slice. Each sample has:
  * two global views from distinct series of the same study whenever possible;
  * four local views from series belonging to that same study.

A ResNet34 student learns by matching an EMA teacher using the DINO objective
(teacher centering and sharpening, no negatives). After training, per-series
mid-slice features are mean-pooled to exactly one vector per study. The output
diagnostic intentionally matches kaggle/embed/run_embed_experiment.py:
PCA->t-SNE and k=10 neighbor label agreement on the same 58 gold + 350 pool
study sample.

This is a representation pilot, not a competition submission model. Labels
are loaded only after self-supervised optimization is finished.
"""
import json
import math
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet34


LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
SEED = 42
N_POOL = 350
GLOBAL_SIZE = 192
LOCAL_SIZE = 96
N_LOCAL = 4
BATCH_SIZE = 32
EPOCHS = 40
LR = 2e-4
WEIGHT_DECAY = 0.04
OUT_DIM = 4096
STUDENT_TEMP = 0.1
TEACHER_TEMP_START = 0.07
TEACHER_TEMP_FINAL = 0.04
TEACHER_TEMP_WARMUP_EPOCHS = 8
CENTER_MOMENTUM = 0.9
BASE_TEACHER_MOMENTUM = 0.996
K_NEIGHBORS = 10
NUM_WORKERS = 2
TIME_BUDGET_S = 7.5 * 3600
PAIR_MODE = "curriculum"


def find_root(input_root: Path, marker: str,
              skip_dirs=("train_series", "test_series")) -> Path:
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{marker!r} not found below {input_root}")


INPUT_ROOT = Path("/kaggle/input")
COMP_ROOT = find_root(INPUT_ROOT, "train_series.csv")
LABELS_ROOT = find_root(INPUT_ROOT, "gold_test.csv")
SERIES_ROOT = COMP_ROOT / "train_series"
WORK = Path("/kaggle/working")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def sorted_series_files(series_dir: Path) -> list[Path]:
    """Sort spatially by DICOM InstanceNumber, never by UID filename."""
    tagged = []
    for path in series_dir.glob("*.dcm"):
        try:
            dcm = pydicom.dcmread(str(path), stop_before_pixels=True)
            instance = int(getattr(dcm, "InstanceNumber", 0))
        except Exception:
            instance = 0
        tagged.append((instance, path))
    tagged.sort(key=lambda x: x[0])
    return [path for _, path in tagged]


def read_dicom(path: Path) -> np.ndarray:
    dcm = pydicom.dcmread(str(path))
    arr = dcm.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi).astype(np.float32, copy=False)
    arr -= arr.min()
    denom = arr.max()
    if denom > 0:
        arr /= denom
    return arr.astype(np.float32, copy=False)


def augment_slice(arr: np.ndarray, size: int, local: bool,
                  rng: np.random.Generator) -> torch.Tensor:
    """MRI-safe augmentation: no horizontal/vertical flips."""
    h, w = arr.shape
    low, high = ((0.40, 0.72) if local else (0.72, 1.0))
    scale = float(rng.uniform(low, high))
    crop_h = max(8, min(h, int(h * math.sqrt(scale))))
    crop_w = max(8, min(w, int(w * math.sqrt(scale))))
    y0 = int(rng.integers(0, max(1, h - crop_h + 1)))
    x0 = int(rng.integers(0, max(1, w - crop_w + 1)))
    out = arr[y0:y0 + crop_h, x0:x0 + crop_w]
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)

    if rng.random() < 0.8:
        gamma = float(rng.uniform(0.75, 1.35))
        out = np.power(np.clip(out, 0, 1), gamma)
    if rng.random() < 0.8:
        out = out * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-0.08, 0.08))
    if rng.random() < 0.5:
        out += rng.normal(0, rng.uniform(0.0, 0.035), out.shape).astype(np.float32)
    if rng.random() < 0.35:
        angle = float(rng.uniform(-7.0, 7.0))
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        out = cv2.warpAffine(out, matrix, (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    out = np.clip(out, 0, 1).astype(np.float32, copy=False)
    return torch.from_numpy(out).unsqueeze(0)


class CaseViewsDataset(Dataset):
    """One item is one case and six views, all from that case."""

    def __init__(self, studies: pd.DataFrame, series_df: pd.DataFrame,
                 series_root: Path, pair_mode: str):
        self.studies = studies.reset_index(drop=True)
        self.series_root = series_root
        self.pair_mode = pair_mode
        # Shared value stays visible to persistent DataLoader workers.
        # 0=easy same-slice, 1=different-series same-plane, 2=different-plane.
        self.curriculum_stage = mp.Value("d", 0.0)
        self.cases = {}
        self.bad_series = 0
        self.single_series_cases = 0
        self.no_same_plane_pair_cases = 0
        self.no_different_plane_pair_cases = 0
        print("Indexing DICOM series in InstanceNumber order...", flush=True)
        for i, uid in enumerate(self.studies.StudyInstanceUID):
            sdf = series_df[(series_df.StudyInstanceUID == uid)]
            entries = []
            for _, row in sdf.iterrows():
                files = sorted_series_files(
                    series_root / uid / row.SeriesInstanceUID
                )
                if files:
                    entries.append({
                        "series_uid": row.SeriesInstanceUID,
                        "plane": row.Anatomical_Plane,
                        "files": files,
                    })
                else:
                    self.bad_series += 1
            if len(entries) < 2:
                self.single_series_cases += 1
            if not any(a != b and entries[a]["plane"] == entries[b]["plane"]
                       for a in range(len(entries)) for b in range(len(entries))):
                self.no_same_plane_pair_cases += 1
            if not any(a != b and entries[a]["plane"] != entries[b]["plane"]
                       for a in range(len(entries)) for b in range(len(entries))):
                self.no_different_plane_pair_cases += 1
            self.cases[uid] = entries
            if (i + 1) % 50 == 0:
                print(f"  indexed {i + 1}/{len(self.studies)} cases", flush=True)
        print(f"single/zero usable-series fallback cases: {self.single_series_cases} / "
              f"{len(self.studies)}; unusable series: {self.bad_series}", flush=True)
        print(f"no same-plane pair: {self.no_same_plane_pair_cases}; "
              f"no different-plane pair: {self.no_different_plane_pair_cases}",
              flush=True)

    def __len__(self):
        return len(self.studies)

    @staticmethod
    def _pick_path(entry, rng):
        files = entry["files"]
        lo = int(0.05 * max(0, len(files) - 1))
        hi = max(lo + 1, int(0.95 * max(1, len(files) - 1)) + 1)
        return files[int(rng.integers(lo, min(len(files), hi)))]

    @staticmethod
    def _path_at_fraction(entry, fraction):
        files = entry["files"]
        index = int(round(float(fraction) * max(0, len(files) - 1)))
        return files[min(len(files) - 1, max(0, index))]

    @staticmethod
    def _easy_views(entries, rng):
        entry = entries[int(rng.integers(0, len(entries)))]
        arr = read_dicom(CaseViewsDataset._pick_path(entry, rng))
        globals_out = [
            augment_slice(arr, GLOBAL_SIZE, False, rng) for _ in range(2)
        ]
        locals_out = [
            augment_slice(arr, LOCAL_SIZE, True, rng) for _ in range(N_LOCAL)
        ]
        return globals_out, locals_out

    def __getitem__(self, idx):
        uid = self.studies.iloc[idx].StudyInstanceUID
        entries = self.cases[uid]
        if not entries:
            raise RuntimeError(f"No usable series for case {uid}")
        # DataLoader seeds NumPy independently in every worker. Draw a fresh
        # child seed so a case receives new slices/crops on every epoch.
        rng = np.random.default_rng(np.random.randint(0, 2 ** 32 - 1))

        if self.pair_mode == "easy_smoke":
            # Isolated vanilla-DINO control: all augmented views share one
            # source slice. If this collapses, cross-plane difficulty is not
            # the root cause and a curriculum cannot rescue the implementation.
            globals_out, locals_out = self._easy_views(entries, rng)
            return torch.stack(globals_out), torch.stack(locals_out), uid

        stage = float(self.curriculum_stage.value)
        if stage < 1.0:
            requested_tier = "medium" if rng.random() < stage else "easy"
        else:
            requested_tier = "hard" if rng.random() < stage - 1.0 else "medium"

        if requested_tier == "easy":
            globals_out, locals_out = self._easy_views(entries, rng)
            return torch.stack(globals_out), torch.stack(locals_out), uid

        want_same_plane = requested_tier == "medium"
        pairs = [
            (a, b) for a in range(len(entries)) for b in range(len(entries))
            if a != b and ((entries[a]["plane"] == entries[b]["plane"])
                           if want_same_plane else
                           (entries[a]["plane"] != entries[b]["plane"]))
        ]
        if not pairs and requested_tier == "hard":
            pairs = [
                (a, b) for a in range(len(entries)) for b in range(len(entries))
                if a != b and entries[a]["plane"] == entries[b]["plane"]
            ]
        if not pairs:
            globals_out, locals_out = self._easy_views(entries, rng)
            return torch.stack(globals_out), torch.stack(locals_out), uid

        a, b = pairs[int(rng.integers(0, len(pairs)))]
        global_entries = [entries[a], entries[b]]
        fraction = float(rng.uniform(0.08, 0.92))
        source_arrays = [
            read_dicom(self._path_at_fraction(entry, fraction))
            for entry in global_entries
        ]
        globals_out = [
            augment_slice(arr, GLOBAL_SIZE, False, rng) for arr in source_arrays
        ]
        locals_out = [
            augment_slice(source_arrays[i % 2], LOCAL_SIZE, True, rng)
            for i in range(N_LOCAL)
        ]
        return torch.stack(globals_out), torch.stack(locals_out), uid


class DINOHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024, bottleneck_dim=256,
                 out_dim=OUT_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1)
        return self.last_layer(x)


class DINOModel(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet34(weights=None)
        net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        net.fc = nn.Identity()
        self.encoder = net
        self.head = DINOHead()

    def features(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.features(x))


class DINOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(1, OUT_DIM))

    def forward(self, student_outputs, teacher_outputs, teacher_temp):
        student = [x / STUDENT_TEMP for x in student_outputs]
        teacher = [F.softmax((x - self.center) / teacher_temp, dim=-1).detach()
                   for x in teacher_outputs]
        total = 0.0
        terms = 0
        for teacher_index, q in enumerate(teacher):
            for student_index, v in enumerate(student):
                if student_index == teacher_index:
                    continue
                total += torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean()
                terms += 1
        with torch.no_grad():
            batch_center = torch.cat(teacher_outputs).float().mean(dim=0, keepdim=True)
            self.center.mul_(CENTER_MOMENTUM).add_(
                batch_center * (1 - CENTER_MOMENTUM)
            )
        return total / terms


def cosine_value(start, end, step, total_steps):
    return end - (end - start) * (math.cos(math.pi * step / total_steps) + 1) / 2


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)
    # ResNet BatchNorm running statistics are buffers, not parameters. The
    # teacher stays in eval mode, so copy the student's current buffers after
    # every step rather than leaving the teacher at fresh-init mean/variance.
    for student_buffer, teacher_buffer in zip(student.buffers(), teacher.buffers()):
        teacher_buffer.copy_(student_buffer)


def train_ssl(studies, series_df, device):
    train_ds = CaseViewsDataset(studies, series_df, SERIES_ROOT, PAIR_MODE)
    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=NUM_WORKERS > 0, drop_last=True,
    )
    student = DINOModel().to(device)
    # Legacy weight_norm tensors cannot be deep-copied on current Kaggle
    # PyTorch. Construct the identical module explicitly, then copy its state.
    teacher = DINOModel().to(device)
    teacher.load_state_dict(student.state_dict())
    # Canonical CNN-DINO keeps the EMA teacher in train mode so BatchNorm uses
    # live batch statistics, matching the student's normalization behavior.
    # There is no Dropout in this model and gradients remain disabled below.
    teacher.train()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    loss_fn = DINOLoss().to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    total_steps = EPOCHS * len(loader)
    warmup_steps = max(1, int(0.1 * total_steps))
    history = []
    global_step = 0
    started = time.time()

    for epoch in range(EPOCHS):
        student.train()
        if PAIR_MODE == "curriculum":
            if epoch < 8:
                curriculum_stage = 0.0
            elif epoch < 16:
                curriculum_stage = (epoch - 7) / 8.0
            elif epoch < 24:
                curriculum_stage = 1.0
            elif epoch < 32:
                curriculum_stage = 1.0 + (epoch - 23) / 8.0
            else:
                curriculum_stage = 2.0
            train_ds.curriculum_stage.value = curriculum_stage
        else:
            curriculum_stage = 0.0
        losses = []
        for global_views, local_views, _ in loader:
            if time.time() - started > TIME_BUDGET_S:
                print("Time budget reached; ending at a complete optimizer step.")
                break
            global_views = global_views.to(device, non_blocking=True)
            local_views = local_views.to(device, non_blocking=True)
            b = global_views.shape[0]
            if global_step < warmup_steps:
                current_lr = LR * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
                current_lr = cosine_value(LR, 1e-6, progress, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            teacher_temp = (
                TEACHER_TEMP_START
                + (TEACHER_TEMP_FINAL - TEACHER_TEMP_START)
                * min(1.0, epoch / max(1, TEACHER_TEMP_WARMUP_EPOCHS - 1))
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                student_global_logits = student(
                    global_views.reshape(-1, 1, GLOBAL_SIZE, GLOBAL_SIZE)
                ).reshape(b, 2, OUT_DIM)
                student_local_logits = student(
                    local_views.reshape(-1, 1, LOCAL_SIZE, LOCAL_SIZE)
                ).reshape(b, N_LOCAL, OUT_DIM)
                student_outputs = [student_global_logits[:, i] for i in range(2)]
                student_outputs += [student_local_logits[:, i] for i in range(N_LOCAL)]
                with torch.no_grad():
                    teacher_logits = teacher(
                        global_views.reshape(-1, 1, GLOBAL_SIZE, GLOBAL_SIZE)
                    ).reshape(b, 2, OUT_DIM)
                    teacher_outputs = [teacher_logits[:, i] for i in range(2)]
                loss = loss_fn(student_outputs, teacher_outputs, teacher_temp)
            scaler.scale(loss).backward()
            if epoch == 0:
                # Standard DINO cold-start stabilization: allow the encoder
                # and MLP to move before the prototype layer can reorganize.
                for parameter in student.head.last_layer.parameters():
                    parameter.grad = None
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()

            momentum = cosine_value(
                BASE_TEACHER_MOMENTUM, 1.0, global_step, max(1, total_steps - 1)
            )
            update_teacher(student, teacher, momentum)
            global_step += 1
            losses.append(float(loss.detach()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        with torch.no_grad():
            center_norm = float(loss_fn.center.norm())
            teacher_prob = F.softmax(
                (teacher_outputs[0] - loss_fn.center) / teacher_temp, dim=-1
            )
            teacher_entropy = float(
                (-teacher_prob * torch.log(teacher_prob + 1e-8)).sum(-1).mean()
            )
        row = {
            "epoch": epoch + 1,
            "loss": mean_loss,
            "teacher_momentum": momentum,
            "learning_rate": current_lr,
            "teacher_temperature": teacher_temp,
            "curriculum_stage": curriculum_stage,
            "center_norm": center_norm,
            "teacher_entropy": teacher_entropy,
            "elapsed_min": (time.time() - started) / 60,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if time.time() - started > TIME_BUDGET_S:
            break

    checkpoint = {
        "encoder": student.encoder.state_dict(),
        "head": student.head.state_dict(),
        "teacher_encoder": teacher.encoder.state_dict(),
        "config": {
            "backbone": "resnet34", "out_dim": OUT_DIM,
            "global_size": GLOBAL_SIZE, "local_size": LOCAL_SIZE,
            "n_local": N_LOCAL, "epochs_requested": EPOCHS,
            "n_studies": len(studies),
            "pair_mode": PAIR_MODE,
            "single_series_fallback_cases": train_ds.single_series_cases,
            "no_same_plane_pair_cases": train_ds.no_same_plane_pair_cases,
            "no_different_plane_pair_cases": train_ds.no_different_plane_pair_cases,
        },
        "history": history,
    }
    torch.save(checkpoint, WORK / "knee_dino_checkpoint.pt")
    with open(WORK / "training_history.json", "w") as handle:
        json.dump(history, handle, indent=2)
    return student.encoder, train_ds, history


@torch.no_grad()
def embed_studies(encoder, studies, series_df, device):
    encoder.eval()
    rows = []
    embeddings = []
    for index, row in studies.iterrows():
        uid = row.StudyInstanceUID
        sdf = series_df[series_df.StudyInstanceUID == uid]
        case_features = []
        for _, series in sdf.iterrows():
            files = sorted_series_files(
                SERIES_ROOT / uid / series.SeriesInstanceUID
            )
            if not files:
                continue
            try:
                arr = read_dicom(files[len(files) // 2])
                arr = cv2.resize(arr, (GLOBAL_SIZE, GLOBAL_SIZE),
                                 interpolation=cv2.INTER_AREA)
                x = torch.from_numpy(arr.astype(np.float32)).view(
                    1, 1, GLOBAL_SIZE, GLOBAL_SIZE
                ).to(device)
                feature = encoder(x)
                case_features.append(feature.cpu().numpy()[0])
            except Exception as error:
                print(f"embedding error {uid} {series.SeriesInstanceUID}: {error}")
        if not case_features:
            continue
        # Keep the aggregation identical to the frozen-DINOv2 experiment:
        # raw mean pooling, without an extra per-case L2 normalization.
        embedding = np.mean(case_features, axis=0)
        embeddings.append(embedding)
        item = row.to_dict()
        item["n_series"] = len(case_features)
        item["fluid_fraction"] = float(sdf.Fluid_Sensitive.mean())
        item["fatsat_fraction"] = float(sdf.Fat_Suppression.mean())
        for plane in ["Axial", "Coronal", "Sagittal"]:
            item[f"plane_{plane.lower()}_fraction"] = float(
                (sdf.Anatomical_Plane == plane).mean()
            )
        rows.append(item)
        if (index + 1) % 50 == 0:
            print(f"  embedded {index + 1}/{len(studies)} cases", flush=True)
    return np.stack(embeddings), pd.DataFrame(rows)


def evaluate_embeddings(embeddings, meta):
    reduced = PCA(
        n_components=min(50, embeddings.shape[0] - 1, embeddings.shape[1]),
        random_state=SEED,
    ).fit_transform(embeddings)
    xy = TSNE(n_components=2, perplexity=30, random_state=SEED,
              init="pca", learning_rate="auto").fit_transform(reduced)
    meta["tsne_x"], meta["tsne_y"] = xy[:, 0], xy[:, 1]

    neighbors = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(embeddings)
    _, neighbor_idx = neighbors.kneighbors(embeddings)
    neighbor_idx = neighbor_idx[:, 1:]
    rng = np.random.default_rng(SEED)
    random_idx = rng.integers(0, len(meta), size=neighbor_idx.shape)

    targets = meta[LABELS].to_numpy(dtype=float)
    neighbor_pred = targets[neighbor_idx].mean(axis=1)
    random_pred = targets[random_idx].mean(axis=1)
    label_results = {}
    for j, label in enumerate(LABELS):
        label_results[label] = {
            "r2_knn_neighbors": round(float(r2_score(targets[:, j], neighbor_pred[:, j])), 4),
            "r2_random_pairs": round(float(r2_score(targets[:, j], random_pred[:, j])), 4),
        }

    protocol_columns = [
        "n_series", "fluid_fraction", "fatsat_fraction",
        "plane_axial_fraction", "plane_coronal_fraction",
        "plane_sagittal_fraction",
    ]
    protocol_results = {}
    for column in protocol_columns:
        values = meta[column].to_numpy(dtype=float)
        protocol_results[column] = {
            "r2_knn_neighbors": round(float(r2_score(values, values[neighbor_idx].mean(axis=1))), 4),
            "r2_random_pairs": round(float(r2_score(values, values[random_idx].mean(axis=1))), 4),
        }

    # Gold-only AUC using pseudo-labeled pool neighbors as a secondary diagnostic.
    pool_mask = meta.split.to_numpy() == "pool"
    gold_mask = meta.split.to_numpy() == "gold"
    pool_nn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, int(pool_mask.sum()))).fit(
        embeddings[pool_mask]
    )
    _, pool_neighbor_idx = pool_nn.kneighbors(embeddings[gold_mask])
    gold_pred = targets[pool_mask][pool_neighbor_idx].mean(axis=1)
    gold_true = targets[gold_mask]
    gold_auc = {}
    for j, label in enumerate(LABELS):
        gold_auc[label] = float(roc_auc_score(gold_true[:, j], gold_pred[:, j]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for split, color in [("pool", "#9db2ae"), ("gold", "#1f6f68")]:
        mask = meta.split == split
        axes[0].scatter(meta.tsne_x[mask], meta.tsne_y[mask], s=14,
                        c=color, label=split, alpha=0.75)
    axes[0].set_title("split (pool vs gold)")
    axes[0].legend()
    scatter = axes[1].scatter(meta.tsne_x, meta.tsne_y, s=14,
                              c=meta.Effusion, cmap="viridis")
    axes[1].set_title("Effusion label")
    plt.colorbar(scatter, ax=axes[1])
    scatter = axes[2].scatter(meta.tsne_x, meta.tsne_y, s=14,
                              c=meta.n_series, cmap="plasma")
    axes[2].set_title("number of series (protocol)")
    plt.colorbar(scatter, ax=axes[2])
    fig.suptitle("Case-DINO knee MRI embeddings, one mean-pooled vector per case")
    fig.tight_layout()
    fig.savefig(WORK / "knee_dino_embedding_scatter.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    results = {
        "label_knn": label_results,
        "protocol_knn": protocol_results,
        "gold_auc_from_pool_knn": gold_auc,
        "macro_label_knn_r2": float(np.mean([
            value["r2_knn_neighbors"] for value in label_results.values()
        ])),
        "macro_gold_auc_from_pool_knn": float(np.mean(list(gold_auc.values()))),
    }
    return results, meta


def main():
    print(f"COMP_ROOT={COMP_ROOT}")
    print(f"LABELS_ROOT={LABELS_ROOT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}")

    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv").copy()
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")
    pool_sample = pool.sample(n=min(N_POOL, len(pool)), random_state=SEED).copy()
    gold["split"] = "gold"
    pool_sample["split"] = "pool"
    studies = pd.concat([
        gold[["StudyInstanceUID", *LABELS, "split"]],
        pool_sample[["StudyInstanceUID", *LABELS, "split"]],
    ], ignore_index=True)
    series_df = pd.read_csv(COMP_ROOT / "train_series.csv")
    print(f"pilot cases={len(studies)} ({len(gold)} gold + {len(pool_sample)} pool)")
    print("Labels are not accessed by the self-supervised training loop.")

    encoder, train_ds, history = train_ssl(
        studies[["StudyInstanceUID"]], series_df, device
    )
    embeddings, meta = embed_studies(encoder, studies, series_df, device)
    np.save(WORK / "knee_dino_study_embeddings.npy", embeddings)
    results, meta = evaluate_embeddings(embeddings, meta)
    meta.to_csv(WORK / "knee_dino_study_embeddings_meta.csv", index=False)
    results["training"] = {
        "cases": len(studies),
        "epochs_run": len(history),
        "pair_mode": PAIR_MODE,
        "single_series_fallback_cases": train_ds.single_series_cases,
        "unusable_series": train_ds.bad_series,
        "no_same_plane_pair_cases": train_ds.no_same_plane_pair_cases,
        "no_different_plane_pair_cases": train_ds.no_different_plane_pair_cases,
        "final_loss": history[-1]["loss"] if history else None,
    }
    with open(WORK / "knee_dino_summary.json", "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))
    print("KNEE_DINO_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
