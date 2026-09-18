"""arm4 (v2): latent case model -- slices as tokens, case latents, finding queries.

v1 diverged (see the training-config comments below for the diagnosis and the
three changes). Architecture is unchanged from v1 and is NOT what failed --
v1 reached 0.6385 gold macro-AUC at epoch 1 while its LR was still only ~1e-5,
i.e. while barely training at all. It was never run at a sane learning rate.


The architecture change arm1-arm3 never made. Those embed each slice in total
isolation and average them; the "attention" there is a per-slice saliency
score computed from that slice alone, so no slice ever sees another. Here:

  all slices -> shared CNN -> tokens (+ slot & depth embeddings)
             -> self-attention   (slices read each other)
             -> case latents     (cross-attend + self-attend, xN blocks)
             -> finding queries  (12 queries cross-attend into the latents)
             -> 12 logits

Everything after the CNN happens in latent space over the whole case at once.

Deliberate properties:
  * Variable N is native. Slice count per case varies 80-400; tokens + a
    key_padding_mask handle that directly, so the rigid 5-slot zero-padded
    grid arm2/arm3 needed is gone. The same code runs on a 30-slice cache or
    a 400-slice one with no change.
  * GroupNorm, not BatchNorm. One case per forward pass means BN would
    normalize across that case's own slices -- leaking case statistics and
    diverging train vs eval. We already lost a debugging round to a BN
    train/eval mismatch in the knee-DINO work; not repeating it.
  * CNN runs in chunks. Slices are encoded independently, so the backbone is
    fed CHUNK at a time and the tokens concatenated. Tokens are tiny
    (400 x 512 floats = 800 KB), so the transformer still sees the whole case
    at once. This is what keeps high slice counts (and later, high
    resolution) inside GPU memory.

Reads the all-slice cache (kaggle/prep_v2, ragged: concatenated slices +
per-case offsets) if attached, else the 30-slice cache (kaggle/prep), else
falls back to reading DICOM directly. Eval is the same 58-study gold set as
every previous arm, so the number is directly comparable to arm3 and to the
0.715 already on the leaderboard.
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]
N_FINDINGS = 12

# --- architecture ---
D_MODEL = 384
N_SELF_ATTN_LAYERS = 2
N_LATENTS = 32
N_LATENT_BLOCKS = 3
N_HEADS = 8
DROPOUT = 0.15
CNN_CHUNK = 64          # slices per backbone forward; tokens are concatenated

# --- data ---
MAX_SLICES = 160        # per case per step; sampled if the case has more
IMG_SIZE = 160

# --- training ---
# v2, after v1 diverged: train loss fell to 0.6006 by epoch 2 then rose to
# 0.6074 by epoch 5 while gold AUC collapsed 0.6385 -> 0.5343 (near chance).
# Loss going back UP is divergence, not overfitting -- and it coincided
# exactly with LR reaching peak as warmup finished (~epoch 4). Three changes:
#   1. LR 3e-4 -> 1e-4. 3e-4 was inherited from arm1-arm3, where it drove a
#      plain ResNet + one linear head. A 32.7M-param transformer trained from
#      scratch on 4,349 cases is a very different optimisation problem.
#   2. EPOCHS 40 -> 16. This sets the COSINE HORIZON, and the time budget only
#      ever allowed ~17 epochs -- so the schedule never annealed, it ramped to
#      peak and stayed there. 16 makes warmup ~1.5 epochs (was 4) and lets the
#      cosine actually come back down across the run that really happens.
#   3. Effective batch 16 -> 32 cases. Free stabilisation; costs no memory
#      since it's accumulation, not a bigger forward pass.
BATCH_CASES = 2
ACCUM_STEPS = 16        # effective batch = 32 cases
EPOCHS = 16             # also the cosine horizon -- keep near what fits
LR = 1e-4
BACKBONE_LR_MULT = 0.25
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
WARMUP_FRAC = 0.1
TIME_BUDGET_S = 5.5 * 3600
SEED = 42

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
COMP_ROOT = find_root(INPUT_ROOT, "train_series.csv")
RESUME_ROOT = find_root(INPUT_ROOT, "arm4_checkpoint.pt")
WORK = Path("/kaggle/working")

# cache discovery: all-slice shards first, then the 30-slice cache.
# Keep the shard NUMBER with its directory -- if only one shard is attached
# (or one shard's build failed), positional indexing would look for
# index_shard0.json inside the shard-1 directory and crash.
SHARDS = [(s, r) for s, r in ((s, find_root(INPUT_ROOT, f"index_shard{s}.json"))
                              for s in (0, 1)) if r]
CACHE30_ROOT = find_root(INPUT_ROOT, "study_index.json")
print(f"LABELS_ROOT={LABELS_ROOT}")
print(f"all-slice shards found: {len(SHARDS)} -> {SHARDS}")
print(f"30-slice cache: {CACHE30_ROOT}")
print(f"RESUME_ROOT={RESUME_ROOT}  ({'resuming' if RESUME_ROOT else 'fresh start'})")


class CaseCache:
    """Ragged all-slice cache: concatenated slices + per-case offsets.

    Loads the shards memory-mapped -- ~16 GB across both shards is more than
    we want resident, and mmap lets the OS page in only the cases a batch
    actually touches.
    """

    def __init__(self, shards):
        """shards: list of (shard_number, directory) -- the number comes from
        discovery, not list position, so a single attached shard still works."""
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
        self.slot_order = self.indexes[0]["slot_order"]

    def __contains__(self, uid):
        return uid in self.uid_to_loc

    def get(self, uid):
        si, ci = self.uid_to_loc[uid]
        idx, arr = self.indexes[si], self.arrays[si]
        lo, hi = idx["offsets"][ci], idx["offsets"][ci + 1]
        slices = np.asarray(arr[lo:hi], dtype=np.uint8)
        slots = np.array(idx["slot_of_slice"][lo:hi], np.int64)
        depth = np.array(idx["depth_frac_of_slice"][lo:hi], np.float32)
        return slices, slots, depth


class Cache30:
    """Fixed 5-slot x 6-slice cache (kaggle/prep). Slot/depth derived from the
    fixed grid, so the same token construction applies."""

    def __init__(self, root):
        idx = json.load(open(root / "study_index.json"))
        self.pixels = np.load(root / "pixels.npy", mmap_mode="r")
        self.masks = np.load(root / "mask.npy", mmap_mode="r")
        self.uid_to_row = {uid: i for i, uid in enumerate(idx["uids"])}
        self.slots = idx["slots"]
        self.sps = idx["slices_per_slot"]
        self.slot_order = self.slots
        print(f"  30-slice cache: {self.pixels.shape}")

    def __contains__(self, uid):
        return uid in self.uid_to_row

    def get(self, uid):
        r = self.uid_to_row[uid]
        px = np.asarray(self.pixels[r], dtype=np.uint8)
        mk = np.asarray(self.masks[r]).astype(bool)
        slot_ids = np.repeat(np.arange(len(self.slots)), self.sps)
        depth = np.tile(np.linspace(0, 1, self.sps, dtype=np.float32), len(self.slots))
        return px[mk], slot_ids[mk], depth[mk]


class CaseDataset(Dataset):
    def __init__(self, studies_df, cache, train: bool):
        self.studies = studies_df[studies_df.StudyInstanceUID.isin(
            [u for u in studies_df.StudyInstanceUID if u in cache])].reset_index(drop=True)
        self.cache = cache
        self.train = train
        dropped = len(studies_df) - len(self.studies)
        if dropped:
            print(f"  {dropped} studies missing from cache, skipped")

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        row = self.studies.iloc[i]
        slices, slots, depth = self.cache.get(row["StudyInstanceUID"])
        n = len(slices)
        if n > MAX_SLICES:
            # Random subset while training (sees every slice across epochs, and
            # doubles as augmentation); evenly spaced at eval for determinism.
            if self.train:
                keep = np.sort(np.random.choice(n, MAX_SLICES, replace=False))
            else:
                keep = np.linspace(0, n - 1, MAX_SLICES).round().astype(int)
            slices, slots, depth = slices[keep], slots[keep], depth[keep]

        x = torch.from_numpy(slices.astype(np.float32) / 255.0).unsqueeze(1)
        return {
            "slices": x,
            "slots": torch.from_numpy(slots),
            "depth": torch.from_numpy(depth),
            "y": torch.tensor([row[c] for c in LABELS], dtype=torch.float32),
        }


def collate(batch):
    """Pad to the longest case in the batch; mask marks the real tokens."""
    n_max = max(b["slices"].shape[0] for b in batch)
    B = len(batch)
    slices = torch.zeros(B, n_max, 1, IMG_SIZE, IMG_SIZE)
    slots = torch.zeros(B, n_max, dtype=torch.long)
    depth = torch.zeros(B, n_max)
    pad_mask = torch.ones(B, n_max, dtype=torch.bool)   # True = padding
    for i, b in enumerate(batch):
        n = b["slices"].shape[0]
        slices[i, :n] = b["slices"]
        slots[i, :n] = b["slots"]
        depth[i, :n] = b["depth"]
        pad_mask[i, :n] = False
    return {"slices": slices, "slots": slots, "depth": depth,
            "pad_mask": pad_mask, "y": torch.stack([b["y"] for b in batch])}


def gn_resnet34(num_groups=32):
    """ResNet34 with GroupNorm -- batch-size independent, which matters when a
    'batch' is one case's slices."""
    from torchvision.models import resnet34
    net = resnet34(weights=None, norm_layer=lambda c: nn.GroupNorm(num_groups, c))
    net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    net.fc = nn.Identity()
    return net


class LatentCaseModel(nn.Module):
    def __init__(self, n_slots=8):
        super().__init__()
        self.backbone = gn_resnet34()
        self.proj = nn.Linear(512, D_MODEL)
        self.slot_emb = nn.Embedding(n_slots, D_MODEL)
        self.depth_proj = nn.Linear(1, D_MODEL)
        self.token_norm = nn.LayerNorm(D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=D_MODEL * 4, dropout=DROPOUT,
            batch_first=True, norm_first=True, activation="gelu")
        self.slice_encoder = nn.TransformerEncoder(enc_layer, N_SELF_ATTN_LAYERS)

        self.latents = nn.Parameter(torch.randn(N_LATENTS, D_MODEL) * 0.02)
        self.latent_cross = nn.ModuleList(
            [nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True)
             for _ in range(N_LATENT_BLOCKS)])
        self.latent_self = nn.ModuleList(
            [nn.TransformerEncoderLayer(D_MODEL, N_HEADS, dim_feedforward=D_MODEL * 4,
                                        dropout=DROPOUT, batch_first=True,
                                        norm_first=True, activation="gelu")
             for _ in range(N_LATENT_BLOCKS)])
        self.latent_norms = nn.ModuleList(
            [nn.LayerNorm(D_MODEL) for _ in range(N_LATENT_BLOCKS)])

        self.finding_queries = nn.Parameter(torch.randn(N_FINDINGS, D_MODEL) * 0.02)
        self.finding_cross = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT,
                                                   batch_first=True)
        self.finding_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 1)

    def encode_slices(self, slices):
        """CNN in chunks -- slices are independent, so this bounds memory
        without changing the result."""
        B, N = slices.shape[:2]
        flat = slices.reshape(B * N, *slices.shape[2:])
        feats = []
        for i in range(0, flat.shape[0], CNN_CHUNK):
            feats.append(self.backbone(flat[i:i + CNN_CHUNK]))
        return torch.cat(feats, 0).reshape(B, N, -1)

    def forward(self, slices, slots, depth, pad_mask):
        B, N = slices.shape[:2]
        tok = self.proj(self.encode_slices(slices))
        tok = tok + self.slot_emb(slots) + self.depth_proj(depth.unsqueeze(-1))
        tok = self.token_norm(tok)

        tok = self.slice_encoder(tok, src_key_padding_mask=pad_mask)

        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        for cross, selfattn, norm in zip(self.latent_cross, self.latent_self, self.latent_norms):
            attended, _ = cross(lat, tok, tok, key_padding_mask=pad_mask)
            lat = norm(lat + attended)
            lat = selfattn(lat)

        q = self.finding_queries.unsqueeze(0).expand(B, -1, -1)
        found, _ = self.finding_cross(q, lat, lat)
        found = self.finding_norm(found)
        return self.head(found).squeeze(-1)          # (B, 12)


def macro_auc(y_true, y_pred):
    per = []
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        per.append(roc_auc_score(yj, y_pred[:, j]) if len(np.unique(yj)) > 1 else np.nan)
    return float(np.nanmean(per)), per


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if SHARDS:
        cache = CaseCache(SHARDS)
        print(f"using ALL-SLICE cache ({len(SHARDS)} shard(s))")
    elif CACHE30_ROOT:
        cache = Cache30(CACHE30_ROOT)
        print("using 30-slice cache")
    else:
        raise FileNotFoundError("no cache attached -- attach the all-slice shards "
                                "or the 30-slice cache")

    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv")
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")
    print(f"gold_test: {len(gold)}; train_pool: {len(pool)}")

    train_ds = CaseDataset(pool, cache, train=True)
    val_ds = CaseDataset(gold, cache, train=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_CASES, shuffle=True,
                          num_workers=2, collate_fn=collate, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_CASES, shuffle=False,
                        num_workers=2, collate_fn=collate)

    n_slots = len(cache.slot_order)
    model = LatentCaseModel(n_slots=n_slots).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params, {n_slots} slot types, "
          f"d={D_MODEL}, {N_LATENTS} latents x {N_LATENT_BLOCKS} blocks")

    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR * BACKBONE_LR_MULT},
        {"params": other_params, "lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()

    start_epoch, best_macro, history, gstep = 0, -1.0, [], 0
    if RESUME_ROOT is not None:
        ck = torch.load(RESUME_ROOT / "arm4_checkpoint.pt", map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, best_macro = ck["epoch"] + 1, ck["best_macro"]
        history, gstep = ck["history"], ck.get("gstep", 0)
        print(f"resumed at epoch {start_epoch}, best {best_macro:.4f}")

    steps_per_epoch = max(len(train_dl) // ACCUM_STEPS, 1)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(int(WARMUP_FRAC * total_steps), 1)

    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def evaluate():
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for b in val_dl:
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    out = model(b["slices"].to(device), b["slots"].to(device),
                                b["depth"].to(device), b["pad_mask"].to(device))
                preds.append(torch.sigmoid(out.float()).cpu().numpy())
                ys.append(b["y"].numpy())
        return macro_auc(np.concatenate(ys), np.concatenate(preds))

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"time budget reached before epoch {epoch}")
            break
        model.train()
        tot, nb, ep_t0, last_print = 0.0, 0, time.time(), time.time()
        opt.zero_grad(set_to_none=True)

        for bi, b in enumerate(train_dl):
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(b["slices"].to(device), b["slots"].to(device),
                            b["depth"].to(device), b["pad_mask"].to(device))
                y = b["y"].to(device)
                target = y * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
                loss = lossf(out.float(), target) / ACCUM_STEPS
            scaler.scale(loss).backward()
            tot += float(loss.detach()) * ACCUM_STEPS
            nb += 1

            if (bi + 1) % ACCUM_STEPS == 0:
                for gi, g in enumerate(opt.param_groups):
                    g["lr"] = base_lrs[gi] * lr_scale(gstep)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                gstep += 1

            if time.time() - last_print > 30:
                el = time.time() - ep_t0
                print(f"  epoch {epoch}  batch {nb}/{len(train_dl)}  "
                      f"loss {tot / nb:.4f}  lr {opt.param_groups[-1]['lr']:.2e}  "
                      f"{el:.0f}s  eta {(len(train_dl) - nb) * el / nb / 60:.1f} min",
                      flush=True)
                last_print = time.time()
            if time.time() - t0 > TIME_BUDGET_S:
                break

        macro, per = evaluate()
        star = ""
        if macro > best_macro:
            best_macro, star = macro, "  <- best"
            torch.save(model.state_dict(), WORK / "arm4_best.pt")
            with open(WORK / "arm4_best_per_finding.json", "w") as f:
                json.dump(dict(zip(LABELS, per)), f, indent=2)
        history.append({"epoch": epoch, "loss": tot / max(nb, 1), "gold_macro_auc": macro,
                        "minutes": (time.time() - ep_t0) / 60})
        print(f"epoch {epoch}  loss {tot / max(nb,1):.4f}  gold macro-AUC {macro:.4f}{star}"
              f"  ({(time.time()-ep_t0)/60:.1f} min, total {(time.time()-t0)/60:.0f} min)",
              flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best_macro": best_macro, "history": history, "gstep": gstep},
                   WORK / "arm4_checkpoint.pt")

    with open(WORK / "arm4_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDONE. best gold macro-AUC {best_macro:.4f} over {len(history)} epochs.")
    print("Baselines: arm2 scratch 0.6769 gold / 0.715 LB; arm3 running.")


if __name__ == "__main__":
    main()
