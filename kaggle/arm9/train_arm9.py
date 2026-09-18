"""arm9-B: arm6's model, slot-structured input. Signal check, not a full run.

WHAT CHANGES vs arm6, and nothing else changes:

  arm6   bag = 32 slices drawn from the flat concatenation of every series in
         the study. Selection is linspace/random over that flat list, so a
         series contributes in proportion to how many slices the scanner
         happened to acquire. A 200-slice axial outweighs a 40-slice sagittal
         5:1 -- an acquisition artefact, not anatomy. A study missing its
         coronal series is indistinguishable from one that merely has few
         coronal slices.

  arm9   bag = 6 SLOTS (plane x fluid-sensitive), each pooled from G slices
         sampled inside that slot. Every slot weighs the same regardless of
         acquisition length, and an absent slot is stated explicitly through
         the presence mask rather than silently under-represented.

Backbone, attention MLP and heads are byte-identical to arm6, so this
warm-starts from arm6_best.pt at 0 missing / 0 unexpected and the run is
measuring the input representation alone.

WHY THE CACHE NEEDS NO REBUILD. build_all_slice_cache.py already stored
`slot_of_slice` (index into SLOT_ORDER) and `depth_frac_of_slice` per slice.
Checked locally against train_series.csv: Fluid_Sensitive and Fat_Suppression
agree on all 24,371 rows, so plane x fluid IS the full 6-slot scheme and there
is no third dimension to recover. Slot 6 ("Other") never occurs.

Slot presence over the 4,407 cached studies -- the mask carries real signal,
it is not decoration:
    Axial_FS 100.0%   Sagittal_nonFS 96.8%   Coronal_FS 96.4%
    Sagittal_FS 94.2% Coronal_nonFS  77.3%   Axial_nonFS 19.4%
Mean 4.9 of 6 slots per study.

THE arm8 TRAP, AND WHY THIS IS NOT IT. arm8 restricted the slice pool to a
depth band and lost 0.0175: arm6's wide random sampling was acting as our only
regulariser, and narrowing it caused overfitting (train loss 0.4184 -> 0.3654).
Here the slot structure is fixed but its CONTENTS are resampled every epoch,
so per-epoch diversity survives. That is the difference between adding
structure and merely deleting choices.

CONFOUND, STATED. The 0.20-0.80 within-slot depth band is part of the recipe
being tested, so this run changes representation AND band together. Within a
slot the band still leaves many distinct draws (a 40-slice series keeps 24,
choose 6), so it does not reproduce arm8's collapse -- but if arm9 wins, the
band and the slot structure still need separating by ablation before either
gets credit.

BUDGET. ~0.5h of GPU quota remains this period. This is a go/no-go probe:
evaluate once at epoch -1 (warm start, zero adaptation) to see whether the
restructured input preserves what arm6 learned, then train as many epochs as
the budget allows. Judge against arm6's 0.8449 plateau, not its epoch-3 0.8161
-- that number came from a cold ImageNet init and is not the comparator here.
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
EMBED_DIM = 2048

N_SLOTS = 6                # SLOT_ORDER[:6]; index 6 ("Other") never occurs
SLICES_PER_SLOT = 6        # 6x6 = 36 slices/case, ~ arm6's 32 -> comparable cost
IMG_SIZE = 160
DEPTH_BAND = (0.20, 0.80)  # applied WITHIN a slot; see CONFOUND above
BATCH_CASES = 4
ACCUM_STEPS = 4
EPOCHS = 4                 # also the cosine horizon -- sized to actually finish
LR = 1e-4                  # warm start: adapt arm6, don't overwrite it
BACKBONE_LR_MULT = 0.1
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
WARMUP_FRAC = 0.1
TIME_BUDGET_S = 0.42 * 3600   # hard stop well inside the remaining quota
SEED = 42

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
WEIGHTS_ROOT = find_root(INPUT_ROOT, "resnet50-0676ba61.pth")
ARM6_ROOT = find_root(INPUT_ROOT, "arm6_best.pt")
SHARDS = [(s, r) for s, r in ((s, find_root(INPUT_ROOT, f"index_shard{s}.json"))
                              for s in (0, 1)) if r]
WORK = Path("/kaggle/working")
print(f"LABELS_ROOT={LABELS_ROOT}")
print(f"WEIGHTS_ROOT={WEIGHTS_ROOT}")
print(f"ARM6_ROOT={ARM6_ROOT}")
print(f"all-slice shards: {len(SHARDS)}")


class CaseCache:
    """Same ragged cache as arm6, but also serves per-slice slot and depth.

    The index stores those as flat Python lists over the whole shard; they are
    converted to arrays once here so __getitem__ does not pay for it 4,349
    times an epoch.
    """

    def __init__(self, shards):
        self.uid_to_loc = {}
        self.arrays, self.indexes = [], []
        for pos, (shard_no, root) in enumerate(shards):
            idx = json.load(open(root / f"index_shard{shard_no}.json"))
            arr = np.load(root / f"slices_shard{shard_no}.npy", mmap_mode="r")
            idx["slot_arr"] = np.asarray(idx["slot_of_slice"], dtype=np.int16)
            idx["depth_arr"] = np.asarray(idx["depth_frac_of_slice"], dtype=np.float32)
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
        return arr, lo, hi, idx["slot_arr"][lo:hi], idx["depth_arr"][lo:hi]


class SlotDataset(Dataset):
    def __init__(self, studies_df, cache, train: bool):
        keep = [u for u in studies_df.StudyInstanceUID if u in cache]
        self.studies = studies_df[studies_df.StudyInstanceUID.isin(keep)].reset_index(drop=True)
        self.cache, self.train = cache, train
        dropped = len(studies_df) - len(self.studies)
        if dropped:
            print(f"  {dropped} studies missing from cache, skipped")

    def __len__(self):
        return len(self.studies)

    def _pick(self, local_idx, depth):
        """Choose SLICES_PER_SLOT positions out of one slot's slices.

        Band first, but only if the band leaves enough to choose from -- a
        short series must not be emptied by it. Random while training so the
        slot's contents vary every epoch (this is what keeps arm6's sampling
        augmentation alive); evenly spaced at eval for determinism.
        """
        band = local_idx
        if DEPTH_BAND is not None:
            lo, hi = DEPTH_BAND
            inb = local_idx[(depth >= lo) & (depth <= hi)]
            if len(inb) >= SLICES_PER_SLOT:
                band = inb
        n = len(band)
        if n == 0:
            return None
        if n < SLICES_PER_SLOT:
            return band[np.concatenate([np.arange(n),
                                        np.full(SLICES_PER_SLOT - n, n - 1)])]
        if self.train:
            return np.sort(np.random.choice(band, SLICES_PER_SLOT, replace=False))
        return band[np.linspace(0, n - 1, SLICES_PER_SLOT).round().astype(int)]

    def __getitem__(self, i):
        row = self.studies.iloc[i]
        arr, lo, hi, slots, depth = self.cache.get(row["StudyInstanceUID"])

        bag = np.zeros((N_SLOTS, SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE), np.float32)
        slot_mask = np.zeros(N_SLOTS, np.float32)
        for s in range(N_SLOTS):
            local = np.flatnonzero(slots == s)
            pick = self._pick(local, depth[local]) if len(local) else None
            if pick is None:
                continue
            # fancy-index the memmap in one absolute, sorted read
            absolute = np.sort(lo + pick)
            bag[s] = np.asarray(arr[absolute], dtype=np.uint8).astype(np.float32) / 255.0
            slot_mask[s] = 1.0

        return {"bag": torch.from_numpy(bag).unsqueeze(2),      # (S, G, 1, H, W)
                "mask": torch.from_numpy(slot_mask),            # (S,)
                "y": torch.tensor([row[c] for c in LABELS], dtype=torch.float32)}


def load_imagenet_backbone():
    """Full resnet50 state_dict -> load into the FULL net, then strip.

    Loading a full state_dict straight into a stripped Sequential matches
    nothing (conv1.weight vs 0.weight) and silently trains from scratch.
    """
    net = resnet50(weights=None)
    sd = torch.load(WEIGHTS_ROOT / "resnet50-0676ba61.pth", map_location="cpu",
                    weights_only=False)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"ImageNet init: {len(missing)} missing, {len(unexpected)} unexpected")
    return nn.Sequential(*list(net.children())[:-2])


class SlotAttentionMIL(nn.Module):
    """arm6's modules exactly; only the pooling hierarchy is new.

    Module names and shapes are unchanged (backbone / att / heads), which is
    what lets arm6_best.pt load 0/0. The within-slot mean introduces no
    parameters, so nothing here is randomly initialised and the warm start
    carries the whole model, not just its features.
    """

    def __init__(self, n_findings=N_FINDINGS, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = load_imagenet_backbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(nn.Linear(embed_dim, 256), nn.Tanh(),
                                 nn.Linear(256, n_findings))
        self.heads = nn.Linear(embed_dim, n_findings)

    def forward(self, bag, mask):
        B, S, G, C, H, W = bag.shape
        x = bag.reshape(B * S * G, C, H, W).repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        feats = self.pool(self.backbone(x)).flatten(1).reshape(B, S, G, -1)

        # within-slot pool: G slices -> one slot embedding
        slot_feats = feats.mean(dim=2)                       # (B, S, D)

        # arm6's per-finding attention, now over slots instead of raw slices
        att = self.att(slot_feats).masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        w = torch.nan_to_num(torch.softmax(att, dim=1))
        pooled = torch.einsum("bsn,bsd->bnd", w, slot_feats)
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
    if WEIGHTS_ROOT is None:
        raise FileNotFoundError("torchvision resnet50 weights not attached")
    if ARM6_ROOT is None:
        raise FileNotFoundError("arm6_best.pt not attached -- warm start is the point")

    cache = CaseCache(SHARDS)
    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv")
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")
    print(f"gold_test: {len(gold)}; train_pool: {len(pool)}; "
          f"{N_SLOTS} slots x {SLICES_PER_SLOT} = {N_SLOTS * SLICES_PER_SLOT} slices/case")

    train_dl = DataLoader(SlotDataset(pool, cache, True), batch_size=BATCH_CASES,
                          shuffle=True, num_workers=2, drop_last=True)
    val_dl = DataLoader(SlotDataset(gold, cache, False), batch_size=BATCH_CASES,
                        shuffle=False, num_workers=2)

    model = SlotAttentionMIL().to(device)
    sd = torch.load(ARM6_ROOT / "arm6_best.pt", map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"arm6 warm start: {len(missing)} missing, {len(unexpected)} unexpected "
          f"(expect 0/0 -- anything else means the modules drifted)")
    if missing or unexpected:
        raise RuntimeError(f"warm start mismatch: {missing[:5]} {unexpected[:5]}")

    bb_ids = {id(p) for p in model.backbone.parameters()}
    opt = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": LR * BACKBONE_LR_MULT},
        {"params": [p for p in model.parameters() if id(p) not in bb_ids], "lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()

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

    # Epoch -1: arm6's weights read through slot-structured input, zero
    # adaptation. This is the control the budget can afford -- it separates
    # "the restructured input destroyed arm6's features" from "the model just
    # needs a few steps to adjust to a new bag shape".
    t0 = time.time()
    macro0, per0 = evaluate()
    print(f"\nepoch -1 (warm start, NO adaptation)  gold macro-AUC {macro0:.4f}"
          f"   [same weights, arm6 flat 32-slice input = 0.8449]", flush=True)
    for lb, v in zip(LABELS, per0):
        print(f"    {lb:18s} {v:.4f}")

    best_macro, history, gstep = macro0, [], 0
    torch.save(model.state_dict(), WORK / "arm9_best.pt")

    for epoch in range(EPOCHS):
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
                print("  time budget hit mid-epoch, evaluating what we have")
                break

        macro, per = evaluate()
        star = ""
        if macro > best_macro:
            best_macro, star = macro, "  <- best"
            torch.save(model.state_dict(), WORK / "arm9_best.pt")
            with open(WORK / "arm9_best_per_finding.json", "w") as f:
                json.dump(dict(zip(LABELS, per)), f, indent=2)
        history.append({"epoch": epoch, "loss": tot / max(nb, 1),
                        "gold_macro_auc": macro,
                        "minutes": (time.time() - ep_t0) / 60})
        print(f"epoch {epoch}  loss {tot/max(nb,1):.4f}  gold macro-AUC {macro:.4f}"
              f"{star}  ({(time.time()-ep_t0)/60:.1f} min, "
              f"total {(time.time()-t0)/60:.0f} min)", flush=True)

    with open(WORK / "arm9_history.json", "w") as f:
        json.dump({"epoch_minus_1": macro0,
                   "per_finding_minus_1": dict(zip(LABELS, per0)),
                   "history": history}, f, indent=2)
    print(f"\nDONE. warm-start {macro0:.4f}, best {best_macro:.4f} "
          f"over {len(history)} epoch(s).")
    print("Verdict line: arm6 flat-input plateau = 0.8449. Above it, slot")
    print("structure earns the next full run; below it, the gap is DINOv2 +")
    print("resolution and we spend the refresh there instead.")


if __name__ == "__main__":
    main()
