"""Measure WHICH slices arm6 actually finds useful, before we cache only 48.

Motivation: to train at 224px we must shrink the cache from ~177 slices/case
to ~48, and those 48 should be the informative ones. arm6 currently selects
with linspace over the concatenated slice list, which (a) includes the
extreme first/last slices of each series -- largely soft tissue and air
outside the joint, which the community's pipeline explicitly trimmed to the
6-94% span -- and (b) weights each series proportionally to how many slices
the scanner happened to acquire, which is arbitrary rather than principled.

Rather than guess a better rule, measure one. arm6's attention head scores
each slice's usefulness directly, so run it over EVERY slice and record where
the signal concentrates.

Two design points that matter:

  * Record raw attention LOGITS, not softmax weights. arm6 was trained with
    K=32; feeding it ~177 slices would renormalise the softmax over a
    different K and make values incomparable across cases. The attention MLP
    is applied per-slice independently, so the pre-softmax logit is a
    K-independent measure of "how much this slice is worth attending to".

  * No label leakage. arm6 was trained on train_pool only and has never seen
    a gold label, so using its attention to choose slices keeps gold_test a
    valid holdout. The same rule is computable at inference on the hidden
    test set, where no labels exist.

Outputs a profile of attention vs normalised depth, vs slot/plane, and per
finding -- enough to design the 48-slice rule from evidence.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]
N_FINDINGS = 12
BACKBONE_DIM = 2048
IMG_SIZE = 160          # profile at arm6's native resolution
CHUNK = 64              # slices per backbone forward
N_CASES = 1500          # enough for a distribution; keeps this ~10 min
DEPTH_BINS = 20
SEED = 42

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

np.random.seed(SEED)
torch.manual_seed(SEED)


def find_root(input_root: Path, marker: str, skip_dirs=("train_series", "test_series")):
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{marker!r} not found under {input_root}")


INPUT_ROOT = Path("/kaggle/input")
LABELS_ROOT = find_root(INPUT_ROOT, "gold_test.csv")
MODEL_ROOT = find_root(INPUT_ROOT, "arm6_best.pt")
SHARDS = []
for s in (0, 1):
    try:
        SHARDS.append((s, find_root(INPUT_ROOT, f"index_shard{s}.json")))
    except FileNotFoundError:
        pass
WORK = Path("/kaggle/working")
print(f"MODEL_ROOT={MODEL_ROOT}  shards={len(SHARDS)}")


class AttentionMIL(nn.Module):
    """Must mirror arm6 exactly so the checkpoint loads 0/0."""

    def __init__(self):
        super().__init__()
        net = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(nn.Linear(BACKBONE_DIM, 256), nn.Tanh(),
                                 nn.Linear(256, N_FINDINGS))
        self.heads = nn.Linear(BACKBONE_DIM, N_FINDINGS)

    @torch.no_grad()
    def slice_logits(self, slices, device):
        """(N,1,H,W) -> (N, 12) raw per-slice attention logits, chunked."""
        outs = []
        for i in range(0, slices.shape[0], CHUNK):
            x = slices[i:i + CHUNK].to(device).repeat(1, 3, 1, 1)
            x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                f = self.pool(self.backbone(x)).flatten(1)
                outs.append(self.att(f).float().cpu())
        return torch.cat(outs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if not SHARDS:
        raise FileNotFoundError("all-slice cache not attached")

    model = AttentionMIL().to(device)
    sd = torch.load(MODEL_ROOT / "arm6_best.pt", map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"checkpoint load: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing or unexpected:
        raise RuntimeError("arm6 checkpoint did not load cleanly")
    model.eval()

    # cache
    arrays, indexes, uid_to_loc = [], [], {}
    for pos, (shard_no, root) in enumerate(SHARDS):
        idx = json.load(open(root / f"index_shard{shard_no}.json"))
        arr = np.load(root / f"slices_shard{shard_no}.npy", mmap_mode="r")
        arrays.append(arr); indexes.append(idx)
        for ci, uid in enumerate(idx["uids"]):
            uid_to_loc[uid] = (pos, ci)
        print(f"  shard {shard_no}: {arr.shape}")
    slot_order = indexes[0]["slot_order"]

    # profile on TRAIN POOL only -- never gold, so gold stays a clean holdout
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")
    uids = [u for u in pool.StudyInstanceUID if u in uid_to_loc]
    uids = list(np.random.choice(uids, min(N_CASES, len(uids)), replace=False))
    print(f"profiling {len(uids)} pool cases (gold never touched)")

    depth_sum = np.zeros(DEPTH_BINS); depth_cnt = np.zeros(DEPTH_BINS)
    depth_sum_pf = np.zeros((N_FINDINGS, DEPTH_BINS))
    slot_sum = defaultdict(float); slot_cnt = defaultdict(int)
    # how concentrated is attention? fraction of total softmax mass in top-k
    topk_mass = {8: [], 16: [], 32: [], 48: []}
    n_slices_all = []

    for i, uid in enumerate(uids):
        pos, ci = uid_to_loc[uid]
        idx, arr = indexes[pos], arrays[pos]
        lo, hi = idx["offsets"][ci], idx["offsets"][ci + 1]
        n = hi - lo
        if n == 0:
            continue
        n_slices_all.append(n)
        px = np.asarray(arr[lo:hi], dtype=np.uint8).astype(np.float32) / 255.0
        slices = torch.from_numpy(px).unsqueeze(1)
        logits = model.slice_logits(slices, device).numpy()      # (n, 12)

        slots = np.array(idx["slot_of_slice"][lo:hi])
        depth = np.array(idx["depth_frac_of_slice"][lo:hi])

        # "useful for ANY finding" = max over the 12 attention channels
        best = logits.max(axis=1)
        # normalise within case so cases are comparable
        z = (best - best.mean()) / (best.std() + 1e-6)

        b = np.clip((depth * DEPTH_BINS).astype(int), 0, DEPTH_BINS - 1)
        for bi, zi in zip(b, z):
            depth_sum[bi] += zi; depth_cnt[bi] += 1
        for j in range(N_FINDINGS):
            zj = (logits[:, j] - logits[:, j].mean()) / (logits[:, j].std() + 1e-6)
            for bi, v in zip(b, zj):
                depth_sum_pf[j, bi] += v
        for s, zi in zip(slots, z):
            slot_sum[int(s)] += zi; slot_cnt[int(s)] += 1

        # concentration: softmax over all slices, mass captured by top-k
        p = np.exp(best - best.max()); p /= p.sum()
        srt = np.sort(p)[::-1]
        for k in topk_mass:
            topk_mass[k].append(float(srt[:k].sum()))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(uids)}", flush=True)

    prof = {
        "n_cases": len(n_slices_all),
        "slices_per_case_mean": float(np.mean(n_slices_all)),
        "depth_profile": [float(depth_sum[k] / max(depth_cnt[k], 1))
                          for k in range(DEPTH_BINS)],
        "depth_profile_per_finding": {
            LABELS[j]: [float(depth_sum_pf[j, k] / max(depth_cnt[k], 1))
                        for k in range(DEPTH_BINS)] for j in range(N_FINDINGS)},
        "slot_mean_attention": {slot_order[s] if s < len(slot_order) else str(s):
                                float(slot_sum[s] / max(slot_cnt[s], 1))
                                for s in sorted(slot_sum)},
        "slot_counts": {slot_order[s] if s < len(slot_order) else str(s): slot_cnt[s]
                        for s in sorted(slot_cnt)},
        "topk_softmax_mass": {str(k): float(np.mean(v)) for k, v in topk_mass.items()},
    }
    with open(WORK / "attention_profile.json", "w") as f:
        json.dump(prof, f, indent=2)

    print("\n=== ATTENTION vs DEPTH (z-scored within case; >0 = above average) ===")
    for k in range(DEPTH_BINS):
        v = prof["depth_profile"][k]
        bar = "#" * int(abs(v) * 40)
        print(f"  {k/DEPTH_BINS:4.2f}-{(k+1)/DEPTH_BINS:4.2f}  {v:+.3f} {bar}")

    print("\n=== ATTENTION by SLOT ===")
    for s, v in sorted(prof["slot_mean_attention"].items(), key=lambda t: -t[1]):
        print(f"  {s:18s} {v:+.3f}   (n={prof['slot_counts'][s]})")

    print("\n=== CONCENTRATION: softmax mass captured by top-k slices ===")
    print(f"  mean slices/case: {prof['slices_per_case_mean']:.1f}")
    for k in sorted(topk_mass):
        print(f"  top-{k:3d}: {prof['topk_softmax_mass'][str(k)]:.3f}")
    print("\nIf top-48 captures most of the mass, a 48-slice cache loses little.")
    print("If mass is spread flat, attention is diffuse -- which would also")
    print("explain why arm7b and more-slices-per-case both failed.")


if __name__ == "__main__":
    main()
