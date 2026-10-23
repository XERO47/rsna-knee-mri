"""arm22: tell the crop model which side each window is a hypothesis for.

WHY. The ROI cache deliberately does NOT choose a side. Their laterality contract
(references/roi_cache/laterality_rules.md) measured the geometric "median DICOM patient-X
sign" -- the same method our orient.npz uses -- at 97.18% agreement on the 2,202
tag-decisive studies, and rejected it for crop selection: a ~3% error does not add noise,
it silently puts a medial/lateral crop on the wrong compartment. So for MCL, lateral
meniscus and lateral OA every study gets FOUR windows: two centred on the left hypothesis
and two on the right. Half are anatomically wrong for every study, and the pooler is
expected to suppress them.

arm21 pooled those four windows knowing nothing about which was which. That is the gap
this arm closes -- WITHOUT touching the cache and without ever choosing a crop by
laterality, so no mirror error is possible:

    side_match(window) = +1  window's hypothesis side == our geometric laterality
                         -1  window's hypothesis side != our geometric laterality
                          0  side-independent target (PF OA, synovitis), or laterality
                             unresolved, or the window is invalid
    attention_logit += gamma[target] * side_match          gamma initialised to ZERO

At initialisation gamma = 0, so arm22 IS arm21 -- the same zero-init discipline as arm14's
seq head and the teammate's ROI residuals. If the side signal is useful, gamma grows away
from zero and the model prefers windows on the side geometry believes in, while remaining
free to override on the ~3% where geometry is wrong. Nothing is masked or discarded.

The window's hypothesis side is read from the builder's own `candidate_laterality`
provenance column, not assumed from window ordering.

PRE-REGISTERED, fixed before the run. The mechanism predicts gains ONLY on the three
laterality-dependent targets; PF OA and synovitis are side-independent and act as built-in
CONTROLS that should not move.
  * PRIMARY: adopt iff mean fold-0 OOF AUC over MCL + Lateral Meniscus + Lateral OA
    improves by >= ADOPT_MARGIN = 0.002 over arm21's, on the same 870 studies.
  * CONTROL: PF OA and Synovitis reported; a large move there would mean the gain is not
    the side mechanism and the result should not be trusted.
  * SECONDARY: the arm21 blend test is repeated so the effect on the ensemble is visible.
  * The learned |gamma| per target is logged -- it says directly whether the model used
    the signal at all.

arm21 reference (fold-0 OOF, 870 studies): MCL .7435, Lateral Meniscus .8199,
Lateral OA .8196, PF OA .8251, Synovitis .8645; mean of the three lateral targets .7943.
"""
import os
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")   # device_count() must not init CUDA
os.environ.setdefault("NCCL_P2P_DISABLE", "1")                # P2P between cloud T4s can deadlock
os.environ.setdefault("NCCL_DEBUG", "WARN")

import faulthandler
import sys
# Periodic stack dumps ONLY in the parent (idle in join). arm14 v1 lost rank 1 to SIGSEGV
# 14.8 min after the job's processes started -- at the first 15-min dump, while a
# DataLoader queue thread was starting. The watchdog reads other threads' frames without
# the GIL, which can crash during thread creation/teardown. Spawned children import this
# module as __mp_main__, so they skip it and get only the fatal-signal handler below.
if __name__ == "__main__":
    faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)


def arm_child():
    """Children: traceback on a real crash (signal handler -- no watchdog thread), and a
    capped intra-op pool on a 4-vCPU box."""
    faulthandler.enable(file=sys.stderr, all_threads=True)
    torch.set_num_threads(2)


import datetime
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

TAG = "arm22"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]
# The five ROI targets, in roi_geometry.TARGET_SPECS order, and their LAB columns.
TARGETS = ["mcl", "lateral_meniscus", "lateral_oa", "pf_oa", "synovitis"]
TARGET_LAB = {"mcl": "MCL", "lateral_meniscus": "Lateral Meniscus", "lateral_oa": "Lateral OA",
              "pf_oa": "PF OA", "synovitis": "Synovitis"}
TARGET_IDX = np.array([LAB.index(TARGET_LAB[t]) for t in TARGETS])
WINDOWS_PER_TARGET = 4
CNN_CHUNK = 48
# v1 died on the first training forward, twice: batch 4 x 5 targets x 4 windows = 80
# ConvNeXt images at 336 per GPU per step, with no gradient checkpointing (the global
# models only ever pushed 48). Train on 2 of the 4 windows per target, sampled from the
# valid ones -- as the teammate's own Arm C trainer does -- which halves memory and compute;
# evaluation still scores all 4. Gradient checkpointing on as insurance.
ROI_K_TRAIN = int(os.environ.get("ARM21_ROI_K_TRAIN", "2"))
GRAD_CKPT = os.environ.get("ARM21_GRAD_CKPT", "1") == "1"
GLOBAL_REF = "arm20_f0_noaug"          # the global fold-0 comparator for the blend test
# Targets whose windows are paired left/right hypotheses. PF OA and synovitis are
# side-independent (overlapping triplets, not side candidates) and are the controls.
SIDE_TARGETS = ("mcl", "lateral_meniscus", "lateral_oa")
ARM21_OOF = {"MCL": 0.7435, "Lateral Meniscus": 0.8199, "Lateral OA": 0.8196,
             "PF OA": 0.8251, "Synovitis": 0.8645}

ARCH = os.environ.get("ARM10_ARCH", "convnext_small.in12k_ft_in1k_384")
PRETRAINED = os.environ.get("ARM10_PRETRAINED", "1") == "1"
RES = int(os.environ.get("ARM10_RES", "336"))
K_TRAIN = int(os.environ.get("ARM10_K", "12"))
BS_PER_GPU = int(os.environ.get("ARM10_BS", "4"))
EPOCHS = int(os.environ.get("ARM10_EPOCHS", "10"))
BB_LR, HEAD_LR, WD = 3e-5, 1e-3, 0.02
PCT_START, CLIP = 0.15, 3.0
GAIN_AUG = 0.20
# Matching the 0.941 public notebook's augmentation strengths.
ROT_DEG = float(os.environ.get("ARM19_ROT", "8.0"))
SCALE_JIT = float(os.environ.get("ARM19_SCALE", "0.08"))
SHIFT_JIT = float(os.environ.get("ARM19_SHIFT", "0.05"))
GAMMA_JIT = float(os.environ.get("ARM19_GAMMA", "0.20"))
SEED = 42
N_WORKERS = int(os.environ.get("ARM10_WORKERS", "2"))
PROGRESS_S = 30
START_METHOD = os.environ.get("ARM10_START", "spawn")

N_FOLDS = int(os.environ.get("ARM14_FOLDS", "5"))
FOLD_SEED = 42
SWA_LAST = 3
ADOPT_MARGIN = 0.002
GRU_HIDDEN = 256
# The extraction prior meaning "this brief report never covered the structure" -- masked.
# 0.025 (comprehensive report that called the knee normal) is deliberately KEPT: that one
# is real evidence of absence. See the header for the report-length/"normal" evidence.
MASK_PRIOR = 0.25
MASK_TOL = 1e-6

# evaluation specs: ("single", K) = one bag of K evenly spaced windows;
#                   ("subsets", K, R) = R phase-shifted evenly spaced bags of K, probs averaged
EVAL_SPEC = {
    "attn": {"epoch": ("single", 24), "oof": ("single", 24), "gold": ("single", 62)},
    "seq":  {"epoch": ("subsets", 12, 2), "oof": ("subsets", 12, 2), "gold": ("subsets", 12, 5)},
}
JOB_DEADLINE_S = float(os.environ.get("ARM14_JOB_DEADLINE_S", str(4.0 * 3600)))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
INPUT_ROOT = Path(os.environ.get("ARM10_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("ARM10_WORK", "/kaggle/working"))

RECIPE = {"name": "cache64-maxspan", "img": 336, "crop_mm": 140.0, "span": [0.02, 0.98],
          "slots": [["Sagittal", 1, 18], ["Sagittal", 0, 14], ["Coronal", 1, 12],
                    ["Coronal", 0, 8], ["Axial", -1, 12]]}


def log(msg, rank=0):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_file(name, skip=("train_series", "test_series")):
    for d, dirs, fs in os.walk(INPUT_ROOT):
        dirs[:] = [x for x in dirs if x not in skip]
        if name in fs:
            return Path(d) / name
    return None


# ------------------------------------------------------------------ corpus --
def need(name):
    p = find_file(name)
    if p is None:
        raise FileNotFoundError(f"{name} not attached")
    return p


class ROICorpus:
    """The Kaggle-rebuilt ROI shards as one view in cache64 row order: shard 0, then 1."""

    def __init__(self):
        self.a = {t: np.load(need(f"roi_{t}_shard0.npy"), mmap_mode="r") for t in TARGETS}
        self.b = {t: np.load(need(f"roi_{t}_shard1.npy"), mmap_mode="r") for t in TARGETS}
        self.n_a = self.a[TARGETS[0]].shape[0]

    def study(self, row):
        src, r = (self.a, row) if row < self.n_a else (self.b, row - self.n_a)
        return np.stack([np.asarray(src[t][r]) for t in TARGETS])       # (T, 4, 3, H, W)


def load_side_match(ids):
    """side_match (N, T, 4): +1 window hypothesis agrees with our geometric laterality,
    -1 disagrees, 0 unknown/side-independent. Uses the builder's own recorded
    `candidate_laterality`, never an assumption about window ordering."""
    import csv
    import gzip

    lat = {}
    p = find_file("orient.npz")
    if p is None:
        raise FileNotFoundError("orient.npz not attached -- needed for geometric laterality")
    z = np.load(p, allow_pickle=True)
    for u, v in zip(z["ids"].astype(str), z["laterality"]):
        lat[u] = int(v)                       # +1 left, -1 right, 0 unresolved

    row_of = {u: i for i, u in enumerate(ids)}
    t_of = {t: j for j, t in enumerate(TARGETS)}
    side = np.zeros((len(ids), len(TARGETS), WINDOWS_PER_TARGET), np.int8)
    seen = 0
    for sh in (0, 1):
        f = find_file(f"roi_provenance_shard{sh}.csv.gz")
        if f is None:
            raise FileNotFoundError(f"roi_provenance_shard{sh}.csv.gz not attached")
        with gzip.open(f, "rt", newline="") as stream:
            for r in csv.DictReader(stream):
                i = row_of.get(r["StudyInstanceUID"])
                j = t_of.get(r["target"])
                if i is None or j is None:
                    continue
                k = int(r["window_index"])
                hyp = {"L": 1, "R": -1}.get((r.get("candidate_laterality") or "").strip().upper(), 0)
                side[i, j, k] = hyp * lat.get(r["StudyInstanceUID"], 0)
                seen += 1
    known = float((side != 0).mean())
    log(f"side_match built from {seen} provenance rows | non-zero on {100 * known:.1f}% of "
        f"(study, target, window) cells | laterality resolved for "
        f"{100 * np.mean([lat.get(u, 0) != 0 for u in ids]):.1f}% of studies")
    for j, t in enumerate(TARGETS):
        nz = float((side[:, j] != 0).mean())
        log(f"    {TARGET_LAB[t]:18s} side-informative windows {100 * nz:5.1f}%"
            + ("   <- CONTROL, expected ~0%" if t not in SIDE_TARGETS else ""))
    return side


def load_index():
    ids = np.concatenate([np.load(need(f"roi_ids_shard{sh}.npy"), allow_pickle=True).astype(str)
                          for sh in (0, 1)])
    masks = np.concatenate([
        np.stack([np.load(need(f"roi_{t}_masks_shard{sh}.npy")) for t in TARGETS], axis=1)
        for sh in (0, 1)]).astype(bool)                                   # (N, T, 4)
    if len(ids) != len(set(ids)) or len(ids) != len(masks):
        raise RuntimeError("ROI ids/masks misaligned")
    if masks.shape[1:] != (len(TARGETS), WINDOWS_PER_TARGET):
        raise RuntimeError(f"unexpected ROI mask shape {masks.shape}")
    return ids, masks


def load_labels(ids):
    root = find_file("gold_test.csv").parent
    gold = pd.read_csv(root / "gold_test.csv")
    pool = pd.read_csv(root / "train_pool.csv")
    idset = set(ids)
    gold_uids = [u for u in gold.StudyInstanceUID.astype(str) if u in idset]
    goldset = set(gold.StudyInstanceUID.astype(str))
    train_uids = sorted(u for u in pool.StudyInstanceUID.astype(str) if u in idset and u not in goldset)
    labels = {}
    for df in (pool, gold):
        for u, vals in zip(df.StudyInstanceUID.astype(str), df[LAB].to_numpy(np.float32)):
            labels[u] = vals
    bad = [u for u in train_uids + gold_uids if not np.isfinite(labels[u]).all()]
    if bad:
        raise RuntimeError(f"{len(bad)} studies have NaN labels, e.g. {bad[0]}")
    if len(gold_uids) != len(gold):
        raise RuntimeError(f"only {len(gold_uids)}/{len(gold)} gold studies are in the corpus")
    return train_uids, gold_uids, labels


def build_label_mask(labels, uids):
    """True where the label is the 'report never covered this' prior and must not train."""
    Y = np.stack([labels[u] for u in uids])
    M = np.abs(Y - MASK_PRIOR) < MASK_TOL
    log(f"label mask: {int(M.sum())}/{M.size} cells ({100 * M.mean():.1f}%) are the "
        f"{MASK_PRIOR} prior and are masked out of the loss")
    for j, n in enumerate(LAB):
        keep = ~M[:, j]
        log(f"    {n:18s} masked {int(M[:, j].sum()):5d} ({100 * M[:, j].mean():4.1f}%)  "
            f"kept {int(keep.sum()):5d} studies, {int((Y[keep, j] > 0.5).sum()):5d} positive")
    dead = int(M.all(1).sum())
    if dead:
        log(f"    {dead} studies have every cell masked and are dropped from training")
    return {u: M[i] for i, u in enumerate(uids)}


def make_folds(train_uids):
    """Deterministic: sorted uids, KFold(shuffle, seed 42). Identical in every job/session."""
    fold_of = {}
    kf = KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for k, (_, te) in enumerate(kf.split(train_uids)):
        for i in te:
            fold_of[train_uids[i]] = k
    return fold_of


SLOT_COUNTS = [18, 14, 12, 8, 12]                  # cache64 SLOTS64, in order
SLOT_EDGES = np.cumsum([0] + SLOT_COUNTS)          # [0,18,32,44,52,64]


CANON_MIN_RATE = 0.02        # below this (or above 1 - this) a correction is constant


def apply_if_informative(flag, valid, label):
    """A correction applied to ~every study, or to ~none, is not a correction: it is a
    global mirror of the whole corpus, which carries no information and changes nothing
    the model can learn. The audit found exactly this on sagittal flip_h (rate 1.000, i.e.
    every sagittal series already agrees and only disagrees with the canonical target this
    code happens to name). Applying it would add a difference between the A/B arms that is
    not the hypothesis under test, so drop it and say so."""
    use = flag & valid
    nv = int(valid.sum())
    rate = (use.sum() / nv) if nv else 0.0
    if nv and CANON_MIN_RATE <= rate <= 1 - CANON_MIN_RATE:
        return use
    log(f"canon: {label} rate {rate:.3f} is constant across the corpus -- not applied")
    return np.zeros_like(use)


def build_canon(ids, masks):
    """Turn the `orient` audit's per-slot flags into per-study slice permutations and axis
    mirrors. canon["perm"][i, d] is the SOURCE slice index that destination index d reads
    from, so the destination axis is canonical: every sagittal stack runs lateral->medial.
    """
    f = find_file("orient.npz")
    if f is None:
        raise FileNotFoundError("orient.npz not attached -- canonicalisation needs the audit")
    z = np.load(f, allow_pickle=True)
    row_of = {u: i for i, u in enumerate(z["ids"].astype(str))}
    missing = [u for u in ids if u not in row_of]
    if missing:
        raise RuntimeError(f"orient.npz missing {len(missing)} studies, e.g. {missing[0]}")
    src = np.array([row_of[u] for u in ids])           # audit order -> corpus order
    rev, fh, fv = z["rev_depth"][src], z["flip_h"][src], z["flip_v"][src]
    valid, tp = z["valid"][src], z["transpose"][src]

    depth = masks.shape[1]
    if SLOT_EDGES[-1] != depth:
        raise RuntimeError(f"slot layout sums to {SLOT_EDGES[-1]}, corpus depth is {depth}")

    n = len(ids)
    perm = np.tile(np.arange(depth, dtype=np.int16), (n, 1))
    FH = np.zeros((n, depth), bool)
    FV = np.zeros((n, depth), bool)
    for j in range(len(SLOT_COUNTS)):
        a, b = int(SLOT_EDGES[j]), int(SLOT_EDGES[j + 1])
        use = valid[:, j]
        r, h, v = [apply_if_informative(x, use, f"slot{j}:{nm}")
                   for x, nm in ((rev[:, j], "rev_depth"), (fh[:, j], "flip_h"),
                                 (fv[:, j], "flip_v"))]
        perm[r, a:b] = np.arange(b - 1, a - 1, -1, dtype=np.int16)
        FH[h, a:b] = True
        FV[v, a:b] = True

    masks_canon = np.take_along_axis(masks, perm.astype(np.intp), axis=1)
    # Report what is APPLIED, not what the audit measured -- the two differ wherever a
    # correction was dropped as constant, and a log that overstates the change is how a
    # later reader concludes the wrong thing about which arm tested what.
    moved = perm != np.arange(depth, dtype=np.int16)
    touched = moved.any(1) | FH.any(1) | FV.any(1)
    log(f"canon: applied -- depth reversed on {int(moved.any(1).sum())} studies, "
        f"flip_h on {int(FH.any(1).sum())}, flip_v on {int(FV.any(1).sum())}")
    log(f"canon: audit measured -- rev_depth {int((rev & valid).sum())}, "
        f"flip_h {int((fh & valid).sum())}, flip_v {int((fv & valid).sum())}, "
        f"transpose {int((tp & valid).sum())} of {int(valid.sum())} valid slots")
    log(f"canon: {int(touched.sum())}/{n} studies ({100 * touched.mean():.1f}%) change")
    return {"perm": perm, "fh": FH, "fv": FV}, masks_canon


def valid_centers(mask):
    depth = len(mask)
    valid = np.flatnonzero(mask > 0)
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, depth - 2))]
    return [max(1, min(c, depth - 2)) for c in cs]


def eval_bags(mask, spec):
    """List of bags (each a sorted list of window centres). ("single", K) reproduces the
    trainers' linspace exactly; ("subsets", K, R) shifts that grid by r/R of one step."""
    cs = valid_centers(mask)
    n = len(cs)
    if spec[0] == "single":
        return [[cs[i] for i in np.linspace(0, n - 1, spec[1]).round().astype(int)]]
    k, r_total = spec[1], spec[2]
    step = (n - 1) / max(k - 1, 1)
    bags = []
    for r in range(r_total):
        idx = np.clip(np.round(np.linspace(0, n - 1, k) + step * r / r_total), 0, n - 1).astype(int)
        bags.append([cs[i] for i in idx])
    return bags


def rand_affine(n, gen=None):
    """n random affine matrices for grid_sample, as (n, 2, 3).

    theta maps OUTPUT normalised coords to INPUT coords, so a factor k in the linear part
    means the output samples input at k*p: k > 1 zooms OUT. To scale the anatomy UP by
    `sc` we therefore divide by sc. Translation is in normalised units where the full
    image spans [-1, 1], so a shift of `f` of the image width is 2*f here.
    """
    th = (torch.rand(n, generator=gen) * 2 - 1) * (ROT_DEG * math.pi / 180.0)
    sc = 1.0 + (torch.rand(n, generator=gen) * 2 - 1) * SCALE_JIT
    tx = (torch.rand(n, generator=gen) * 2 - 1) * SHIFT_JIT * 2.0
    ty = (torch.rand(n, generator=gen) * 2 - 1) * SHIFT_JIT * 2.0
    cos, sin = torch.cos(th) / sc, torch.sin(th) / sc
    m = torch.zeros(n, 2, 3)
    m[:, 0, 0], m[:, 0, 1], m[:, 0, 2] = cos, -sin, tx
    m[:, 1, 0], m[:, 1, 1], m[:, 1, 2] = sin, cos, ty
    return m


def augment(x):
    """One affine per WINDOW, shared across that window's 3 slices.

    Accepts any leading shape: the caller passes (R, K, 3, H, W) because the squeeze to
    (K, 3, H, W) happens AFTER this call. v1 assumed 4D and died on the first batch with
    "Expected a batch of 3D affine matrices ... for size [1, 12, 3, 336, 336]". Flatten
    everything but the trailing (3, H, W) so the contract cannot drift again.
    """
    shp = x.shape
    flat = x.reshape(-1, *shp[-3:])                 # (N, 3, H, W), N = R*K
    n = flat.shape[0]
    if ROT_DEG > 0 or SCALE_JIT > 0 or SHIFT_JIT > 0:
        grid = F.affine_grid(rand_affine(n), list(flat.shape), align_corners=False)
        # zeros padding matches the MRI background, which is ~0 after percentile windowing
        flat = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False)
    if GAMMA_JIT > 0:
        g = torch.exp((torch.rand(n, 1, 1, 1) * 2 - 1) * GAMMA_JIT)
        flat = flat.clamp(1e-6, 1.0).pow(g)
    if GAIN_AUG > 0:
        flat = (flat * (1.0 + (random.random() - 0.5) * GAIN_AUG)).clamp(0, 1)
    return flat.reshape(shp)


class ROIWindows(Dataset):
    """x (T, 4, 3, H, W), window mask m (T, 4), labels y (T,), loss weight w (T,)."""

    def __init__(self, uids, id2row, masks, labels, train, side=None):
        self.uids, self.id2row, self.masks = uids, id2row, masks
        self.labels, self.train = labels, train
        self.side = side
        self.corpus = None

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        if self.corpus is None:
            self.corpus = ROICorpus()
        uid = self.uids[i]
        row = self.id2row[uid]
        arr = self.corpus.study(row)                             # (T, 4, 3, H, W) uint8
        mrow = np.asarray(self.masks[row])                       # (T, 4) bool
        srow = (np.zeros_like(mrow, np.int8) if self.side is None
                else np.asarray(self.side[row]))                 # (T, 4) in {-1, 0, +1}
        if self.train and ROI_K_TRAIN < arr.shape[1]:
            picks = []
            for j in range(arr.shape[0]):
                valid = np.flatnonzero(mrow[j]).tolist()
                invalid = np.flatnonzero(~mrow[j]).tolist()
                if len(valid) >= ROI_K_TRAIN:
                    pick = random.sample(valid, ROI_K_TRAIN)
                else:                                            # pad with masked slots
                    pick = valid + invalid[:ROI_K_TRAIN - len(valid)]
                picks.append(sorted(pick))
            arr = np.stack([arr[j, p] for j, p in enumerate(picks)])
            mrow = np.stack([mrow[j, p] for j, p in enumerate(picks)])
            srow = np.stack([srow[j, p] for j, p in enumerate(picks)])
        x = torch.from_numpy(arr).float().div_(255.0)
        if x.shape[-1] != RES:
            T, K = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), size=(RES, RES), mode="bilinear",
                              align_corners=False).view(T, K, 3, RES, RES)
        if self.train and GAIN_AUG > 0:
            x = (x * (1.0 + (random.random() - 0.5) * GAIN_AUG)).clamp_(0, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        m = torch.from_numpy(np.ascontiguousarray(mrow))
        y = torch.from_numpy(self.labels[uid][TARGET_IDX].astype(np.float32))
        w = m.any(1).float()                    # a target with no valid window trains nothing
        sd = torch.from_numpy(np.ascontiguousarray(srow)).float()
        return x, m, y, w, sd


def collate(batch):
    return tuple(torch.stack([b[k] for b in batch]) for k in range(5))


def build_backbone(arch, pretrained):
    if arch == "tiny_test":
        net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        net.num_features = 16
        return net
    import timm
    return timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=3,
                             global_pool="avg")


class CropNet(nn.Module):
    """Shared backbone; one masked attention pool + one read-out PER TARGET, so each target
    pools only its own four crops."""

    def __init__(self, backbone, n_targets=len(TARGETS), drop=0.2):
        super().__init__()
        f = backbone.num_features
        self.backbone = backbone
        self.norm = nn.LayerNorm(f)
        self.att = nn.ModuleList([nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(drop),
                                                nn.Linear(256, 1)) for _ in range(n_targets)])
        self.cls = nn.ModuleList([nn.Linear(f, 1) for _ in range(n_targets)])
        # Zero-init: at step 0 the side term contributes nothing and arm22 == arm21.
        self.side_gate = nn.Parameter(torch.zeros(n_targets))

    def forward(self, x, m, side=None):
        B, T, K = x.shape[:3]
        flat = x.flatten(0, 2)
        if self.training or flat.shape[0] <= CNN_CHUNK:
            feats = self.backbone(flat)
        else:                                   # bounded eval memory; identical in eval mode
            feats = torch.cat([self.backbone(flat[i:i + CNN_CHUNK])
                               for i in range(0, flat.shape[0], CNN_CHUNK)])
        h = self.norm(feats.view(B, T, K, -1))
        outs = []
        for j in range(T):
            hj = h[:, j]
            a = self.att[j](hj).squeeze(-1).float()
            if side is not None:
                a = a + self.side_gate[j].float() * side[:, j].float()
            a = a.masked_fill(~m[:, j], -1e4)
            a = torch.softmax(a, dim=1).to(hj.dtype)
            outs.append(self.cls[j](torch.einsum("bk,bkf->bf", a, hj)).squeeze(-1))
        return torch.stack(outs, 1)             # (B, T)


def make_model(dev):
    bb = build_backbone(ARCH, PRETRAINED)
    if ARCH != "tiny_test" and GRAD_CKPT:
        try:
            bb.set_grad_checkpointing(True)
        except Exception as e:
            log(f"grad checkpointing unavailable for {ARCH}: {e}")
    return CropNet(bb).to(dev)


def make_opt(model):
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return torch.optim.AdamW([{"params": bb, "lr": BB_LR},
                              {"params": head, "lr": HEAD_LR}], weight_decay=WD)


def macro_auc(Y, P, M=None):
    """M marks cells the report never covered. Scoring against those would PENALISE a
    model that correctly fires on a finding the radiologist did not discuss, so both A/B
    arms are scored with the same cells excluded."""
    aucs = {}
    for j, name in enumerate(LAB):
        keep = np.ones(len(Y), bool) if M is None else ~M[:, j]
        yj = (Y[keep, j] > 0.5).astype(int)
        if len(set(yj)) > 1:
            aucs[name] = float(roc_auc_score(yj, P[keep, j]))
    return float(np.mean(list(aucs.values()))), aucs


@torch.no_grad()
def dist_predict(model, ctx, uids, dev, rank, world):
    """Predict uids split across ranks; every rank returns the full {uid: (T,) probs}."""
    model.eval()
    mine = uids[rank::world]
    ds = ROIWindows(mine, ctx["id2row"], ctx["masks"], ctx["labels"], train=False,
                    side=ctx.get("side"))
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=N_WORKERS, collate_fn=collate)
    out, j = [], 0
    for x, m, _, _, sd in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            logit = model(x.to(dev, non_blocking=True), m.to(dev), sd.to(dev))
        p = torch.sigmoid(logit.float()).cpu().numpy()
        for b in range(len(p)):
            out.append((mine[j], p[b]))
            j += 1
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, out)
        out = [kv for part in gathered for kv in part]
    return dict(out)


def score(pred, uids, ctx):
    """Per-target AUC over studies with >=1 valid window for that target."""
    P = np.stack([pred[u] for u in uids])
    Y = np.stack([ctx["labels"][u][TARGET_IDX] for u in uids])
    V = np.stack([ctx["masks"][ctx["id2row"][u]].any(1) for u in uids])
    aucs = {}
    for j, t in enumerate(TARGETS):
        k = V[:, j]
        yj = (Y[k, j] > 0.5).astype(int)
        if len(set(yj)) > 1:
            aucs[TARGET_LAB[t]] = float(roc_auc_score(yj, P[k, j]))
    return float(np.mean(list(aucs.values()))), aucs, P, Y, V


def _train_worker(rank, world, ctx, job):
    arm_child()
    ddp = world > 1
    if ddp:
        dist.init_process_group(os.environ.get("ARM10_BACKEND", "nccl"),
                                init_method=f"tcp://127.0.0.1:{ctx['port'] + job['idx']}",
                                rank=rank, world_size=world, timeout=datetime.timedelta(hours=2))
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        dev = torch.device(f"cuda:{rank}")
    else:
        dev = torch.device("cpu")
    random.seed(SEED + 1000 * job["fold"] + rank)
    np.random.seed(SEED + 1000 * job["fold"] + rank)
    torch.manual_seed(SEED)                          # identical init on every rank

    name, fold = job["name"], job["fold"]
    fold_of = ctx["fold_of"]
    tr_uids = [u for u in ctx["train_uids"] if fold_of[u] != fold]
    oof_uids = [u for u in ctx["train_uids"] if fold_of[u] == fold]
    # Prevalence from the (study, target) cells that actually train.
    Ytr = np.stack([ctx["labels"][u][TARGET_IDX] for u in tr_uids])
    Wtr = np.stack([ctx["masks"][ctx["id2row"][u]].any(1) for u in tr_uids]).astype(np.float32)
    prev = np.clip((Ytr * Wtr).sum(0) / np.maximum(Wtr.sum(0), 1.0), 0.03, 0.7)
    pw = torch.from_numpy(np.clip((1 - prev) / prev, 1, 10).astype(np.float32)).to(dev)

    if ddp and rank != 0:
        dist.barrier()
    model = make_model(dev)
    if ddp and rank == 0:
        dist.barrier()
    net = DDP(model, device_ids=[rank] if dev.type == "cuda" else None,
              static_graph=True) if ddp else model

    tds = ROIWindows(tr_uids, ctx["id2row"], ctx["masks"], ctx["labels"], train=True,
                     side=ctx.get("side"))
    sampler = DistributedSampler(tds, world, rank, shuffle=True, seed=SEED + fold,
                                 drop_last=True) if ddp else None
    tl = DataLoader(tds, batch_size=BS_PER_GPU, sampler=sampler, shuffle=sampler is None,
                    num_workers=N_WORKERS, drop_last=True, collate_fn=collate,
                    pin_memory=dev.type == "cuda", persistent_workers=N_WORKERS > 0)
    opt = make_opt(model)
    steps_per_epoch = len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[BB_LR, HEAD_LR],
                                                total_steps=EPOCHS * steps_per_epoch,
                                                pct_start=PCT_START)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    log(f"[{name}] fold {fold} | train {len(tr_uids)} | oof {len(oof_uids)} | "
        f"gold {len(ctx['gold_uids'])} | {steps_per_epoch} steps/epoch x {EPOCHS} | "
        f"{len(TARGETS)} targets x {WINDOWS_PER_TARGET} windows | world {world}", rank)

    hist = []
    for ep in range(EPOCHS):
        if sampler is not None:
            sampler.set_epoch(ep)
        net.train()
        t_ep = last = time.time()
        tot, nb, skipped = 0.0, 0, 0
        for x, m, y, w, sd in tl:
            x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
            y, w = y.to(dev, non_blocking=True), w.to(dev, non_blocking=True)
            sd = sd.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = net(x, m, sd)
            per = F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pw,
                                                     reduction="none")
            loss = (per * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if torch.isfinite(loss):
                tot += float(loss)
            else:
                skipped += 1
            nb += 1
            if rank == 0 and time.time() - last > PROGRESS_S:
                el = time.time() - t_ep
                log(f"  [{name}] ep{ep} {nb}/{steps_per_epoch} loss {tot / max(nb - skipped, 1):.4f} "
                    f"lr {opt.param_groups[1]['lr']:.2e} | {el / 60:.1f} min, eta "
                    f"{(steps_per_epoch - nb) * el / nb / 60:.1f} min")
                last = time.time()
        minutes = (time.time() - t_ep) / 60
        if ddp:
            dist.barrier()
        gp = dist_predict(model, ctx, ctx["gold_uids"], dev, rank, world)
        if rank == 0:
            au = score(gp, ctx["gold_uids"], ctx)[0]
            gates = [float(v) for v in model.side_gate.detach().cpu()]
            hist.append({"ep": ep, "loss": tot / max(nb - skipped, 1), "gold_auc": au,
                         "minutes": minutes, "side_gate": gates})
            if ep >= EPOCHS - SWA_LAST:
                torch.save(model.state_dict(), WORK / f"{TAG}_{name}_ep{ep}.pt")
            log(f"[{name}] ep{ep} loss {hist[-1]['loss']:.4f} | gold(5) {au:.4f} (log only) | "
                f"gate " + " ".join(f"{g:+.2f}" for g in gates) + " | "
                f"{minutes:.1f} min" + (f" | non-finite {skipped}" if skipped else ""))
        if ddp:
            dist.barrier()

    finalize_job(model, ctx, job, dev, rank, world, oof_uids, hist)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def train_worker(rank, world, ctx, job):                        # noqa: F811
    """Wrap the real worker so a failure prints its type, message and peak GPU memory.
    Kaggle's log cut arm17's and arm21 v1's tracebacks before the exception line, which
    left the cause to inference both times."""
    try:
        _train_worker(rank, world, ctx, job)
    except BaseException as e:
        peak = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
        print(f"[rank {rank}] CHILD FAILED: {type(e).__name__}: {str(e)[:600]} | "
              f"peak GPU memory {peak:.2f} GiB", flush=True)
        raise


def finalize_job(model, ctx, job, dev, rank, world, oof_uids, hist):
    """Mean of the last SWA_LAST epochs' weights; score gold and this fold's OOF studies."""
    name = job["name"]
    files = [WORK / f"{TAG}_{name}_ep{e}.pt" for e in range(EPOCHS - SWA_LAST, EPOCHS)]
    sds = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    avg = {k: (sum(sd[k].double() for sd in sds) / len(sds)).to(sds[0][k].dtype)
           if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
    model.load_state_dict(avg)
    gp = dist_predict(model, ctx, ctx["gold_uids"], dev, rank, world)
    op = dist_predict(model, ctx, oof_uids, dev, rank, world)
    if rank != 0:
        return
    g_au, g_aucs, g_P, g_Y, g_V = score(gp, ctx["gold_uids"], ctx)
    o_au, o_aucs, o_P, o_Y, o_V = score(op, oof_uids, ctx)
    res = {"job": name, "fold": job["fold"], "swa_epochs": list(range(EPOCHS - SWA_LAST, EPOCHS)),
           "gold_auc": g_au, "gold_aucs": g_aucs, "oof_auc": o_au, "oof_aucs": o_aucs,
           "n_oof": len(oof_uids), "hist": hist}
    torch.save({"model": avg, "arch": ARCH, "res": RES, "targets": TARGETS,
                "target_labels": [TARGET_LAB[t] for t in TARGETS],
                "windows_per_target": WINDOWS_PER_TARGET, "recipe": "roi-cache-ffcffe7-kaggle",
                "fold": job["fold"], "gold_auc": g_au, "oof_auc": o_au,
                "epoch": res["swa_epochs"]}, WORK / f"{TAG}_{name}_swa.pt")
    np.savez(WORK / f"{TAG}_{name}_preds.npz", gold_ids=np.array(ctx["gold_uids"]), gold_pred=g_P,
             gold_truth=g_Y, gold_valid=g_V, oof_ids=np.array(oof_uids), oof_pred=o_P,
             oof_truth=o_Y, oof_valid=o_V, targets=np.array(TARGETS))
    for f in files:
        f.unlink(missing_ok=True)
    json.dump(res, open(WORK / f"{TAG}_{name}_result.json", "w"), indent=1)
    log(f"=== [{name}] DONE: standalone OOF mean AUC over 5 targets {o_au:.4f} | "
        f"gold(5) {g_au:.4f} (not used) ===")


# -------------------------------------------------------------------- main --
def result_path(name):
    return WORK / f"{TAG}_{name}_result.json"


def carry_forward(name):
    """If a previous session already finished this job, copy its artefacts and skip it."""
    if result_path(name).exists():
        return True
    prev = find_file(f"{TAG}_{name}_result.json")
    if prev is None:
        return False
    for suffix in ("_result.json", "_swa.pt", "_preds.npz"):
        src = prev.parent / f"{TAG}_{name}{suffix}"
        if src.exists():
            shutil.copy(src, WORK / src.name)
    log(f"[{name}] finished in a previous session -- carried forward, skipped")
    return True


def run_job(job, world, ctx):
    if carry_forward(job["name"]):
        return json.load(open(result_path(job["name"])))
    log(f">>> job {job['idx'] + 1}: {job['name']} (fold {job['fold']}, head {job['head']})")
    if world > 1:
        ok = False
        for attempt in (1, 2):                      # one automatic retry: v1 lost a job to a rare crash
            job = dict(job, idx=job["idx"] + 10 * (attempt - 1))   # fresh rendezvous port on retry
            ok = run_children(train_worker, (world, ctx, job), world, JOB_DEADLINE_S, job["name"])
            if ok:
                break
            log(f"[{job['name']}] attempt {attempt} failed" + (" -- retrying once" if attempt == 1 else ""))
        if not ok:
            raise SystemExit(f"job {job['name']} failed twice; finished jobs are saved -- rerun to resume")
    else:
        train_worker(0, 1, ctx, job)
    return json.load(open(result_path(job["name"])))


def pct_rank(v):
    return pd.Series(v).rank(pct=True).to_numpy()          # NaN stays NaN


def blend_test(crop_npz):
    """Does the crop model add a new perspective to a global model? Pre-registered rule:
    equal-weight rank blend with arm20 f0_noaug, mean AUC over the 5 targets, >= +0.002."""
    g = np.load(need(f"{GLOBAL_REF}_preds.npz"), allow_pickle=True)
    c = np.load(crop_npz, allow_pickle=True)
    g_ids, c_ids = g["oof_ids"].astype(str), c["oof_ids"].astype(str)
    if set(g_ids) != set(c_ids):
        raise RuntimeError(f"fold-0 sets differ: {len(set(g_ids) ^ set(c_ids))} studies")
    order = {u: i for i, u in enumerate(c_ids)}
    sel = np.array([order[u] for u in g_ids])
    c_pred, c_valid = c["oof_pred"][sel], c["oof_valid"][sel]
    g_pred, Y = g["oof_pred"], (g["oof_truth"] > 0.5)

    rows, deltas = [], []
    for j, t in enumerate(TARGETS):
        col = LAB.index(TARGET_LAB[t])
        y = Y[:, col].astype(int)
        rg = pct_rank(g_pred[:, col])
        rc = pct_rank(np.where(c_valid[:, j], c_pred[:, j], np.nan))
        auc_g = roc_auc_score(y, g_pred[:, col])
        k = c_valid[:, j]
        auc_c = roc_auc_score(y[k], c_pred[k, j]) if len(set(y[k])) > 1 else float("nan")
        # Where the crop model has no valid window, the global model alone.
        b50 = roc_auc_score(y, np.where(np.isnan(rc), rg, 0.5 * rg + 0.5 * rc))
        b20 = roc_auc_score(y, np.where(np.isnan(rc), rg, 0.8 * rg + 0.2 * rc))
        deltas.append(b50 - auc_g)
        rows.append((TARGET_LAB[t], auc_g, auc_c, b50, b20, int(k.sum()), int(y.sum())))
    return rows, float(np.mean(deltas))


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    ids, masks = load_index()
    train_uids, gold_uids, labels = load_labels(ids)
    fold_of = make_folds(train_uids)
    json.dump(fold_of, open(WORK / f"{TAG}_folds.json", "w"))

    # The blend test is only meaningful on the SAME 870 held-out studies as the comparator.
    ref = find_file(f"{GLOBAL_REF.split('_')[0]}_folds.json")
    if ref is not None:
        ref_fold = json.load(open(ref))
        diff = sum(1 for u, k in fold_of.items() if ref_fold.get(u) != k)
        if diff:
            raise RuntimeError(f"fold split differs from {ref.name} on {diff} studies")
        log(f"fold split identical to {ref.name} ({len(fold_of)} studies)")
    else:
        log(f"WARNING: {GLOBAL_REF.split('_')[0]}_folds.json not attached -- split not cross-checked")

    valid_rate = {TARGET_LAB[t]: float(masks[:, j].any(1).mean()) for j, t in enumerate(TARGETS)}
    log("studies with >=1 valid window per target: " +
        ", ".join(f"{k} {100 * v:.1f}%" for k, v in valid_rate.items()))

    side = load_side_match(ids)
    ctx = {"id2row": {u: i for i, u in enumerate(ids)}, "masks": masks, "train_uids": train_uids,
           "gold_uids": gold_uids, "labels": labels, "fold_of": fold_of, "side": side,
           "port": 29500 + random.randint(0, 900)}
    n_gpu = torch.cuda.device_count()
    world = int(os.environ.get("ARM10_WORLD", max(n_gpu, 1)))
    log(f"{TAG} | arch {ARCH} @{RES} | gpus {n_gpu} | train {len(train_uids)} in {N_FOLDS} folds "
        f"{np.bincount(list(fold_of.values())).tolist()} | gold {len(gold_uids)}")

    if world > 1 and not run_children(spawn_canary, (), 1, 180, "spawn canary"):
        raise SystemExit("child processes cannot start under " + START_METHOD)

    r = run_job({"idx": 0, "name": "f0_crop", "fold": 0, "head": "crop"}, world, ctx)

    # PRIMARY: the side mechanism can only help the three laterality-dependent targets.
    lat_names = [TARGET_LAB[t] for t in SIDE_TARGETS]
    ctrl_names = [TARGET_LAB[t] for t in TARGETS if t not in SIDE_TARGETS]
    now, ref = r["oof_aucs"], ARM21_OOF
    d_lat = float(np.mean([now[n] - ref[n] for n in lat_names if n in now]))
    d_ctrl = float(np.mean([now[n] - ref[n] for n in ctrl_names if n in now]))
    adopt_side = d_lat >= ADOPT_MARGIN
    gates = (r.get("hist") or [{}])[-1].get("side_gate", [])
    log("")
    log("=== SIDE-AWARENESS DECISION (pre-registered before the run) ===")
    log(f"    {'finding':18s} {'arm21':>8} {'arm22':>8} {'delta':>8}")
    for n in lat_names + ctrl_names:
        if n in now:
            tag = "" if n in lat_names else "   <- CONTROL, expected flat"
            log(f"    {n:18s} {ref[n]:8.4f} {now[n]:8.4f} {now[n] - ref[n]:+8.4f}{tag}")
    log(f"  mean over the 3 laterality-dependent targets: {d_lat:+.4f} | margin {ADOPT_MARGIN}")
    log(f"  mean over the 2 side-independent controls:    {d_ctrl:+.4f} (should be ~0)")
    if gates:
        log("  learned |side_gate| per target: " +
            " ".join(f"{TARGET_LAB[t]} {abs(g):.3f}" for t, g in zip(TARGETS, gates)))
        log("  (a gate that stayed near zero means the model did not use the side signal)")
    log("  -> " + ("ADOPT side-awareness" if adopt_side else "REJECT: side signal did not help"))

    rows, mean_delta = blend_test(WORK / f"{TAG}_f0_crop_preds.npz")
    adopt = mean_delta >= ADOPT_MARGIN
    log("")
    log("=== BLEND TEST (pre-registered): does the crop model add a new perspective? ===")
    log(f"  fold-0 OOF | global = {GLOBAL_REF} | equal-weight rank blend decides")
    log(f"    {'finding':18s} {'global':>7} {'crop':>7} {'blend.5':>8} {'d.5':>7} "
        f"{'blend.2':>8} {'valid':>6} {'n+':>5}")
    for nme, ag, ac, b50, b20, nv, npos in rows:
        tag = "   <- predicted NOT to gain (label problem)" if nme == "Synovitis" else ""
        log(f"    {nme:18s} {ag:7.4f} {ac:7.4f} {b50:8.4f} {b50 - ag:+7.4f} {b20:8.4f} "
            f"{nv:6d} {npos:5d}{tag}")
    log(f"  mean d(blend.5 - global) over 5 targets: {mean_delta:+.4f} | margin {ADOPT_MARGIN}")
    log("  -> " + ("ADOPT: the crop model adds a new perspective -- a blend member"
                   if adopt else "REJECT: no complementary signal worth a member"))
    log(f"  standalone crop OOF {r['oof_auc']:.4f} | gold(5) {r['gold_auc']:.4f} (reported, not used)")
    json.dump({"side_rule": f"PRIMARY: adopt iff mean OOF over {lat_names} improves >= "
                            f"{ADOPT_MARGIN} vs arm21",
               "side_delta_lateral": d_lat, "side_delta_control": d_ctrl,
               "side_adopt": bool(adopt_side), "side_gate": gates, "arm21_reference": ARM21_OOF,
               "rule": f"adopt iff mean over 5 targets of AUC(equal rank blend) - AUC(global) "
                       f">= {ADOPT_MARGIN} on fold-0 OOF",
               "global_ref": GLOBAL_REF, "mean_delta": mean_delta, "adopt": bool(adopt),
               "per_target": [dict(zip(("finding", "global", "crop", "blend_05", "blend_02",
                                        "n_valid", "n_pos"), row)) for row in rows],
               "crop_standalone_oof": r["oof_auc"], "crop_gold": r["gold_auc"]},
              open(WORK / f"{TAG}_decision.json", "w"), indent=1)


def spawn_canary(rank):
    print(f"[canary] child {rank} alive, pid {os.getpid()}, torch {torch.__version__}, "
          f"cuda visible {torch.cuda.device_count()}", flush=True)


def run_children(fn, args, nprocs, deadline_s, label):
    """start_processes with a hard deadline. Returns True only on clean completion."""
    pctx = mp.start_processes(fn, args=args, nprocs=nprocs, join=False, start_method=START_METHOD)
    t0 = time.time()
    try:
        while not pctx.join(timeout=15):
            if time.time() - t0 > deadline_s:
                log(f"{label}: exceeded {deadline_s / 60:.0f} min -- terminating children")
                for proc in pctx.processes:
                    if proc.is_alive():
                        proc.terminate()
                return False
    except Exception as e:
        log(f"{label}: child failed -- {type(e).__name__}: {str(e)[:2000]}")
        for proc in pctx.processes:
            if proc.is_alive():
                proc.terminate()
        return False
    return True


if __name__ == "__main__":
    main()
