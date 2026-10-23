"""arm20: the 20-epoch rerun of the augmentation A/B. Tests underfitting, not a new idea.

WHY. arm19 rejected augmentation at -0.0003 OOF, but its own training loss says the
augmented arm never converged:

    f0_noaug  loss ep0 0.9235 -> ep9 0.6727
    f0_aug    loss ep0 0.9219 -> ep9 0.6999

The augmented model finishes HARDER to fit, which is what augmentation is supposed to do,
and at 10 epochs it is being judged mid-convergence. arm19's docstring pre-registered this
exact follow-up before the run: "if this arm shows a positive trend that misses the margin,
the correct follow-up is a longer run, NOT abandoning augmentation."

HONEST CAVEAT. I wrote "positive trend" and arm19 came in flat (-0.0003), not positive. So
this rerun is NOT automatically licensed by that clause and should be read as a second
test of the same hypothesis, with the prior lowered accordingly. What keeps it worth 3.7
GPU-h is the per-finding split, which is not noise-shaped: augmentation helped the diffuse,
position-variable findings -- Lateral Meniscus +.0089, Baker's +.0083, MCL +.0081, Lateral
OA +.0040, ACL +.0039, three of them among our weakest -- and hurt the small focal ones --
Contusion -.0162, Fracture -.0099, Synovitis -.0078. Rotating and rescaling at 336 px
plausibly interpolates small focal lesions away while genuinely regularising larger
structures.

WHY NOT EARLY STOPPING (the obvious alternative, and it is wrong here):
  1. It breaks OneCycle. The LR anneals to ~0 at the end of the schedule; stopping at
     epoch 12 of 20 leaves the model at a high LR, strictly worse than one that finished.
     And our reported model is the top-SWA_LAST epoch weight average, which only means
     anything over the annealed tail -- early stopping would break that too.
  2. There is no clean signal to stop on. Gold-58 is retired and stopping on it leaks gold
     into model selection (the exact criticism we made of the public notebook). Fold-0 OOF
     is the DECISION metric, so stopping on it inflates the number we then report. Doing it
     properly needs a fresh inner validation split carved from the 3,479 training studies,
     which costs data and adds a confound to the A/B.
A fixed longer schedule gets the same extra training with the anneal and SWA intact.

BOTH ARMS RETRAIN. f0_noaug at 20 epochs is a different model from f0_noaug at 10, so the
carried-forward arm19 baseline cannot be reused -- the comparison has to be at matched
epochs or it measures schedule length rather than augmentation.

PRE-REGISTERED, unchanged from arm19 and fixed before this run: adopt iff fold-0 OOF over
all cells improves by >= ADOPT_MARGIN = 0.002. Gold-58 reported, does not decide.
PRE-REGISTERED STOPPING RULE FOR THE IDEA ITSELF: if augmentation is flat or negative at
20 epochs too, augmentation is done -- no third run at 30 epochs, no weaker-settings
variant fished for afterwards. That is written here so it cannot be relitigated later.
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

TAG = "arm20"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]

ARCH = os.environ.get("ARM10_ARCH", "convnext_small.in12k_ft_in1k_384")
PRETRAINED = os.environ.get("ARM10_PRETRAINED", "1") == "1"
RES = int(os.environ.get("ARM10_RES", "336"))
K_TRAIN = int(os.environ.get("ARM10_K", "12"))
BS_PER_GPU = int(os.environ.get("ARM10_BS", "4"))
EPOCHS = int(os.environ.get("ARM10_EPOCHS", "20"))
RANK_LAMBDA = float(os.environ.get("ARM20_RANK", "1.0"))    # weight of the pairwise AUC-ranking term
RANK_POS, RANK_NEG = 0.6, 0.4                               # reliable-pos / reliable-neg thresholds (soft labels)
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
class Corpus:
    """Both cache64 shards as one (4407, 64, 336, 336) view; shard 0 rows then shard 1."""

    def __init__(self):
        a = np.load(find_file("vols_shard0.npy"), mmap_mode="r")
        b = np.load(find_file("vols_shard1.npy"), mmap_mode="r")
        self.a, self.b, self.n_a = a, b, a.shape[0]
        self.depth = a.shape[1]

    def study(self, row, slices):
        src = self.a[row] if row < self.n_a else self.b[row - self.n_a]
        return np.asarray(src[slices])


def load_index():
    ids = np.concatenate([np.load(find_file("ids_shard0.npy"), allow_pickle=True).astype(str),
                          np.load(find_file("ids_shard1.npy"), allow_pickle=True).astype(str)])
    masks = np.concatenate([np.load(find_file("masks_shard0.npy")),
                            np.load(find_file("masks_shard1.npy"))])
    if len(ids) != len(set(ids)) or len(ids) != len(masks):
        raise RuntimeError("corpus ids/masks misaligned")
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


class StudyWindows(Dataset):
    """train: one bag of K_TRAIN random centres, SORTED by depth (order matters to the GRU;
    the attention head is permutation-invariant, so sorting changes nothing for it).
    eval:  R bags per study per `spec`. Returns x (R,K,3,H,W), pos (R,K), y."""

    def __init__(self, uids, id2row, masks, labels, train, spec=None, canon=None,
                 lab_mask=None, aug=False):
        self.uids, self.id2row, self.masks = uids, id2row, masks
        self.labels, self.train, self.spec = labels, train, spec
        self.canon = canon
        self.lab_mask = lab_mask
        self.aug = aug
        self.corpus = None

    def __len__(self):
        return len(self.uids)

    def bags(self, row):
        if self.train:
            cs = valid_centers(self.masks[row])
            pool = cs * (K_TRAIN // len(cs) + 1)
            random.shuffle(pool)
            return [sorted(pool[:K_TRAIN])]
        return eval_bags(self.masks[row], self.spec)

    def __getitem__(self, i):
        if self.corpus is None:
            self.corpus = Corpus()
        uid = self.uids[i]
        row = self.id2row[uid]
        bags = self.bags(row)
        need = sorted({s for b in bags for c in b for s in (c - 1, c, c + 1)})
        at = {s: j for j, s in enumerate(need)}
        if self.canon is None:
            stack = self.corpus.study(row, need)
        else:
            # `need` indexes the CANONICAL axis; read the source slices it maps
            # to, then mirror each as its slot requires. masks were permuted the
            # same way, so centres were already chosen in canonical space.
            stack = self.corpus.study(row, self.canon["perm"][row][need].tolist())
            fh = self.canon["fh"][row][need]
            fv = self.canon["fv"][row][need]
            if fh.any():
                stack[fh] = stack[fh][:, :, ::-1]
            if fv.any():
                stack[fv] = stack[fv][:, ::-1, :]
        tri = np.stack([np.stack([stack[[at[c - 1], at[c], at[c + 1]]] for c in b]) for b in bags])
        x = torch.from_numpy(tri).float().div_(255.0)                  # (R, K, 3, H, W)
        if x.shape[-1] != RES:
            R, K = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), size=(RES, RES), mode="bilinear",
                              align_corners=False).view(R, K, 3, RES, RES)
        if self.train:
            x = augment(x) if self.aug else (
                (x * (1.0 + (random.random() - 0.5) * GAIN_AUG)).clamp_(0, 1)
                if GAIN_AUG > 0 else x)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        pos = torch.tensor(bags, dtype=torch.long)                      # (R, K)
        if self.train:
            x, pos = x[0], pos[0]
        w = (np.ones(len(LAB), np.float32) if self.lab_mask is None
             else (~self.lab_mask[uid]).astype(np.float32))
        return x, pos, torch.from_numpy(self.labels[uid]), torch.from_numpy(w)


def collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), torch.stack([b[3] for b in batch]))


# ------------------------------------------------------------------- model --
def build_backbone(arch, pretrained):
    if arch == "tiny_test":
        net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        net.num_features = 16
        return net
    import timm
    return timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=3,
                             global_pool="avg")


class KneeNet(nn.Module):
    """Per-finding attention MIL (module names as arm10-13: norm / att / clsW / clsb),
    optionally preceded by a depth-aware residual BiGRU (head="seq")."""

    def __init__(self, backbone, head, depth, n=len(LAB), drop=0.2):
        super().__init__()
        f = backbone.num_features
        self.backbone, self.head = backbone, head
        self.norm = nn.LayerNorm(f)
        if head == "seq":
            self.pos_emb = nn.Embedding(depth, f)
            self.gru = nn.GRU(f, GRU_HIDDEN, batch_first=True, bidirectional=True)
            self.gru_proj = nn.Linear(2 * GRU_HIDDEN, f)
            nn.init.zeros_(self.pos_emb.weight)          # zero-init: step 0 == baseline head
            nn.init.zeros_(self.gru_proj.weight)
            nn.init.zeros_(self.gru_proj.bias)
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, f))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(self, x, pos):
        B, K = x.shape[:2]
        h = self.norm(self.backbone(x.flatten(0, 1)).view(B, K, -1))
        if self.head == "seq":
            g, _ = self.gru(h + self.pos_emb(pos).to(h.dtype))
            h = h + self.gru_proj(g).to(h.dtype)
        a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def make_model(head, dev, depth):
    return KneeNet(build_backbone(ARCH, PRETRAINED), head, depth).to(dev)


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
def dist_predict(model, ctx, uids, spec, dev, rank, world):
    """Predict uids split across ranks; every rank returns the full {uid: probs} dict."""
    model.eval()
    mine = uids[rank::world]
    ds = StudyWindows(mine, ctx["id2row"], ctx["masks"], ctx["labels"], train=False,
                      spec=spec, canon=ctx.get("canon"))
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=N_WORKERS, collate_fn=collate)
    out, j = [], 0
    for x, pos, _, _ in dl:
        B, R, K = x.shape[:3]
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            logit = model(x.flatten(0, 1).to(dev, non_blocking=True), pos.flatten(0, 1).to(dev))
        p = torch.sigmoid(logit.float()).view(B, R, -1).mean(1).cpu().numpy()
        for b in range(B):
            out.append((mine[j], p[b]))
            j += 1
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, out)
        out = [kv for part in gathered for kv in part]
    return dict(out)


def score(pred, uids, labels, lab_mask=None):
    P = np.stack([pred[u] for u in uids])
    Y = np.stack([labels[u] for u in uids])
    M = None if lab_mask is None else np.stack([lab_mask[u] for u in uids])
    au, aucs = macro_auc(Y, P, M)
    return au, aucs, P, Y


# ------------------------------------------------------------------- train --
def train_worker(rank, world, ctx, job):
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

    name, head, fold = job["name"], job["head"], job["fold"]
    # Rebind ctx for this arm BEFORE anything reads masks: the canonical arm sees the
    # permuted masks, so window centres are chosen in canonical space.
    if job.get("canon"):
        ctx = dict(ctx, masks=ctx["masks_canon"], canon=ctx["canon_arrays"])
    else:
        ctx = dict(ctx, canon=None)
    # `lab_mask` gates the LOSS and so differs per arm. `lab_mask_all` gates the
    # EVALUATION and is always present, so both arms are scored the same way.
    ctx = dict(ctx, lab_mask=ctx["lab_mask_all"] if job.get("mask") else None)
    fold_of = ctx["fold_of"]
    tr_uids = [u for u in ctx["train_uids"] if fold_of[u] != fold]
    oof_uids = [u for u in ctx["train_uids"] if fold_of[u] == fold]
    Ytr = np.stack([ctx["labels"][u] for u in tr_uids])
    if ctx.get("lab_mask") is not None:
        # Prevalence must come from the cells that still train, or pos_weight is computed
        # against a distribution the loss no longer sees: the masked cells are all
        # negatives, so leaving them in would understate prevalence and over-boost
        # positives for exactly the findings that were masked hardest.
        Wtr = np.stack([~ctx["lab_mask"][u] for u in tr_uids]).astype(np.float32)
        prev = (Ytr * Wtr).sum(0) / np.maximum(Wtr.sum(0), 1.0)
        dead = [u for u in tr_uids if not (~ctx["lab_mask"][u]).any()]
        if dead:
            tr_uids = [u for u in tr_uids if (~ctx["lab_mask"][u]).any()]
            log(f"[{job['name']}] dropped {len(dead)} studies with every cell masked", rank)
    else:
        prev = Ytr.mean(0)
    prev = np.clip(prev, 0.03, 0.7)
    pw = torch.from_numpy(np.clip((1 - prev) / prev, 1, 10).astype(np.float32)).to(dev)
    depth = ctx["masks"].shape[1]

    if ddp and rank != 0:
        dist.barrier()
    model = make_model(head, dev, depth)
    if ddp and rank == 0:
        dist.barrier()
    net = DDP(model, device_ids=[rank] if dev.type == "cuda" else None,
              static_graph=True) if ddp else model

    tds = StudyWindows(tr_uids, ctx["id2row"], ctx["masks"], ctx["labels"], train=True,
                       canon=ctx.get("canon"), lab_mask=ctx.get("lab_mask"),
                       aug=bool(job.get("aug")))
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
    n_head = sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith("backbone."))
    log(f"[{name}] fold {fold} head {head} | train {len(tr_uids)} | oof {len(oof_uids)} | "
        f"gold {len(ctx['gold_uids'])} | {steps_per_epoch} steps/epoch x {EPOCHS} | "
        f"head params {n_head / 1e6:.2f}M | world {world}", rank)

    hist = []
    for ep in range(EPOCHS):
        if sampler is not None:
            sampler.set_epoch(ep)
        net.train()
        t_ep = last = time.time()
        tot, nb, skipped = 0.0, 0, 0
        for x, pos, y, w in tl:
            x, pos, y = x.to(dev, non_blocking=True), pos.to(dev), y.to(dev, non_blocking=True)
            w = w.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = net(x, pos)
            per = F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pw,
                                                     reduction="none")
            # Mean over the cells that carry evidence. Denominator is the live cell count,
            # not B*12, so a batch of mostly-masked studies is not silently down-weighted.
            loss = (per * w).sum() / w.sum().clamp(min=1.0)
            # Pairwise ranking term (RankNet surrogate for AUC, the actual metric). Per finding,
            # over SUPERVISED cells only (w>0), reliable positives vs reliable negatives; uncertain
            # pairs ignored. Directly optimizes which study ranks above which -- what AUC measures.
            if RANK_LAMBDA > 0:
                z = logits.float(); terms = []
                for j in range(y.shape[1]):
                    m = w[:, j] > 0
                    if int(m.sum()) < 2:
                        continue
                    yj, zj = y[m, j], z[m, j]
                    pos, neg = zj[yj >= RANK_POS], zj[yj <= RANK_NEG]
                    if pos.numel() == 0 or neg.numel() == 0:
                        continue
                    terms.append(F.softplus(-(pos.unsqueeze(1) - neg.unsqueeze(0))).mean())
                if terms:
                    loss = loss + RANK_LAMBDA * torch.stack(terms).mean()
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
                    f"{(steps_per_epoch - nb) * el / nb / 60:.1f} min"
                    + (f" | non-finite {skipped}" if skipped else ""))
                last = time.time()
        minutes = (time.time() - t_ep) / 60
        if ddp:
            dist.barrier()
        gp = dist_predict(model, ctx, ctx["gold_uids"], EVAL_SPEC[head]["epoch"], dev, rank, world)
        if rank == 0:
            au = score(gp, ctx["gold_uids"], ctx["labels"])[0]
            gate = float(model.gru_proj.weight.abs().mean()) if head == "seq" else float("nan")
            hist.append({"ep": ep, "loss": tot / max(nb - skipped, 1), "gold_auc": au,
                         "minutes": minutes, "gru_proj_mean_abs": gate})
            if ep >= EPOCHS - SWA_LAST:
                torch.save(model.state_dict(), WORK / f"{TAG}_{name}_ep{ep}.pt")
            log(f"[{name}] ep{ep} loss {hist[-1]['loss']:.4f} | gold {au:.4f} (log only) | "
                f"{minutes:.1f} min" + (f" | |gru_proj| {gate:.2e}" if head == "seq" else "")
                + (f" | non-finite {skipped}" if skipped else ""))
        if ddp:
            dist.barrier()

    finalize_job(model, ctx, job, dev, rank, world, oof_uids, hist)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def finalize_job(model, ctx, job, dev, rank, world, oof_uids, hist):
    """Mean of the last SWA_LAST epochs' weights; score gold and this fold's OOF studies."""
    name, head = job["name"], job["head"]
    files = [WORK / f"{TAG}_{name}_ep{e}.pt" for e in range(EPOCHS - SWA_LAST, EPOCHS)]
    sds = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    avg = {k: (sum(sd[k].double() for sd in sds) / len(sds)).to(sds[0][k].dtype)
           if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
    model.load_state_dict(avg)
    spec = EVAL_SPEC[head]
    gp = dist_predict(model, ctx, ctx["gold_uids"], spec["gold"], dev, rank, world)
    op = dist_predict(model, ctx, oof_uids, spec["oof"], dev, rank, world)
    gp62 = dist_predict(model, ctx, ctx["gold_uids"], ("single", 62), dev, rank, world) \
        if head == "seq" else None
    if rank != 0:
        return
    # Gold labels are real 0/1 and carry no extraction prior, so no mask there.
    g_au, g_aucs, g_P, g_Y = score(gp, ctx["gold_uids"], ctx["labels"])
    # OOF is scored BOTH ways. `oof_auc_inf` (informative cells only) is the pre-registered
    # decision metric and is computed identically for both arms, whether or not that arm
    # trained with the mask. `oof_auc` keeps the old all-cells definition for continuity
    # with arm14/arm16 and does NOT decide.
    o_au, o_aucs, o_P, o_Y = score(op, oof_uids, ctx["labels"])
    lm = ctx.get("lab_mask_all")
    o_inf, o_inf_aucs = (o_au, o_aucs) if lm is None else \
        macro_auc(o_Y, o_P, np.stack([lm[u] for u in oof_uids]))
    res = {"job": name, "fold": job["fold"], "head": head, "swa_epochs": list(range(EPOCHS - SWA_LAST, EPOCHS)),
           "gold_auc": g_au, "gold_aucs": g_aucs, "gold_spec": list(spec["gold"]),
           "oof_auc": o_au, "oof_aucs": o_aucs, "oof_spec": list(spec["oof"]), "n_oof": len(oof_uids),
           "oof_auc_inf": o_inf, "oof_aucs_inf": o_inf_aucs, "masked_train": bool(job.get("mask")),
           "hist": hist}
    if gp62 is not None:
        res["gold_auc_single62"] = score(gp62, ctx["gold_uids"], ctx["labels"])[0]
    torch.save({"model": avg, "arch": ARCH, "res": RES, "lab": LAB, "recipe": RECIPE,
                "head": head, "depth": int(ctx["masks"].shape[1]), "gru_hidden": GRU_HIDDEN,
                "eval_spec": list(spec["gold"]), "fold": job["fold"], "gold_auc": g_au,
                "oof_auc": o_au, "epoch": res["swa_epochs"]}, WORK / f"{TAG}_{name}_swa.pt")
    np.savez(WORK / f"{TAG}_{name}_preds.npz", gold_ids=np.array(ctx["gold_uids"]), gold_pred=g_P,
             gold_truth=g_Y, oof_ids=np.array(oof_uids), oof_pred=o_P, oof_truth=o_Y)
    for f in files:
        f.unlink(missing_ok=True)
    json.dump(res, open(WORK / f"{TAG}_{name}_result.json", "w"), indent=1)
    extra = f" | gold single-62 bag {res['gold_auc_single62']:.4f}" if gp62 is not None else ""
    log(f"=== [{name}] DONE: OOF informative-cell macro-AUC {o_inf:.4f} <- DECIDES | "
        f"OOF all-cells {o_au:.4f} | gold {g_au:.4f} (not used){extra} ===")


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


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    ids, masks = load_index()
    train_uids, gold_uids, labels = load_labels(ids)
    fold_of = make_folds(train_uids)
    json.dump(fold_of, open(WORK / f"{TAG}_folds.json", "w"))
    canon_arrays, masks_canon = build_canon(ids, masks)
    lab_mask_all = build_label_mask(labels, train_uids + gold_uids)
    ctx = {"id2row": {u: i for i, u in enumerate(ids)}, "masks": masks, "train_uids": train_uids,
           "gold_uids": gold_uids, "labels": labels, "fold_of": fold_of,
           "canon_arrays": canon_arrays, "masks_canon": masks_canon,
           "lab_mask_all": lab_mask_all,
           "port": 29500 + random.randint(0, 900)}
    n_gpu = torch.cuda.device_count()
    world = int(os.environ.get("ARM10_WORLD", max(n_gpu, 1)))
    log(f"{TAG} | arch {ARCH} | gpus {n_gpu} | train {len(train_uids)} in {N_FOLDS} folds "
        f"{np.bincount(list(fold_of.values())).tolist()} | gold {len(gold_uids)}")

    if world > 1 and not run_children(spawn_canary, (), 1, 180, "spawn canary"):
        raise SystemExit("child processes cannot start under " + START_METHOD)

    r = run_job({"idx": 0, "name": "f3_noaug", "fold": 3, "head": "attn",
                 "canon": False, "mask": False, "aug": False}, world, ctx)
    log(f"[f3_noaug] DONE: oof {r['oof_auc']:.4f} gold {r.get('gold_auc')} (single job, no aug ablation)")
    print("ARM_COMPLETE", flush=True)


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
