"""arm17: a real MCL measurement for CoAtNet, and the CoAtNet OOF we have never had.

WHY. MCL is our weakest finding on the large set (OOF .795 over 4,349 studies) and the
per-member numbers split cleanly by architecture: ConvNeXt arm10 .846 and arm12 .850,
CoAtNet arm11 .927 and arm13 .941. Corpus width does not explain it -- within an
architecture, going 44 -> 64 slices moves MCL only +.004 and +.014.

BUT ALL FOUR OF THOSE NUMBERS COME FROM THE SAME 58 GOLD STUDIES, WHICH HOLD 9 MCL
POSITIVES. Two CoAtNets scored against the same 9 cases is one measurement made twice,
not a replication; at n+ = 9 the AUC standard error is about .10 and the gap is one SE.
We made exactly this mistake once already today, building a whole research direction on
gold-58 medial/lateral gaps that turned out to be a third their apparent size on OOF.

WHAT THIS RUN BUYS. One CoAtNet job on the fold-0 split scores MCL on ~131 positives
instead of 9 -- a 14x larger positive count -- and produces out-of-fold predictions for
those 870 studies from a CoAtNet. That second product is the blocker on per-finding member
routing: we have OOF for ConvNeXt (arm14c) and nothing for CoAtNet, so we cannot currently
tell whether routing MCL to CoAtNet is real or an artefact of a small gold set.

WHAT IT DOES NOT DO. It does not settle routing by itself -- one fold, one architecture.
It settles whether the MCL effect survives contact with a properly sized sample, which is
the question that decides if routing is worth any further compute.

CRASH PRE-EMPTED. arm13 trained all 10 epochs and then died inside finalize -> evaluate,
OOM: the gold spec for the attention head is ("single", 62), so the final eval pushes
62 windows x 2 studies = 124 CoAtNet images at 384 through one backbone call. The chunked
backbone from submit11 is ported here (CNN_CHUNK, eval path only, numerically identical in
eval mode). Training stays whole so DDP and autograd are untouched.

CANON. Takes its canonicalisation setting from ARM17_CANON so it can match whatever arm16
decides; default off. Everything else is arm16/arm14: cache64, K=12 windows, per-finding
attention MIL, pos_weight BCE, AdamW 3e-5/1e-3, OneCycle 10 epochs, top-3 weight average,
2x T4 DDP, spawn, job-level resume.
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

TAG = "arm17"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]

ARCH = os.environ.get("ARM10_ARCH", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
PRETRAINED = os.environ.get("ARM10_PRETRAINED", "1") == "1"
RES = int(os.environ.get("ARM10_RES", "384"))
K_TRAIN = int(os.environ.get("ARM10_K", "12"))
BS_PER_GPU = int(os.environ.get("ARM10_BS", "4"))
EPOCHS = int(os.environ.get("ARM10_EPOCHS", "10"))
BB_LR, HEAD_LR, WD = 3e-5, 1e-3, 0.02
PCT_START, CLIP = 0.15, 3.0
GAIN_AUG = 0.20
SEED = 42
N_WORKERS = int(os.environ.get("ARM10_WORKERS", "2"))
PROGRESS_S = 30
START_METHOD = os.environ.get("ARM10_START", "spawn")

N_FOLDS = int(os.environ.get("ARM14_FOLDS", "5"))
CANON = os.environ.get("ARM17_CANON", "0") == "1"
# CoAtNet@384 at bs4 x K=12 is 48 images per GPU per step. arm13's probe measured 10.9 GB
# WITH gradient checkpointing; without it the first forward OOMs on a 16 GB T4 -- which is
# exactly how arm17 v1 died, 12 s into the job, twice. arm14/arm16 never needed this
# because ConvNeXt-small @336 fits comfortably, so the flag did not survive the port.
GRAD_CKPT = os.environ.get("ARM10_GRAD_CKPT", "1") == "1"
FOLD_SEED = 42
SWA_LAST = 3
ADOPT_MARGIN = 0.002
GRU_HIDDEN = 256
CNN_CHUNK = int(os.environ.get("ARM17_CNN_CHUNK", "32"))

# evaluation specs: ("single", K) = one bag of K evenly spaced windows;
#                   ("subsets", K, R) = R phase-shifted evenly spaced bags of K, probs averaged
EVAL_SPEC = {
    "attn": {"epoch": ("single", 24), "oof": ("single", 24), "gold": ("single", 62)},
    "seq":  {"epoch": ("subsets", 12, 2), "oof": ("subsets", 12, 2), "gold": ("subsets", 12, 5)},
}
# arm14 set this to 2.5 h as a 2x margin over its ~75 min ConvNeXt jobs. CoAtNet@384 with
# gradient checkpointing runs 29 min/epoch, so 10 epochs needs ~4.8 h -- arm17 v2 hit the
# 150 min wall at ~ep5, was killed, restarted from scratch, and hit it again, burning
# ~4.8 GPU-h for zero output. Epoch checkpoints only start at ep7 (EPOCHS - SWA_LAST), so
# nothing was recoverable either time.
JOB_DEADLINE_S = float(os.environ.get("ARM14_JOB_DEADLINE_S", str(6.5 * 3600)))

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


class StudyWindows(Dataset):
    """train: one bag of K_TRAIN random centres, SORTED by depth (order matters to the GRU;
    the attention head is permutation-invariant, so sorting changes nothing for it).
    eval:  R bags per study per `spec`. Returns x (R,K,3,H,W), pos (R,K), y."""

    def __init__(self, uids, id2row, masks, labels, train, spec=None, canon=None):
        self.uids, self.id2row, self.masks = uids, id2row, masks
        self.labels, self.train, self.spec = labels, train, spec
        self.canon = canon
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
        if self.train and GAIN_AUG > 0:
            x = (x * (1.0 + (random.random() - 0.5) * GAIN_AUG)).clamp_(0, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        pos = torch.tensor(bags, dtype=torch.long)                      # (R, K)
        if self.train:
            x, pos = x[0], pos[0]
        return x, pos, torch.from_numpy(self.labels[uid])


def collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


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
        flat = x.flatten(0, 1)
        if self.training or flat.shape[0] <= CNN_CHUNK:
            feats = self.backbone(flat)
        else:
            # arm13 died here: final gold eval is K=62 windows x 2 studies = 124 CoAtNet
            # images at 384 in one backbone call, which does not fit a T4. Chunking is
            # numerically identical in eval mode (no batch-statistics dependence) and was
            # verified as such in submit11. Training is left whole so DDP/autograd and the
            # activation checkpointing pattern are untouched.
            feats = torch.cat([self.backbone(flat[i:i + CNN_CHUNK])
                               for i in range(0, flat.shape[0], CNN_CHUNK)])
        h = self.norm(feats.view(B, K, -1))
        if self.head == "seq":
            g, _ = self.gru(h + self.pos_emb(pos).to(h.dtype))
            h = h + self.gru_proj(g).to(h.dtype)
        a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def make_model(head, dev, depth):
    bb = build_backbone(ARCH, PRETRAINED)
    if ARCH != "tiny_test" and GRAD_CKPT:
        try:
            bb.set_grad_checkpointing(True)
        except Exception as e:
            log(f"grad checkpointing unavailable for {ARCH}: {e}")
    return KneeNet(bb, head, depth).to(dev)


def make_opt(model):
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return torch.optim.AdamW([{"params": bb, "lr": BB_LR},
                              {"params": head, "lr": HEAD_LR}], weight_decay=WD)


def macro_auc(Y, P):
    aucs = {}
    for j, name in enumerate(LAB):
        yj = (Y[:, j] > 0.5).astype(int)
        if len(set(yj)) > 1:
            aucs[name] = float(roc_auc_score(yj, P[:, j]))
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
    for x, pos, _ in dl:
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


def score(pred, uids, labels):
    P = np.stack([pred[u] for u in uids])
    Y = np.stack([labels[u] for u in uids])
    au, aucs = macro_auc(Y, P)
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
    fold_of = ctx["fold_of"]
    tr_uids = [u for u in ctx["train_uids"] if fold_of[u] != fold]
    oof_uids = [u for u in ctx["train_uids"] if fold_of[u] == fold]
    prev = np.clip(np.stack([ctx["labels"][u] for u in tr_uids]).mean(0), 0.03, 0.7)
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
                       canon=ctx.get("canon"))
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
        for x, pos, y in tl:
            x, pos, y = x.to(dev, non_blocking=True), pos.to(dev), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = net(x, pos)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pw)
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
    g_au, g_aucs, g_P, g_Y = score(gp, ctx["gold_uids"], ctx["labels"])
    o_au, o_aucs, o_P, o_Y = score(op, oof_uids, ctx["labels"])
    res = {"job": name, "fold": job["fold"], "head": head, "swa_epochs": list(range(EPOCHS - SWA_LAST, EPOCHS)),
           "gold_auc": g_au, "gold_aucs": g_aucs, "gold_spec": list(spec["gold"]),
           "oof_auc": o_au, "oof_aucs": o_aucs, "oof_spec": list(spec["oof"]), "n_oof": len(oof_uids),
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
    log(f"=== [{name}] DONE: OOF macro-AUC {o_au:.4f} (n={len(oof_uids)}, noisy labels) | "
        f"gold {g_au:.4f}{extra} ===")


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
    ctx = {"id2row": {u: i for i, u in enumerate(ids)}, "masks": masks, "train_uids": train_uids,
           "gold_uids": gold_uids, "labels": labels, "fold_of": fold_of,
           "canon_arrays": canon_arrays, "masks_canon": masks_canon,
           "port": 29500 + random.randint(0, 900)}
    n_gpu = torch.cuda.device_count()
    world = int(os.environ.get("ARM10_WORLD", max(n_gpu, 1)))
    log(f"{TAG} | arch {ARCH} | gpus {n_gpu} | train {len(train_uids)} in {N_FOLDS} folds "
        f"{np.bincount(list(fold_of.values())).tolist()} | gold {len(gold_uids)}")

    if world > 1 and not run_children(spawn_canary, (), 1, 180, "spawn canary"):
        raise SystemExit("child processes cannot start under " + START_METHOD)

    log(f"canonicalisation: {'ON' if CANON else 'OFF'} (ARM17_CANON)")
    r = run_job({"idx": 0, "name": "f0_coatnet", "fold": 0, "head": "attn",
                 "canon": CANON}, world, ctx)

    z = np.load(WORK / f"{TAG}_{r['job']}_preds.npz", allow_pickle=True)
    _, aucs = macro_auc(z["oof_truth"], z["oof_pred"])
    # arm14c fold-0, ConvNeXt, same split and same 870 studies -- the only fair comparator.
    ref = {"ACL": 0.8685, "MCL": 0.7949, "Medial Meniscus": 0.9131,
           "Lateral Meniscus": 0.8666, "Medial OA": 0.8793, "Lateral OA": 0.8461,
           "PF OA": 0.8347, "Effusion": 0.8607, "Synovitis": 0.8917,
           "Baker's": 0.9026, "Contusion": 0.8564, "Fracture": 0.8820}
    log("")
    log("=== CoAtNet vs ConvNeXt, fold-0 OOF (870 studies) ===")
    log("  NOTE: `ref` is ConvNeXt pooled over ALL 5 folds (4,349 studies), so it is the")
    log("  better-measured side. Treat per-finding deltas as indicative, not decisive.")
    log(f"    {'finding':18s} {'CoAtNet':>8} {'ConvNeXt':>9} {'delta':>8} {'n+':>5}")
    npos = (z["oof_truth"] > 0.5).sum(0)
    for j, nme in enumerate(LAB):
        if nme not in aucs:
            continue
        d = aucs[nme] - ref[nme]
        mark = "   <-- MCL, the question this run exists to answer" if nme == "MCL" else ""
        log(f"    {nme:18s} {aucs[nme]:8.4f} {ref[nme]:9.4f} {d:+8.4f} {int(npos[j]):5d}{mark}")
    json.dump({"coatnet_fold0_oof": aucs, "convnext_all_folds_oof": ref,
               "oof_macro": r["oof_auc"], "gold": r["gold_auc"]},
              open(WORK / f"{TAG}_compare.json", "w"), indent=1)
    log("")
    log(f"  macro OOF {r['oof_auc']:.4f} | gold-58 {r['gold_auc']:.4f} (reported, not used)")


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
