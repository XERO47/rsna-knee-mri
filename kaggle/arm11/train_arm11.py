"""arm11: arm10's exact pipeline with CoAtNet-RMLP-2 at 384 -- the second ensemble member.

WHY. arm10 (ConvNeXt-small, 336) reached gold 0.8870 and reproduced exactly from
DICOM. The 0.941 public blend routes most weight to a CoAtNet branch, and an
ensemble gains most from members that differ: a hybrid conv/attention backbone
at a different resolution is the most diverse member available on the same corpus.

WHY 384. CoAtNet's total stride is 32 and 336/32 = 10.5, so it cannot run at the
corpus's native 336 (arm10 probe: rel-pos shape errors). Windows are bilinearly
upsampled 336 -> 384, as Raptor did; inference applies the same resize because
the checkpoint records res=384.

Everything else -- data, labels, windows, loss, schedule, DDP launch, budget,
resume, top-3 weight averaging -- is arm10's code unchanged; see its docstring.
"""
import os
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")   # device_count() must not init CUDA
# NCCL peer-to-peer between cloud T4s can deadlock on the first all-reduce; routing
# through host memory costs little at this model size and removes that failure mode.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import faulthandler
import sys
# Kaggle shows no log until a kernel ends, so a silent stall is indistinguishable
# from slow progress. Dump every thread's stack every 15 min: a hang then says where.
faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)


def arm_child():
    """Every child starts as a fresh spawned interpreter, which already armed its own
    stack-dump timer when it imported this module. Do NOT re-arm it here: that is what
    froze probe v2 -- under fork, the child inherited the timer's held lock but not its
    thread, so re-arming blocked forever. Just cap the intra-op pool on a 4-vCPU box."""
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

MODE = os.environ.get("ARM10_MODE", "train")          # "probe" | "train"

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]

# Probe v3 (2x T4, K=12, 336 px): convnext_small bs4 -> 2.8 GB peak, 0.92 s/step,
# ~17 min/epoch on one T4, ~9 on two. CoAtNet-RMLP-2 cannot run at 336 at all: total
# stride 32 and 336/32 = 10.5, which is why Raptor upsampled to 384. CoAtNet@384 is
# ~2.5-3x the compute (~25-30 min/epoch on 2x T4), so ConvNeXt proves the pipeline
# first and CoAtNet follows as the heavier ensemble member.
ARCH = os.environ.get("ARM10_ARCH", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
GRAD_CKPT = os.environ.get("ARM10_GRAD_CKPT", "1") == "1"   # CoAtNet@384: ~3 GB/study even with it (Raptor)
PROBE_ARCHS = ["coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"]
PROBE_BS = [2, 3, 4]
PROBE_STEPS = 12
PRETRAINED = os.environ.get("ARM10_PRETRAINED", "1") == "1"

RES = int(os.environ.get("ARM10_RES", "384"))    # CoAtNet needs a multiple of 32
K_TRAIN = int(os.environ.get("ARM10_K", "12"))
K_EVAL = 24               # per-epoch gold eval
K_FINAL = 42              # every window position a 44-slice stack holds
# arm11 probe (384, grad ckpt, 1x T4): bs2 6.0 GB 1.63 s/step | bs3 8.5 GB 2.56 s |
# bs4 10.9 GB 3.66 s -> ~59-66 min/epoch per GPU, compute-bound. bs4 keeps the
# effective batch (8 across 2 ranks) identical to arm10, so the members compare.
BS_PER_GPU = int(os.environ.get("ARM10_BS", "4"))
EPOCHS = int(os.environ.get("ARM10_EPOCHS", "10"))   # also the OneCycle horizon
BB_LR, HEAD_LR, WD = 3e-5, 1e-3, 0.02
PCT_START, CLIP = 0.15, 3.0
GAIN_AUG = 0.20
TOPK = 3
TIME_BUDGET_S = float(os.environ.get("ARM10_BUDGET_S", str(11.0 * 3600)))
SEED = 42
N_WORKERS = int(os.environ.get("ARM10_WORKERS", "2"))  # per rank
PROGRESS_S = 30
# spawn, not fork. Kaggle executes script kernels as a real file (tracebacks show
# /kaggle/src/script.py <module>), so spawned children can re-import this module.
# fork cost two probe sessions: children inherit the parent's threads' lock state
# without the threads (probe v1: fork after loading pretrained weights; probe v2:
# the faulthandler timer's lock). A spawned child is a fresh interpreter.
START_METHOD = os.environ.get("ARM10_START", "spawn")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

INPUT_ROOT = Path(os.environ.get("ARM10_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("ARM10_WORK", "/kaggle/working"))


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
    """Both published parts as one (4407, 44, 336, 336) view. Row order is part 1
    then part 2, matching the id concatenation -- never sort one independently."""

    def __init__(self):
        a = np.load(find_file("all_vols.npy"), mmap_mode="r")
        b = np.load(find_file("extra_vols.npy"), mmap_mode="r")
        self.a, self.b, self.n_a = a, b, a.shape[0]
        self.depth = a.shape[1]

    def study(self, row, slices):
        src = self.a[row] if row < self.n_a else self.b[row - self.n_a]
        return np.asarray(src[slices])


def load_index():
    ids = np.concatenate([np.load(find_file("all_ids.npy"), allow_pickle=True).astype(str),
                          np.load(find_file("extra_ids.npy"), allow_pickle=True).astype(str)])
    masks = np.concatenate([np.load(find_file("all_masks.npy")),
                            np.load(find_file("extra_masks.npy"))])
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
    train_uids = [u for u in pool.StudyInstanceUID.astype(str) if u in idset and u not in goldset]
    labels = {}
    for df in (pool, gold):
        for u, vals in zip(df.StudyInstanceUID.astype(str), df[LAB].to_numpy(np.float32)):
            labels[u] = vals
    missing = [u for u in train_uids + gold_uids if not np.isfinite(labels[u]).all()]
    if missing:
        raise RuntimeError(f"{len(missing)} studies have NaN labels, e.g. {missing[0]}")
    if len(gold_uids) != len(gold):
        raise RuntimeError(f"only {len(gold_uids)}/{len(gold)} gold studies are in the corpus")
    return train_uids, gold_uids, labels


class StudyWindows(Dataset):
    """Bag of 2.5D windows per study: three physically adjacent slices -> RGB.
    Window-centre logic mirrors Raptor's so eval matches its public inference."""

    def __init__(self, uids, id2row, masks, labels, k, train):
        self.uids, self.id2row, self.masks = uids, id2row, masks
        self.labels, self.k, self.train = labels, k, train
        self.corpus = None

    def __len__(self):
        return len(self.uids)

    def centers(self, row):
        depth = self.masks.shape[1]
        valid = np.flatnonzero(self.masks[row] > 0)
        if len(valid) < 3:
            valid = np.arange(min(3, depth))
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi)]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, depth - 2))]
        if self.train:
            pool = cs * (self.k // len(cs) + 1)
            random.shuffle(pool)
            return pool[:self.k]
        idx = np.linspace(0, len(cs) - 1, self.k).round().astype(int)
        return [cs[i] for i in idx]

    def __getitem__(self, i):
        if self.corpus is None:                 # lazily, once per worker
            self.corpus = Corpus()
        uid = self.uids[i]
        row = self.id2row[uid]
        cs = [max(1, min(c, self.corpus.depth - 2)) for c in self.centers(row)]
        need = sorted({s for c in cs for s in (c - 1, c, c + 1)})
        pos = {s: j for j, s in enumerate(need)}
        stack = self.corpus.study(row, need)    # one read of just the needed slices
        tri = np.stack([stack[[pos[c - 1], pos[c], pos[c + 1]]] for c in cs])
        x = torch.from_numpy(tri).float().div_(255.0)          # (K, 3, H, W)
        if x.shape[-1] != RES:
            x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
        if self.train and GAIN_AUG > 0:
            x = (x * (1.0 + (random.random() - 0.5) * GAIN_AUG)).clamp_(0, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, torch.from_numpy(self.labels[uid])


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


# ------------------------------------------------------------------- model --
def build_backbone(arch, pretrained):
    if arch == "tiny_test":                     # local wiring tests only
        net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        net.num_features = 16
        return net
    import timm
    return timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=3,
                             global_pool="avg")


class RaptorClassifier(nn.Module):
    """arm6's per-finding attention MIL, with Raptor's LayerNorm + dropout.
    Module names match Raptor's checkpoints (norm / att / clsW / clsb)."""

    def __init__(self, backbone, n=len(LAB), drop=0.2):
        super().__init__()
        f = backbone.num_features
        self.backbone = backbone
        self.norm = nn.LayerNorm(f)
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, f))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(self, x):
        B, K = x.shape[:2]
        h = self.norm(self.backbone(x.flatten(0, 1)).view(B, K, -1))
        a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def make_model(arch, pretrained, dev):
    bb = build_backbone(arch, pretrained)
    if arch != "tiny_test" and GRAD_CKPT:
        try:
            bb.set_grad_checkpointing(True)
        except Exception as e:
            log(f"grad checkpointing unavailable for {arch}: {e}")
    return RaptorClassifier(bb).to(dev)


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
def evaluate(model, loader, dev):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            P.append(torch.sigmoid(model(x.to(dev, non_blocking=True)).float()).cpu().numpy())
        Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    au, aucs = macro_auc(Y, P)
    return au, aucs, P, Y


# ------------------------------------------------------------------- probe --
def probe(ctx):
    dev = torch.device("cuda:0")
    ds = StudyWindows(ctx["train_uids"][:256], ctx["id2row"], ctx["masks"],
                      ctx["labels"], K_TRAIN, train=True)
    pw = torch.from_numpy(ctx["pos_weight"]).to(dev)
    report = []
    for arch in PROBE_ARCHS:
        for bs in PROBE_BS:
            model = opt = None
            rec = {"arch": arch, "bs": bs}
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                model = make_model(arch, PRETRAINED, dev)
                opt = make_opt(model)
                scaler = torch.amp.GradScaler("cuda")
                dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4,
                                drop_last=True, collate_fn=collate, pin_memory=True)
                model.train()
                t_data = t_step = 0.0
                n = 0
                t_prev = time.time()
                for step, (x, y) in enumerate(dl):
                    if step >= PROBE_STEPS:
                        break
                    t_got = time.time()
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = F.binary_cross_entropy_with_logits(
                            model(x.to(dev)).float(), y.to(dev), pos_weight=pw)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                    scaler.step(opt)
                    scaler.update()
                    torch.cuda.synchronize(dev)
                    t_done = time.time()
                    if step >= 3:                   # skip warm-up / cudnn autotune
                        t_data += t_got - t_prev
                        t_step += t_done - t_got
                        n += 1
                    t_prev = time.time()
                s_step = t_step / max(n, 1)
                s_data = t_data / max(n, 1)
                epoch_min_1gpu = len(ctx["train_uids"]) / bs * (s_step + s_data) / 60
                rec.update(peak_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                           s_step=s_step, s_data=s_data, epoch_min_1gpu=epoch_min_1gpu,
                           epoch_min_2gpu_est=epoch_min_1gpu / 1.8, loss=float(loss))
                # eval cost at the final window count
                model.eval()
                ev = StudyWindows(ctx["gold_uids"][:8], ctx["id2row"], ctx["masks"],
                                  ctx["labels"], K_FINAL, train=False)
                el = DataLoader(ev, batch_size=2, num_workers=2, collate_fn=collate)
                t0 = time.time()
                with torch.no_grad():
                    for xe, _ in el:
                        with torch.autocast("cuda", dtype=torch.float16):
                            model(xe.to(dev))
                torch.cuda.synchronize(dev)
                rec["eval_s_per_study_k42"] = (time.time() - t0) / len(ev)
            except torch.cuda.OutOfMemoryError:
                rec["oom"] = True
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            finally:
                del model, opt
                torch.cuda.empty_cache()
            log(f"PROBE {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})}")
            report.append(rec)
    json.dump(report, open(WORK / "arm11_probe.json", "w"), indent=1)

    ok = [r for r in report if "s_step" in r and r["arch"] == ARCH]
    log("\n=== PROBE SUMMARY (steady-state, K=12 windows/study, 336 px) ===")
    for r in report:
        if "s_step" in r:
            log(f"  {r['arch'][:40]:40s} bs{r['bs']}  peak {r['peak_gb']:.1f} GB  "
                f"step {r['s_step']:.2f}s  data {r['s_data']:.2f}s  "
                f"epoch 1xT4 {r['epoch_min_1gpu']:.0f} min  ~2xT4 {r['epoch_min_2gpu_est']:.0f} min  "
                f"eval {r['eval_s_per_study_k42']:.2f} s/study@K42")
        else:
            log(f"  {r['arch'][:40]:40s} bs{r['bs']}  {'OOM' if r.get('oom') else r.get('error')}")
    if ok:
        best = max(ok, key=lambda r: r["bs"])
        log(f"largest bs that fits for {ARCH}: {best['bs']}")


# ------------------------------------------------------------------- train --
def train_worker(rank, world, ctx):
    arm_child()
    ddp = world > 1
    backend = os.environ.get("ARM10_BACKEND", "nccl")
    if ddp:
        dist.init_process_group(backend, init_method=f"tcp://127.0.0.1:{ctx['port']}",
                                rank=rank, world_size=world,
                                timeout=datetime.timedelta(hours=2))
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        dev = torch.device(f"cuda:{rank}")
    else:
        dev = torch.device("cpu")
    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED)                      # identical init on every rank

    if ddp and rank != 0:
        dist.barrier()                           # wait for rank 0 to cache pretrained weights
    model = make_model(ARCH, PRETRAINED, dev)
    if ddp and rank == 0:
        dist.barrier()
    net = DDP(model, device_ids=[rank] if dev.type == "cuda" else None,
              static_graph=True) if ddp else model

    tds = StudyWindows(ctx["train_uids"], ctx["id2row"], ctx["masks"], ctx["labels"],
                       K_TRAIN, train=True)
    sampler = DistributedSampler(tds, world, rank, shuffle=True, seed=SEED,
                                 drop_last=True) if ddp else None
    tl = DataLoader(tds, batch_size=BS_PER_GPU, sampler=sampler, shuffle=sampler is None,
                    num_workers=N_WORKERS, drop_last=True, collate_fn=collate,
                    pin_memory=dev.type == "cuda", persistent_workers=N_WORKERS > 0)
    vl = None
    if rank == 0:
        vds = StudyWindows(ctx["gold_uids"], ctx["id2row"], ctx["masks"], ctx["labels"],
                           K_EVAL, train=False)
        vl = DataLoader(vds, batch_size=2, shuffle=False, num_workers=N_WORKERS,
                        collate_fn=collate)

    opt = make_opt(model)
    steps_per_epoch = len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[BB_LR, HEAD_LR], total_steps=EPOCHS * steps_per_epoch,
        pct_start=PCT_START)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    pw = torch.from_numpy(ctx["pos_weight"]).to(dev)

    state = {"epoch": -1, "best": -1.0, "hist": [], "topk": []}
    resume = find_file("arm11_last.pt")
    if resume is not None:
        ck = torch.load(resume, map_location=dev, weights_only=False)
        if ck["arch"] != ARCH or ck["epochs"] != EPOCHS or ck["steps_per_epoch"] != steps_per_epoch:
            raise RuntimeError("resume checkpoint was made under a different schedule")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        if rank == 0:                            # carry earlier sessions' artefacts forward
            for f in list(resume.parent.glob("arm11_topk_ep*.pt")) + \
                    [resume.parent / "arm11_best.pt"]:
                if f.exists() and not (WORK / f.name).exists():
                    shutil.copy(f, WORK / f.name)
        log(f"resumed after epoch {state['epoch']} (best {state['best']:.4f})", rank)

    log(f"world {world} | {len(tds)} train studies | {steps_per_epoch} steps/epoch x "
        f"{EPOCHS} epochs | bs/gpu {BS_PER_GPU} | K {K_TRAIN} | res {RES}", rank)

    t_session = time.time()
    epoch_times = []
    for ep in range(state["epoch"] + 1, EPOCHS):
        # stop cleanly if this epoch cannot finish inside the session budget
        need = (np.mean(epoch_times) * 1.15) if epoch_times else 0.0
        stop = torch.tensor([float(time.time() - t_session + need > TIME_BUDGET_S)], device=dev)
        if ddp:
            dist.broadcast(stop, 0)
        if stop.item() > 0:
            log(f"session budget: stopping before epoch {ep}; attach this output to resume", rank)
            break
        if sampler is not None:
            sampler.set_epoch(ep)
        net.train()
        t_ep = last = time.time()
        tot, nb, skipped = 0.0, 0, 0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = net(x)
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
                log(f"  ep{ep} {nb}/{steps_per_epoch} loss {tot / max(nb - skipped, 1):.4f} "
                    f"lr {opt.param_groups[1]['lr']:.2e} | {el / 60:.1f} min, eta "
                    f"{(steps_per_epoch - nb) * el / nb / 60:.1f} min"
                    + (f" | non-finite {skipped}" if skipped else ""))
                last = time.time()
        epoch_times.append(time.time() - t_ep)

        if rank == 0:
            au, aucs, P, Y = evaluate(model, vl, dev)
            state["hist"].append({"ep": ep, "loss": tot / max(nb - skipped, 1),
                                  "gold_auc": au, "minutes": epoch_times[-1] / 60})
            star = ""
            if au > state["best"]:
                state["best"], star = au, "  <- best"
                torch.save({"model": model.state_dict(), "arch": ARCH, "res": RES,
                            "lab": LAB, "epoch": ep, "gold_auc": au, "aucs": aucs},
                           WORK / "arm11_best.pt")
                json.dump(aucs, open(WORK / "arm11_best_per_finding.json", "w"), indent=1)
            # top-K epochs on disk, so weight averaging survives a session boundary
            state["topk"].append({"ep": ep, "gold_auc": au})
            state["topk"].sort(key=lambda t: -t["gold_auc"])
            keep = {t["ep"] for t in state["topk"][:TOPK]}
            if ep in keep:
                torch.save(model.state_dict(), WORK / f"arm11_topk_ep{ep}.pt")
            for t in state["topk"][TOPK:]:
                (WORK / f"arm11_topk_ep{t['ep']}.pt").unlink(missing_ok=True)
            state["topk"] = state["topk"][:TOPK]
            state["epoch"] = ep
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "state": state, "arch": ARCH, "epochs": EPOCHS,
                        "steps_per_epoch": steps_per_epoch}, WORK / "arm11_last.pt")
            json.dump(state["hist"], open(WORK / "arm11_history.json", "w"), indent=1)
            log(f"ep{ep} loss {state['hist'][-1]['loss']:.4f} | GOLD macro-AUC {au:.4f}{star} "
                f"| {epoch_times[-1] / 60:.1f} min" + (f" | non-finite {skipped}" if skipped else ""))
        if ddp:
            dist.barrier()

    if rank == 0 and state["epoch"] == EPOCHS - 1:
        finalize(model, ctx, dev, state)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def finalize(model, ctx, dev, state):
    """Best epoch and top-K weight average, both re-scored at K=42 windows."""
    fl = DataLoader(StudyWindows(ctx["gold_uids"], ctx["id2row"], ctx["masks"],
                                 ctx["labels"], K_FINAL, train=False),
                    batch_size=2, shuffle=False, num_workers=N_WORKERS, collate_fn=collate)
    out = {}
    best = torch.load(WORK / "arm11_best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(best["model"])
    au, aucs, P, Y = evaluate(model, fl, dev)
    out["best_epoch"] = {"epoch": best["epoch"], "auc_k42": au, "aucs": aucs}
    np.savez(WORK / "arm11_gold_best.npz", pred=P, truth=Y, ids=np.array(ctx["gold_uids"]))

    files = [WORK / f"arm11_topk_ep{t['ep']}.pt" for t in state["topk"]]
    files = [f for f in files if f.exists()]
    if len(files) > 1:
        sds = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
        avg = {k: (sum(sd[k].double() for sd in sds) / len(sds)).to(sds[0][k].dtype)
               if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
        model.load_state_dict(avg)
        au_s, aucs_s, P_s, _ = evaluate(model, fl, dev)
        out["swa"] = {"epochs": [t["ep"] for t in state["topk"]], "auc_k42": au_s, "aucs": aucs_s}
        torch.save({"model": avg, "arch": ARCH, "res": RES, "lab": LAB,
                    "epoch": out["swa"]["epochs"], "gold_auc": au_s, "aucs": aucs_s},
                   WORK / "arm11_swa.pt")
        np.savez(WORK / "arm11_gold_swa.npz", pred=P_s, truth=Y, ids=np.array(ctx["gold_uids"]))
    json.dump(out, open(WORK / "arm11_final.json", "w"), indent=1)
    log("\n=== FINAL (gold, K=42 windows) ===")
    log(f"best epoch {out['best_epoch']['epoch']}: {out['best_epoch']['auc_k42']:.4f}")
    if "swa" in out:
        log(f"top-{len(files)} weight average {out['swa']['epochs']}: {out['swa']['auc_k42']:.4f}")
    src = out.get("swa", out["best_epoch"]) if out.get("swa", {}).get("auc_k42", -1) > \
        out["best_epoch"]["auc_k42"] else out["best_epoch"]
    for k, v in src["aucs"].items():
        log(f"   {k:18s} {v:.4f}")
    log("reference: arm6 0.8449 gold / 0.854 LB")


# -------------------------------------------------------------------- main --
def main():
    WORK.mkdir(parents=True, exist_ok=True)
    ids, masks = load_index()
    train_uids, gold_uids, labels = load_labels(ids)
    prev = np.clip(np.stack([labels[u] for u in train_uids]).mean(0), 0.03, 0.7)
    ctx = {"id2row": {u: i for i, u in enumerate(ids)}, "masks": masks,
           "train_uids": train_uids, "gold_uids": gold_uids, "labels": labels,
           "pos_weight": np.clip((1 - prev) / prev, 1, 10).astype(np.float32),
           "port": 29500 + random.randint(0, 999)}
    n_gpu = torch.cuda.device_count()
    log(f"mode {MODE} | arch {ARCH} | gpus {n_gpu} | corpus {len(ids)} studies "
        f"| train {len(train_uids)} | gold {len(gold_uids)}")
    log("pos_weight " + " ".join(f"{n}={w:.1f}" for n, w in zip(LAB, ctx["pos_weight"].tolist())))

    # The parent process must stay free of torch compute: the first probe run forked
    # DDP ranks after loading two pretrained models here, and the children deadlocked
    # (fork after the intra-op thread pool had started). Every heavy step now runs in a
    # forked child, each with a deadline, so a stall costs minutes, not a session.
    # 3-minute canary before any heavy work: if child processes cannot start, fail now.
    if not run_children(spawn_canary, (), 1, 180, "spawn canary"):
        raise SystemExit("child processes cannot start under " + START_METHOD)
    log(f"spawn canary PASS ({START_METHOD})")

    if MODE == "probe":
        run_children(probe_child, (ctx,), 1, 45 * 60, "single-GPU probe")
        if n_gpu > 1:
            smoke = dict(ctx)
            smoke["train_uids"] = train_uids[:16]
            log("DDP smoke test: 2 ranks, bs/gpu 2, 4 steps")
            ok = run_children(smoke_worker, (2, smoke), 2, 15 * 60, "DDP smoke")
            log(f"DDP smoke {'PASS' if ok else 'FAIL'}")
        log("probe DONE")
        return

    world = int(os.environ.get("ARM10_WORLD", max(n_gpu, 1)))   # override: local DDP tests
    if world > 1:
        if not run_children(train_worker, (world, ctx), world, TIME_BUDGET_S + 3600, "train"):
            raise SystemExit("training failed or stalled; see child stack dumps above")
    else:
        train_worker(0, 1, ctx)


def spawn_canary(rank):
    print(f"[canary] child {rank} alive, pid {os.getpid()}, torch {torch.__version__}, "
          f"cuda visible {torch.cuda.device_count()}", flush=True)


def probe_child(rank, ctx):
    arm_child()
    probe(ctx)


def run_children(fn, args, nprocs, deadline_s, label):
    """start_processes with a hard deadline. Returns True only on clean completion."""
    pctx = mp.start_processes(fn, args=args, nprocs=nprocs, join=False,
                              start_method=START_METHOD)
    t0 = time.time()
    try:
        while not pctx.join(timeout=15):
            if time.time() - t0 > deadline_s:
                log(f"{label}: exceeded {deadline_s / 60:.0f} min -- terminating children "
                    f"(faulthandler stack dumps above show where they were)")
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


def smoke_worker(rank, world, ctx):
    """Four real DDP steps through the exact training path: NCCL, static_graph with
    gradient checkpointing, GradScaler. Catches the failures that would otherwise
    only surface an hour into the real run."""
    arm_child()
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{ctx['port']}",
                            rank=rank, world_size=world, timeout=datetime.timedelta(minutes=10))
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    torch.manual_seed(SEED)
    if rank != 0:
        dist.barrier()
    model = make_model(ARCH, PRETRAINED, dev)
    if rank == 0:
        dist.barrier()
    net = DDP(model, device_ids=[rank], static_graph=True)
    ds = StudyWindows(ctx["train_uids"], ctx["id2row"], ctx["masks"], ctx["labels"],
                      K_TRAIN, train=True)
    dl = DataLoader(ds, batch_size=2, sampler=DistributedSampler(ds, world, rank),
                    num_workers=1, drop_last=True, collate_fn=collate)
    opt = make_opt(model)
    scaler = torch.amp.GradScaler("cuda")
    pw = torch.from_numpy(ctx["pos_weight"]).to(dev)
    for step, (x, y) in enumerate(dl):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(net(x.to(dev)).float(), y.to(dev),
                                                      pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if rank == 0:
            log(f"  DDP smoke step {step} loss {float(loss):.4f}")
        if step >= 3:
            break
    # ranks must hold identical weights after synchronised steps
    p = next(model.parameters()).detach().float().sum()
    ps = [torch.zeros_like(p) for _ in range(world)]
    dist.all_gather(ps, p)
    if rank == 0:
        log(f"  DDP weight sync check: {[round(float(v), 4) for v in ps]} "
            f"-> {'OK' if len({round(float(v), 4) for v in ps}) == 1 else 'DIVERGED'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
