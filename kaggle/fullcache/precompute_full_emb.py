"""Encode every slice of the full-cache into a frozen-encoder embedding, once.

Reads the sharded full-cache (load_full.FullCache), runs a frozen timm encoder over
EVERY slice (3-channel triplet built from in-series neighbours), and writes a compact
CSR embedding set the variable-length transformer trains on:

  full_emb.npy   (Ttot, F) float16   -- pooled feature per slice, study-contiguous
  full_idx.npy   (nstudy, 2) int64    -- [offset, count]
  full_meta.npy  (Ttot, 4) float32    -- [plane, fluid, series_idx, pos_norm] (carried through)
  full_ids.npy   (nstudy,) str

144 GB of pixels -> ~1-2 GB of vectors; after this the pixels are only needed to re-embed
with a different encoder. GPU step (quota-free Kaggle is exhausted -> run on the pod).
"""
import os, time
from pathlib import Path
import numpy as np
import torch
import timm
from load_full import FullCache

ROOT = os.environ.get("FULL_ROOT", "/workspace/full")
OUT = Path(os.environ.get("FULL_EMB_OUT", "/workspace/full_emb"))
ARCH = os.environ.get("FULL_ENC_ARCH", "convnext_small.in12k_ft_in1k_384")
RES = int(os.environ.get("FULL_ENC_RES", "448"))
BS = int(os.environ.get("FULL_ENC_BS", "128"))          # slices per encoder batch
WORKERS = int(os.environ.get("FULL_ENC_WORKERS", "8"))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def triplets(sl, meta):
    """(K,H,W) uint8 + meta -> (K,3,H,W) float32 in [0,1]; neighbours stay inside a series."""
    K = sl.shape[0]
    series = meta[:, 2].astype(int)
    prev = np.arange(K) - 1; nxt = np.arange(K) + 1
    prev[0] = 0; nxt[-1] = K - 1
    prev[1:][series[1:] != series[:-1]] = np.arange(K)[1:][series[1:] != series[:-1]]   # no cross-series
    nxt[:-1][series[1:] != series[:-1]] = np.arange(K)[:-1][series[1:] != series[:-1]]
    x = np.stack([sl[prev], sl, sl[nxt]], 1).astype(np.float32) / 255.0                   # (K,3,H,W)
    return x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc = FullCache(ROOT)
    uids = fc.uids()
    enc = timm.create_model(ARCH, pretrained=True, num_classes=0, global_pool="avg").eval().to(dev)
    for p in enc.parameters(): p.requires_grad_(False)
    feat = enc.num_features
    total = sum(fc.count(u) for u in uids)
    log(f"device {dev} | enc {ARCH} feat {feat} RES {RES} | studies {len(uids)} slices {total}")

    emb = np.lib.format.open_memmap(OUT / "full_emb.npy", mode="w+", dtype=np.float16, shape=(total, feat))
    meta_all = np.zeros((total, 4), np.float32)
    index = np.zeros((len(uids), 2), np.int64)
    mean, std = MEAN.to(dev), STD.to(dev)
    off = 0; t0 = time.time(); last = time.time()
    for si, u in enumerate(uids):
        sl, mt = fc.study(u)
        x = torch.from_numpy(triplets(sl, mt))
        if x.shape[-1] != RES:
            x = torch.nn.functional.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
        feats = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            for b in range(0, x.shape[0], BS):
                xb = ((x[b:b + BS].to(dev) - mean) / std)
                feats.append(enc(xb).float().cpu())
        f = torch.cat(feats).numpy().astype(np.float16)
        k = f.shape[0]
        emb[off:off + k] = f; meta_all[off:off + k] = mt
        index[si] = (off, k); off += k
        if time.time() - last > 30 or si == len(uids) - 1:
            el = time.time() - t0
            log(f"  {si+1}/{len(uids)} studies | {off}/{total} slices | {el/60:.1f} min | "
                f"eta {el/(si+1)*(len(uids)-si-1)/60:.1f} min")
            last = time.time()
    emb.flush(); del emb
    np.save(OUT / "full_meta.npy", meta_all)
    np.save(OUT / "full_idx.npy", index)
    np.save(OUT / "full_ids.npy", np.asarray(uids, dtype="U"))
    log(f"DONE {total} slices -> {OUT}/full_emb.npy ({total*feat*2/1e9:.2f} GB) in {(time.time()-t0)/60:.1f} min")
    print("PRECOMPUTE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
