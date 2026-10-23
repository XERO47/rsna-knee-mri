import os
os.environ.setdefault("EMB_GROUP", "g2")
os.environ.setdefault("EMB_LIMIT", "0")

"""Kaggle GPU embed kernel: encode every slice of the attached full-cache shards into a
frozen DINOv2 ViT-L/14 embedding. Self-contained (single-file kernel). The full-cache
shard kernel outputs are attached as kernel_sources and mount under /kaggle/input/<slug>/,
so the cache is read IN PLACE -- no 160 GB download. Output (per attached group) is the
compact CSR embedding set the variable-length transformer trains on:

  full_emb_g{G}.npy   (Ttot, F) float16   -- pooled feature per slice, study-contiguous
  full_idx_g{G}.npy   (nstudy, 2) int64    -- [offset, count]
  full_meta_g{G}.npy  (Ttot, 4) float32    -- [plane, fluid, series_idx, pos_norm]
  full_ids_g{G}.npy   (nstudy,) str

Encoder is self-supervised (DINOv2) -> a different lineage from the arms' supervised
ImageNet ConvNeXt, the point being to decorrelate from the 0.904 ensemble.
"""
import glob, os, time
import numpy as np
import torch, timm

ROOT = os.environ.get("EMB_ROOT", "/kaggle/input")
OUT = os.environ.get("EMB_OUT", "/kaggle/working")
GROUP = os.environ.get("EMB_GROUP", "0")
ARCH = os.environ.get("EMB_ARCH", "vit_large_patch14_dinov2.lvd142m")
RES = int(os.environ.get("EMB_RES", "336"))           # div by patch 14 -> 24x24 tokens
BS = int(os.environ.get("EMB_BS", "48"))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class FullCache:
    """Recursive reader: finds slices_shard*.npy anywhere under root; each shard's
    index/meta/ids sit in the same directory as its slices file."""
    def __init__(self, root):
        self.slices, self.index, self.meta, self.ids = {}, {}, {}, {}
        self.uid2loc = {}
        found = glob.glob(os.path.join(root, "**", "slices_shard*.npy"), recursive=True)
        if not found:
            raise FileNotFoundError(f"no slices_shard*.npy under {root}")
        pairs = sorted((int(os.path.basename(p).split("shard")[1].split(".")[0]), os.path.dirname(p))
                       for p in found)
        for s, d in pairs:
            self.slices[s] = np.load(os.path.join(d, f"slices_shard{s}.npy"), mmap_mode="r")
            self.index[s] = np.load(os.path.join(d, f"index_shard{s}.npy"))
            self.meta[s] = np.load(os.path.join(d, f"meta_shard{s}.npy"), mmap_mode="r")
            self.ids[s] = np.load(os.path.join(d, f"ids_shard{s}.npy"))
            for row, uid in enumerate(self.ids[s]):
                self.uid2loc[str(uid)] = (s, row)
        self.shards = [s for s, _ in pairs]

    def uids(self):
        return [str(u) for s in self.shards for u in self.ids[s]]      # shard-contiguous order

    def count(self, uid):
        s, row = self.uid2loc[str(uid)]; return int(self.index[s][row, 1])

    def study(self, uid):
        s, row = self.uid2loc[str(uid)]
        off, cnt = (int(x) for x in self.index[s][row])
        return np.asarray(self.slices[s][off:off + cnt]), np.asarray(self.meta[s][off:off + cnt])


class Pooled(torch.nn.Module):
    """Per-slice feature = CLS (+) mean of patch tokens. DINOv2's CLS alone is a weak
    downstream feature; the mean patch token carries the dense signal, so we concat both."""
    def __init__(self, enc, n_prefix):
        super().__init__(); self.enc = enc; self.n = n_prefix

    def forward(self, x):
        t = self.enc.forward_features(x)             # (B, n_prefix+N, C), post-norm
        return torch.cat([t[:, 0], t[:, self.n:].mean(1)], -1)   # (B, 2C)


def triplets(sl, meta):
    """(K,H,W) uint8 + meta -> (K,3,H,W) float32 in [0,1]; neighbours stay inside a series."""
    K = sl.shape[0]
    series = meta[:, 2].astype(int)
    prev = np.arange(K) - 1; nxt = np.arange(K) + 1
    prev[0] = 0; nxt[-1] = K - 1
    brk = series[1:] != series[:-1]
    prev[1:][brk] = np.arange(K)[1:][brk]
    nxt[:-1][brk] = np.arange(K)[:-1][brk]
    return np.stack([sl[prev], sl, sl[nxt]], 1).astype(np.float32) / 255.0


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc = FullCache(ROOT)
    uids = fc.uids()
    lim = int(os.environ.get("EMB_LIMIT", "0"))
    if lim:
        uids = uids[:lim]                                  # smoke cap
    # DINOv2 checkpoints carry a `norm`+CLS head; forcing global_pool="avg" makes timm
    # expect `fc_norm` and the load fails. Load native, then pool CLS (+) mean-patch ourselves.
    base = timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=RES).eval()
    nprefix = int(getattr(base, "num_prefix_tokens", 1))
    feat = base.num_features * 2                           # CLS (+) mean-patch
    enc = Pooled(base, nprefix).eval().to(dev)
    for p in enc.parameters(): p.requires_grad_(False)
    ngpu = max(1, torch.cuda.device_count())
    bs = BS if os.environ.get("EMB_BS") else 48 * ngpu     # default scales with GPU count
    if ngpu > 1:                                           # Kaggle gives T4 x2 -> use both
        enc = torch.nn.DataParallel(enc)
    total = sum(fc.count(u) for u in uids)
    log(f"device {dev} x{ngpu} bs {bs} | shards {fc.shards} | studies {len(uids)} slices {total} | "
        f"enc {ARCH} feat {feat} RES {RES}")

    emb = np.lib.format.open_memmap(os.path.join(OUT, f"full_emb_g{GROUP}.npy"),
                                    mode="w+", dtype=np.float16, shape=(total, feat))
    meta_all = np.zeros((total, 4), np.float32)
    index = np.zeros((len(uids), 2), np.int64)
    mean, std = MEAN.to(dev), STD.to(dev)
    off = 0; last = time.time()
    for si, u in enumerate(uids):
        sl, mt = fc.study(u)
        x = torch.from_numpy(triplets(sl, mt))
        if x.shape[-1] != RES:
            x = torch.nn.functional.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
        feats = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            for b in range(0, x.shape[0], bs):
                feats.append(enc(((x[b:b + bs].to(dev) - mean) / std)).float().cpu())
        f = torch.cat(feats).numpy().astype(np.float16); k = f.shape[0]
        emb[off:off + k] = f; meta_all[off:off + k] = mt
        index[si] = (off, k); off += k
        if time.time() - last > 30 or si == len(uids) - 1:
            el = time.time() - t0
            log(f"  {si+1}/{len(uids)} studies | {off}/{total} slices | {el/60:.1f} min | "
                f"eta {el/(si+1)*(len(uids)-si-1)/60:.1f} min")
            last = time.time()
    emb.flush(); del emb
    np.save(os.path.join(OUT, f"full_meta_g{GROUP}.npy"), meta_all)
    np.save(os.path.join(OUT, f"full_idx_g{GROUP}.npy"), index)
    np.save(os.path.join(OUT, f"full_ids_g{GROUP}.npy"), np.asarray(uids, dtype="U"))
    log(f"DONE group {GROUP}: {total} slices -> {total*feat*2/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min")
    print("EMBED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
