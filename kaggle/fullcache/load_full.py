"""Reader for the sharded full-cache (build_full.py output).

Layout per shard S in {0..N_SHARDS-1}, all in one directory:
  slices_shard{S}.npy  (Ttot, IMG, IMG) uint8   -- every slice of every study in the shard
  index_shard{S}.npy   (n_studies, 2) int64     -- [offset, count] into slices_shard{S}
  meta_shard{S}.npy    (Ttot, 4) float32        -- [plane, fluid, series_idx, pos_norm]
  ids_shard{S}.npy     (n_studies,) str          -- StudyInstanceUID per row (train.csv order)

study(uid) -> (slices (K,IMG,IMG) uint8, meta (K,4) float32), K variable.
Slices are memmapped, so opening is cheap and RAM stays flat.
"""
import glob, os
import numpy as np

META_PLANE, META_FLUID, META_SERIES, META_POS = 0, 1, 2, 3


class FullCache:
    def __init__(self, root):
        """root may hold the shard files directly (pod) OR contain per-shard
        subdirectories (Kaggle mounts each kernel-source output under its own dir),
        so slices_shard*.npy is searched recursively and each shard's index/meta/ids
        are read from the SAME directory as its slices file."""
        self.root = root
        self.slices, self.index, self.meta, self.ids = {}, {}, {}, {}
        self.uid2loc = {}                                   # uid -> (shard, local_row)
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
        self.img = self.slices[self.shards[0]].shape[-1]

    def __contains__(self, uid): return str(uid) in self.uid2loc

    def uids(self): return list(self.uid2loc.keys())

    def count(self, uid):
        s, row = self.uid2loc[str(uid)]
        return int(self.index[s][row, 1])

    def study(self, uid):
        s, row = self.uid2loc[str(uid)]
        off, cnt = (int(x) for x in self.index[s][row])
        sl = np.asarray(self.slices[s][off:off + cnt])      # (K,IMG,IMG) uint8
        mt = np.asarray(self.meta[s][off:off + cnt])         # (K,4) float32
        return sl, mt


if __name__ == "__main__":                                   # quick self-test on the smoke output
    import sys
    fc = FullCache(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(f"shards {fc.shards} | studies {len(fc.uid2loc)} | img {fc.img}")
    u = fc.uids()[0]
    sl, mt = fc.study(u)
    print(f"study {u}: slices {sl.shape} meta {mt.shape} | "
          f"planes {sorted(set(mt[:, 0].astype(int)))} series {int(mt[:, 2].max()) + 1} "
          f"px[{sl.min()},{sl.max()}]")
