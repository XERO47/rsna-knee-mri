"""Merge the 4 embed-group outputs (full_emb_g{G}.npy + idx/meta/ids) into the single
CSR embedding set train_full_tx.py reads: full_emb.npy / full_idx.npy / full_meta.npy /
full_ids.npy. Group offsets are local, so each group's index is shifted by the running
slice total as it is concatenated.

Usage: python assemble_emb.py <dir-with-group-files> <out-dir>
"""
import glob, os, sys
import numpy as np


def main(src, out):
    os.makedirs(out, exist_ok=True)
    groups = sorted(g.split("full_emb_")[1].split(".")[0]
                    for g in glob.glob(os.path.join(src, "full_emb_*.npy")))
    if not groups:
        raise FileNotFoundError(f"no full_emb_*.npy under {src}")
    print("groups:", groups)
    embs = [np.load(os.path.join(src, f"full_emb_{g}.npy"), mmap_mode="r") for g in groups]
    total = sum(e.shape[0] for e in embs); feat = embs[0].shape[1]
    nstud = sum(np.load(os.path.join(src, f"full_idx_{g}.npy")).shape[0] for g in groups)
    print(f"total slices {total} | feat {feat} | studies {nstud} | {total*feat*2/1e9:.2f} GB")

    emb = np.lib.format.open_memmap(os.path.join(out, "full_emb.npy"), mode="w+",
                                    dtype=np.float16, shape=(total, feat))
    meta = np.zeros((total, 4), np.float32)
    index = np.zeros((nstud, 2), np.int64)
    ids = []
    soff, roff = 0, 0                                         # slice offset, study-row offset
    for g, e in zip(groups, embs):
        gi = np.load(os.path.join(src, f"full_idx_{g}.npy"))
        gm = np.load(os.path.join(src, f"full_meta_{g}.npy"))
        gids = np.load(os.path.join(src, f"full_ids_{g}.npy"))
        k = e.shape[0]; ns = gi.shape[0]
        emb[soff:soff + k] = e; meta[soff:soff + k] = gm
        gi2 = gi.copy(); gi2[:, 0] += soff                   # shift local offsets into the merged array
        index[roff:roff + ns] = gi2; ids.extend(str(u) for u in gids)
        soff += k; roff += ns
        print(f"  {g}: +{ns} studies +{k} slices -> {soff}")
    emb.flush(); del emb
    np.save(os.path.join(out, "full_meta.npy"), meta)
    np.save(os.path.join(out, "full_idx.npy"), index)
    np.save(os.path.join(out, "full_ids.npy"), np.asarray(ids, dtype="U"))
    assert len(ids) == len(set(ids)) == nstud, f"duplicate/missing uids: {len(ids)} vs {nstud}"
    print(f"DONE -> {out} | {nstud} studies, {total} slices")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
