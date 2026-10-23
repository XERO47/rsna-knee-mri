"""arm28: did arm27's +0.0031 replicate, or was fold 0 of arm20 lucky?

arm27 measured the retrieval member once: arm20 f0_noaug, 870 held-out studies, +0.0031
blended with that head and +0.0023 inside a 7-member blend. Two things were unverified --
it was one fold and one training configuration. arm28 repeats the identical measurement on
arm14c's five folds (ConvNeXt at 10 epochs, not 20), covering all 4,349 studies, with k=32
and T=0.07 carried over from arm27 and deliberately NOT re-tuned.

What would falsify the result: a delta that is positive on fold 0 and scattered around zero
elsewhere. What would confirm it: the same sign and rough size on all five, and a pooled
estimate whose confidence interval excludes zero over four times as many studies.

The blend here is head + retrieval only, because folds 1-4 have no other members -- so the
comparable arm27 number is +0.0031 (its single-head, pooled-space figure), not the
7-member +0.0023.
"""
import io
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

W = 0.20            # arm27's pre-registered blend weight
N_BOOT = 2000


def pct_rank(v):
    return (np.argsort(np.argsort(v, kind="stable"), kind="stable") + 1.0) / len(v)


def aucs(Y, P):
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        y = (Y[:, j] > 0.5).astype(int)
        if 0 < y.sum() < len(y):
            out[j] = roc_auc_score(y, P[:, j])
    return out


def delta(Y, head, retr, w=W):
    hr = np.stack([pct_rank(head[:, j]) for j in range(head.shape[1])], 1)
    rr = np.stack([pct_rank(retr[:, j]) for j in range(retr.shape[1])], 1)
    return aucs(Y, (1 - w) * hr + w * rr) - aucs(Y, head)


def main(path, out_path):
    with np.load(path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    LAB = [str(x) for x in d["lab"]]
    rng = np.random.default_rng(0)
    rows, allY, allH, allR, per_fold_delta = [], [], [], [], []
    for fold in range(5):
        Y, head, retr = d[f"fold{fold}_y"], d[f"fold{fold}_head"], d[f"fold{fold}_retr"]
        dj = delta(Y, head, retr)
        rho = float(np.nanmean([spearmanr(head[:, j], retr[:, j]).statistic
                                for j in range(12)]))
        rows.append({"fold": fold, "n": len(Y), "head": float(np.nanmean(aucs(Y, head))),
                     "retrieval": float(np.nanmean(aucs(Y, retr))), "spearman": rho,
                     "delta": float(np.nanmean(dj)),
                     "improved": int(np.nansum(dj > 0))})
        per_fold_delta.append(dj)
        allY.append(Y); allH.append(head); allR.append(retr)

    print(f"per-fold, blend weight {W:.0%}, k=32 T=0.07 carried over from arm27 unchanged\n")
    print(f"{'fold':>4} {'n':>5} {'head':>7} {'retrieval':>10} {'rho':>6} {'delta':>9} "
          f"{'improved':>9}")
    for r in rows:
        print(f"{r['fold']:>4} {r['n']:>5} {r['head']:>7.4f} {r['retrieval']:>10.4f} "
              f"{r['spearman']:>6.3f} {r['delta']:>+9.4f} {r['improved']:>6}/12")
    ds = np.array([r["delta"] for r in rows])
    print(f"{'mean':>4} {sum(r['n'] for r in rows):>5} "
          f"{np.mean([r['head'] for r in rows]):>7.4f} "
          f"{np.mean([r['retrieval'] for r in rows]):>10.4f} "
          f"{np.mean([r['spearman'] for r in rows]):>6.3f} {ds.mean():>+9.4f}")
    print(f"\nfolds with a positive delta: {int((ds > 0).sum())}/5 | "
          f"spread {ds.min():+.4f} .. {ds.max():+.4f} | sd {ds.std(ddof=1):.4f}")

    # Pooled over all 4,349 studies: rank within each fold (a fold is its own population),
    # then score the concatenation.
    Y = np.concatenate(allY)
    hr = np.concatenate([np.stack([pct_rank(h[:, j]) for j in range(12)], 1) for h in allH])
    rr = np.concatenate([np.stack([pct_rank(r[:, j]) for j in range(12)], 1) for r in allR])
    a_h = aucs(Y, hr)
    a_b = aucs(Y, (1 - W) * hr + W * rr)
    pooled = float(np.nanmean(a_b) - np.nanmean(a_h))
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(Y), len(Y))
        boot.append(np.nanmean(aucs(Y[i], (1 - W) * hr[i] + W * rr[i]))
                    - np.nanmean(aucs(Y[i], hr[i])))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\npooled over all {len(Y)} studies: {np.nanmean(a_h):.4f} -> {np.nanmean(a_b):.4f} "
          f"= {pooled:+.4f}  (95% CI {lo:+.4f}..{hi:+.4f}, P(>0) {(boot > 0).mean():.2f})")

    print(f"\n{'finding':<18} {'head':>7} {'retr':>7} {'delta':>9}  per-fold deltas")
    pf = np.stack(per_fold_delta)
    a_r = aucs(Y, rr)
    for j, name in enumerate(LAB):
        print(f"{name:<18} {a_h[j]:>7.4f} {a_r[j]:>7.4f} {a_b[j] - a_h[j]:>+9.4f}  " +
              " ".join(f"{x:+.4f}" for x in pf[:, j]))
    print(f"{'MACRO':<18} {np.nanmean(a_h):>7.4f} {np.nanmean(a_r):>7.4f} {pooled:>+9.4f}")
    consistent = int((np.nansum(pf > 0, axis=0) == 5).sum())
    print(f"\nfindings that improved in ALL five folds: {consistent}/12")

    if (ds > 0).sum() == 5 and lo > 0:
        v = ("REPLICATED -- positive in every fold on a different training configuration, "
             "and the pooled interval excludes zero over all 4,349 studies. arm27 was not a "
             "fold-0 accident")
    elif (ds > 0).sum() >= 4 and pooled > 0.0015:
        v = ("MOSTLY REPLICATED -- the effect is real but fold-dependent; size, not sign, "
             "is what varies")
    else:
        v = ("NOT REPLICATED -- arm27's fold-0 result does not generalise; the retrieval "
             "member should come out of the submission")
    print(f"\nVERDICT: {v}")
    json.dump({"weight": W, "per_fold": rows, "pooled_delta": pooled,
               "pooled_ci": [float(lo), float(hi)],
               "pooled_p_positive": float((boot > 0).mean()),
               "folds_positive": int((ds > 0).sum()),
               "findings_positive_in_all_folds": consistent,
               "per_finding": {LAB[j]: {"head": float(a_h[j]), "retrieval": float(a_r[j]),
                                        "delta": float(a_b[j] - a_h[j]),
                                        "per_fold": [float(x) for x in pf[:, j]]}
                               for j in range(12)},
               "verdict": v}, io.open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["kaggle/arm28/output/arm28_folds.npz",
                            "kaggle/arm28/output/arm28_replication.json"]))
