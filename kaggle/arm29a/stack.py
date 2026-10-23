"""arm29c: is the value of retrieval a constant weight, or does it depend on the case?

arm29a produced two retrieval scores whose mean Spearman correlation is 0.615 -- genuinely
different neighbourhoods, not one geometry seen twice. That makes their DISAGREEMENT a
candidate signal: a case where both retrievals agree strongly is different from one where they
disagree, and a fixed 20% blend weight cannot express that.

The feature set is deliberately tiny and the model deliberately rigid, because 870 studies
with ~100-500 positives per finding will happily fit anything flexible:

    head       the existing blend's rank for this finding
    R20        arm20 pooled-space retrieval
    R21        arm21 crop-space retrieval
    R20 - R21  signed disagreement
    |R20-R21|  unsigned disagreement
    nn20       cosine to the nearest reference neighbour in arm20's space
    nn21       the same in arm21's space

Per finding: logistic regression with strong L2, on rank-transformed features, scored by
5-fold cross-validation WITHIN the 870 OOF studies. Nothing here is fitted on data used to
judge it, and no feature comes from a query study's own label.

HONEST STATEMENT OF WHAT THIS CAN SHOW. A cross-validated gain over the fixed-weight blend on
870 studies is evidence that case-dependent weighting helps; it is NOT a deployable number,
because the stack's coefficients would themselves need fitting on data the hidden test does
not share. The pre-registered question is only: does the disagreement signal carry information
that a constant weight throws away? A gain under +0.002 here would say no, given that fitting
seven features per finding is itself worth something on a set this size.
"""
import io
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

C_GRID = (0.01, 0.1, 1.0)     # strong to moderate L2; no unregularised option
N_SPLITS = 5
SEED = 0


def pct_rank(v):
    ok = np.isfinite(v)
    out = np.full(len(v), 0.5)
    if ok.sum() > 1:
        r = np.argsort(np.argsort(v[ok], kind="stable"), kind="stable")
        out[ok] = (r + 1.0) / ok.sum()
    return out


def main(path, out_path):
    with np.load(path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    LAB = [str(x) for x in d["lab"]]
    Y, crop_cols = d["y"], [int(c) for c in d["crop_cols"]]
    feats = {"head": d["head_blend"], "R20": d["R20"], "R21": d["R21"],
             "diff": d["diff"], "absdiff": d["absdiff"],
             "nn20": d["nn_sim_R20"], "nn21": d["nn_sim_R21"]}
    names = list(feats)
    print(f"{len(Y)} OOF studies | {len(names)} features | logistic regression, "
          f"L2 C in {C_GRID}, {N_SPLITS}-fold CV inside the OOF set\n")
    print(f"{'finding':<18} {'fixed blend':>12} {'stacked (CV)':>13} {'delta':>9}  "
          f"{'best C':>7}  top coefficient")
    rows, deltas = {}, []
    for j in crop_cols:
        y = (Y[:, j] > 0.5).astype(int)
        X = np.stack([pct_rank(feats[n][:, j]) for n in names], 1)
        if not (0 < y.sum() < len(y)):
            continue
        base = roc_auc_score(y, feats["head"][:, j])
        best = (None, -1)
        for C in C_GRID:
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y):
                lr = LogisticRegression(C=C, max_iter=2000)
                lr.fit(X[tr], y[tr])
                oof[te] = lr.decision_function(X[te])
            a = roc_auc_score(y, oof)
            if a > best[1]:
                best = (C, a)
        C, a = best
        lr = LogisticRegression(C=C, max_iter=2000).fit(X, y)
        coef = dict(zip(names, lr.coef_[0]))
        top = max(coef, key=lambda k: abs(coef[k]))
        rows[LAB[j]] = {"fixed": float(base), "stacked": float(a), "delta": float(a - base),
                        "C": C, "coef": {k: float(v) for k, v in coef.items()}}
        deltas.append(a - base)
        print(f"{LAB[j]:<18} {base:>12.4f} {a:>13.4f} {a - base:>+9.4f}  {C:>7}  "
              f"{top} {coef[top]:+.2f}")
    md = float(np.mean(deltas))
    print(f"{'mean over ' + str(len(deltas)) + ' findings':<18} {'':>12} {'':>13} {md:>+9.4f}")

    # how much of that is just "fitting seven features on 870 studies"? Refit with the two
    # disagreement features replaced by noise: the same model capacity, no disagreement signal.
    rng = np.random.default_rng(SEED)
    null = []
    for j in crop_cols:
        y = (Y[:, j] > 0.5).astype(int)
        if not (0 < y.sum() < len(y)):
            continue
        X = np.stack([pct_rank(feats[n][:, j]) if n not in ("diff", "absdiff")
                      else rng.permutation(pct_rank(feats[n][:, j])) for n in names], 1)
        base = roc_auc_score(y, feats["head"][:, j])
        oof = np.zeros(len(y))
        C = rows[LAB[j]]["C"]
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y):
            lr = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
            oof[te] = lr.decision_function(X[te])
        null.append(roc_auc_score(y, oof) - base)
    mn = float(np.mean(null))
    print(f"\nsame model with the two disagreement features PERMUTED: {mn:+.4f} "
          f"-> disagreement itself is worth {md - mn:+.4f}")

    agree = {}
    for j in crop_cols:
        ok = np.isfinite(d["R20"][:, j]) & np.isfinite(d["R21"][:, j])
        ad = d["absdiff"][ok, j]
        lo, hi = ad <= np.median(ad), ad > np.median(ad)
        y = (Y[ok, j] > 0.5).astype(int)
        h = d["head_blend"][ok, j]
        a_lo = roc_auc_score(y[lo], h[lo]) if 0 < y[lo].sum() < lo.sum() else np.nan
        a_hi = roc_auc_score(y[hi], h[hi]) if 0 < y[hi].sum() < hi.sum() else np.nan
        agree[LAB[j]] = {"blend_auc_when_retrievals_agree": float(a_lo),
                         "blend_auc_when_they_disagree": float(a_hi)}
    print(f"\n{'finding':<18} {'blend AUC | agree':>18} {'blend AUC | disagree':>21}")
    for k, v in agree.items():
        print(f"{k:<18} {v['blend_auc_when_retrievals_agree']:>18.4f} "
              f"{v['blend_auc_when_they_disagree']:>21.4f}")
    print("If the blend is reliably worse on the cases where the two retrievals disagree, "
          "disagreement is a usable confidence signal even without a stacker.")

    v = ("DISAGREEMENT CARRIES SIGNAL -- worth building properly" if md - mn >= 0.002 else
         "NO CASE-DEPENDENT GAIN -- a constant retrieval weight is not leaving anything on "
         "the table that seven features per finding can pick up on 870 studies")
    print(f"\nVERDICT: {v}")
    json.dump({"features": names, "C_grid": list(C_GRID), "per_finding": rows,
               "mean_delta": md, "mean_delta_disagreement_permuted": mn,
               "attributable_to_disagreement": md - mn,
               "blend_auc_by_agreement": agree, "verdict": v},
              io.open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["kaggle/arm29a/output/arm29a_retrieval_meta_features.npz",
                            "kaggle/arm29a/output/arm29c_stack.json"]))
