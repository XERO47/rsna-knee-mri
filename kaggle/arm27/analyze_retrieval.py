"""arm27 a/b/c: is arm20's embedding locally organised by pathology, and does that
organisation disagree with arm20's own head enough to be worth anything?

This is a hypothesis, not an assumption. The head and the embedding geometry were shaped
by the same BCE objective on the same studies, so the honest prior is that a
nearest-neighbour vote AGREES with the head and adds nothing. The experiment is built so
that outcome is visible rather than hidden: every table reports the correlation between the
two, not just the blend.

    Pre-registered question -- Does arm20's representation contain locally useful pathology
    structure that is sufficiently decorrelated from its own classifier head to improve the
    blend?

THE BOUNDARY. Reference database = the 3,479 fold-0 TRAINING studies. Queries = the 870
held-out studies. A query's own label is used for exactly one thing: computing the AUC of a
score that was produced without it. No report, no report embedding and no query label ever
enters the database, so the identical procedure runs on the hidden test set.

THREE SCORES, all from cosine similarity between L2-normalised study embeddings:
    A  weighted kNN vote     softmax(sim/T) over the k nearest, dotted with their labels
    B  neighbour contrast    mean(sim | positive) - mean(sim | negative) among the k nearest
    C  prototype contrast    cos(z, mu+) - cos(z, mu-), centroids over the WHOLE reference

PRE-REGISTERED GRID AND DECISION, fixed before looking at any number:
    grid        k in {8, 16, 32, 64}, T in {0.03, 0.07, 0.15}
    PRIMARY     score A, k = 32, T = 0.07, blend weight 0.20   (the centre of the grid)
    adopt       macro blend delta >= +0.0015 on the primary
    serious     macro blend delta >= +0.0030
Everything else in the grid is exploratory and is labelled as such. Because the best of 17
configurations is an upward-biased estimate, the grid maximum is compared against a null
built by permuting the retrieval score across queries: that null says how large a
best-of-grid delta looks when the retrieval carries no information whatsoever.
"""
import io
import json
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

KS = (8, 16, 32, 64)
TS = (0.03, 0.07, 0.15)
WEIGHTS = (0.10, 0.20, 0.30)
PRIMARY = ("A", 32, 0.07, 0.20)
ADOPT, SERIOUS = 0.0015, 0.0030
N_NULL = 200
N_BOOT = 2000


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def pct_rank(v):
    """Percentile rank -- the space the submission blends in."""
    order = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return (order + 1.0) / len(v)


def score_A(sim, Yref, k, T):
    """Weighted kNN label vote. One neighbourhood per query, shared by all 12 findings."""
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(sim, idx, 1)
    w = np.exp((s - s.max(1, keepdims=True)) / T)
    w /= w.sum(1, keepdims=True)
    return np.einsum("qk,qkj->qj", w, Yref[idx])


def score_B(sim, Yref, k):
    """mean(sim | neighbour positive) - mean(sim | neighbour negative), per finding.

    If one side is absent from the k neighbours the difference is undefined; that side then
    falls back to the class's mean similarity over the WHOLE reference set, which is still
    query-dependent and still uses no query label."""
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(sim, idx, 1)
    Yn = Yref[idx]
    out = np.empty((sim.shape[0], Yref.shape[1]), np.float64)
    for j in range(Yref.shape[1]):
        p = Yn[:, :, j] > 0.5
        npos, nneg = p.sum(1), (~p).sum(1)
        pos = np.where(npos > 0, (s * p).sum(1) / np.maximum(npos, 1), np.nan)
        neg = np.where(nneg > 0, (s * ~p).sum(1) / np.maximum(nneg, 1), np.nan)
        gp = sim[:, Yref[:, j] > 0.5].mean(1)
        gn = sim[:, Yref[:, j] <= 0.5].mean(1)
        out[:, j] = np.where(np.isnan(pos), gp, pos) - np.where(np.isnan(neg), gn, neg)
    return out


def score_C(Zq, Zref, Yref):
    """cos(z, mu+) - cos(z, mu-), centroids over the entire reference set."""
    out = np.empty((len(Zq), Yref.shape[1]), np.float64)
    for j in range(Yref.shape[1]):
        p = Yref[:, j] > 0.5
        out[:, j] = Zq @ unit(Zref[p].mean(0)) - Zq @ unit(Zref[~p].mean(0))
    return out


def aucs(Y, P):
    """Per-finding AUC, NaN where the queries are single-class (that finding is unscorable)."""
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        y = (Y[:, j] > 0.5).astype(int)
        if 0 < y.sum() < len(y):
            out[j] = roc_auc_score(y, P[:, j])
    return out


def blend_auc(Y, base_rank, retr, w):
    r = np.stack([pct_rank(retr[:, j]) for j in range(retr.shape[1])], 1)
    return aucs(Y, (1 - w) * base_rank + w * r)


def main(emb_path, base_path, out_path):
    with np.load(emb_path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    LAB = [str(x) for x in d["lab"]]
    ref = d["is_reference"] == 1
    Zref, Zq = unit(d["z"][ref].astype(np.float64)), unit(d["z"][~ref].astype(np.float64))
    Yref, Yq = d["y"][ref], d["y"][~ref]
    ids_q = d["ids"][~ref]

    # the model to beat: arm20's own fold-0 OOF predictions, aligned by study id
    with np.load(base_path, allow_pickle=True) as f:
        b = {k: f[k] for k in f.files}
    pos = {str(u): i for i, u in enumerate(b["oof_ids"])}
    rows = [pos[str(u)] for u in ids_q]
    base = b["oof_pred"][rows]
    assert np.abs(b["oof_truth"][rows] - Yq).max() < 1e-4, "query labels disagree with comparator"
    base_rank = np.stack([pct_rank(base[:, j]) for j in range(base.shape[1])], 1)

    a_base = aucs(Yq, base)
    live = ~np.isnan(a_base)
    macro_base = float(np.nanmean(a_base))
    print(f"queries {len(Yq)} | reference {len(Yref)} | dim {Zq.shape[1]} | "
          f"scorable findings {int(live.sum())}/12")
    print(f"BASELINE arm20 f0_noaug macro {macro_base:.4f}\n")

    sim = Zq @ Zref.T
    print(f"cosine to the reference set: mean {sim.mean():.3f} | nearest {sim.max(1).mean():.3f}"
          f" | spread {sim.std():.3f}\n")

    cfgs = {}
    for k in KS:
        for T in TS:
            cfgs[("A", k, T)] = score_A(sim, Yref, k, T)
        cfgs[("B", k, None)] = score_B(sim, Yref, k)
    cfgs[("C", None, None)] = score_C(Zq, Zref, Yref)

    table = []
    for key, s in cfgs.items():
        a_r = aucs(Yq, s)
        rho = np.array([spearmanr(base[:, j], s[:, j]).statistic for j in range(12)])
        bl = {w: blend_auc(Yq, base_rank, s, w) for w in WEIGHTS}
        table.append({"score": key[0], "k": key[1], "T": key[2],
                      "retrieval_macro": float(np.nanmean(a_r)),
                      "spearman_mean": float(np.nanmean(rho[live])),
                      "auc": a_r, "rho": rho,
                      "blend": {w: float(np.nanmean(v)) for w, v in bl.items()},
                      "blend_auc": bl})

    print("GRID  (macro over the scorable findings; the three right columns are the blend "
          "delta vs the baseline)")
    head = f"{'score':>5} {'k':>4} {'T':>6} {'retr':>7} {'rho':>6} "
    print(head + " ".join(f"{'+' + str(int(w * 100)) + '%':>9}" for w in WEIGHTS))
    for r in sorted(table, key=lambda x: (x["score"], x["k"] or 0, x["T"] or 0)):
        print(f"{r['score']:>5} {str(r['k'] or '-'):>4} {str(r['T'] or '-'):>6} "
              f"{r['retrieval_macro']:>7.4f} {r['spearman_mean']:>6.3f} " +
              " ".join(f"{r['blend'][w] - macro_base:>+9.4f}" for w in WEIGHTS))

    prim = next(r for r in table
                if (r["score"], r["k"], r["T"]) == PRIMARY[:3])
    pw = PRIMARY[3]
    delta = prim["blend"][pw] - macro_base

    print(f"\nPRE-REGISTERED PRIMARY: score {PRIMARY[0]}, k={PRIMARY[1]}, T={PRIMARY[2]}, "
          f"blend weight {pw:.0%}")
    print(f"{'finding':<20} {'model':>7} {'retr':>7} {'rho':>6} " +
          " ".join(f"{'+' + str(int(w * 100)) + '%':>8}" for w in WEIGHTS))
    for j, name in enumerate(LAB):
        if not live[j]:
            print(f"{name:<20} unscorable (single-class among the queries)")
            continue
        print(f"{name:<20} {a_base[j]:>7.4f} {prim['auc'][j]:>7.4f} {prim['rho'][j]:>6.3f} " +
              " ".join(f"{prim['blend_auc'][w][j] - a_base[j]:>+8.4f}" for w in WEIGHTS))
    print(f"{'MACRO':<20} {macro_base:>7.4f} {prim['retrieval_macro']:>7.4f} "
          f"{prim['spearman_mean']:>6.3f} " +
          " ".join(f"{prim['blend'][w] - macro_base:>+8.4f}" for w in WEIGHTS))

    # How much of the grid maximum is just selection? Permute the retrieval score across
    # queries so it carries no information, then take the best of the same configurations.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_NULL):
        best = -9.9
        for r in table:
            s = cfgs[(r["score"], r["k"], r["T"])][rng.permutation(len(Yq))]
            for w in WEIGHTS:
                best = max(best, float(np.nanmean(blend_auc(Yq, base_rank, s, w))) - macro_base)
        null.append(best)
    null = np.array(null)
    grid_best = max(r["blend"][w] - macro_base for r in table for w in WEIGHTS)

    # Is the primary delta larger than this 870-study sample can explain?
    boot, prim_s = [], cfgs[PRIMARY[:3]]
    for _ in range(N_BOOT):
        i = rng.integers(0, len(Yq), len(Yq))
        br = np.stack([pct_rank(base[i][:, j]) for j in range(12)], 1)
        boot.append(np.nanmean(blend_auc(Yq[i], br, prim_s[i], pw))
                    - np.nanmean(aucs(Yq[i], base[i])))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\nprimary delta          {delta:+.4f}  (bootstrap 95% CI {lo:+.4f} .. {hi:+.4f}, "
          f"P(delta > 0) = {(boot > 0).mean():.2f})")
    print(f"best of the whole grid {grid_best:+.4f}  (exploratory: the max over "
          f"{len(table)} configs x {len(WEIGHTS)} weights)")
    print(f"  the same maximum, taken over scores permuted across queries so they carry no "
          f"information: {null.mean():+.4f} mean, {np.percentile(null, 95):+.4f} at the 95th "
          f"percentile. It is NEGATIVE because blending a useless score into a good baseline "
          f"costs AUC -- that is the bar grid-search luck alone can clear.")

    corr = prim["spearman_mean"]
    if delta >= SERIOUS:
        verdict = ("ADOPT AND OPTIMISE -- the embedding carries pathology structure its own "
                   "head is not using; worth real GPU time")
    elif delta >= ADOPT:
        verdict = "ADOPT -- above the pre-registered +0.0015, worth folding into the blend"
    elif corr > 0.95:
        verdict = ("INTERESTING FAILURE -- retrieval reproduces the head (rho > 0.95) instead "
                   "of disagreeing with it, so there is nothing left to blend: the linear "
                   "head already extracts what the neighbourhood knows")
    elif prim["retrieval_macro"] < 0.65:
        verdict = ("USEFUL DIAGNOSTIC FAILURE -- neighbours barely predict pathology "
                   "(retrieval macro < 0.65), so the space is organised by something else "
                   "(protocol, scanner, anatomy). Metric learning, not retrieval on this "
                   "embedding, would be the next lever")
    else:
        verdict = ("REJECT -- retrieval works but adds nothing beyond the head; below the "
                   "pre-registered +0.0015")
    print(f"\nVERDICT: {verdict}")

    json.dump({"baseline_macro": macro_base,
               "n_query": int(len(Yq)), "n_reference": int(len(Yref)),
               "primary": {"score": PRIMARY[0], "k": PRIMARY[1], "T": PRIMARY[2], "weight": pw,
                           "retrieval_macro": prim["retrieval_macro"],
                           "spearman_mean": prim["spearman_mean"], "delta": float(delta),
                           "ci": [float(lo), float(hi)],
                           "p_positive": float((boot > 0).mean()),
                           "per_finding": {LAB[j]: {"model": None if not live[j] else a_base[j],
                                                    "retrieval": float(prim["auc"][j])
                                                    if live[j] else None,
                                                    "spearman": float(prim["rho"][j]),
                                                    "blend": {str(w): float(
                                                        prim["blend_auc"][w][j] - a_base[j])
                                                        for w in WEIGHTS} if live[j] else None}
                                           for j in range(12)}},
               "grid": [{"score": r["score"], "k": r["k"], "T": r["T"],
                         "retrieval_macro": r["retrieval_macro"],
                         "spearman_mean": r["spearman_mean"],
                         "blend_delta": {str(w): r["blend"][w] - macro_base for w in WEIGHTS}}
                        for r in table],
               "grid_best_delta": float(grid_best),
               "null_best_of_grid_mean": float(null.mean()),
               "null_best_of_grid_p95": float(np.percentile(null, 95)),
               "thresholds": {"adopt": ADOPT, "serious": SERIOUS},
               "verdict": verdict},
              io.open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or [
        "kaggle/arm27/output/arm27_embeddings.npz",
        "kaggle/arm20/output/arm20_f0_noaug_preds.npz",
        "kaggle/arm27/output/arm27_retrieval.json"]))
