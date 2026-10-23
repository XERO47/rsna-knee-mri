"""Stage 0C (RE-RUN with fastText language IDs): does retrieval work by transmitting LABEL STYLE rather than pathology?

THE SUSPICION, and why it is specific. arm27/arm28 measured a replicated +0.0038 OOF gain from a
kNN vote over per-finding pooled embeddings, and it was worth ~0.000 on the LB. One explanation
that fits both facts: the embedding encodes acquisition site; site correlates with language, which
correlates with reporting convention, which is what C was extracted from. A kNN vote would then
transmit a study's site-specific LABEL STYLE, which OOF-vs-C rewards and the host's image-derived
labels do not.

THREE MEASUREMENTS, each falsifiable:

  1. NEIGHBOUR HOMOGENEITY. For each query, the fraction of its 32 neighbours sharing its language
     divided by that language's base rate among the reference studies. 1.0 means the neighbourhood
     is language-indifferent; >1 means retrieval preferentially returns same-language studies.

  2. A LANGUAGE PROBE. 5-fold logistic regression from the per-finding pooled embedding to
     language, against the majority-class baseline. If the embedding cannot predict language, the
     mimicry route is closed regardless of anything else.

  3. INCREMENT BY HOMOGENEITY TERTILE. The retrieval blend increment computed separately for
     queries in the bottom, middle and top tertile of homogeneity. Mimicry predicts the increment
     RISES with homogeneity: where the neighbourhood is language-pure, more label style is
     available to copy.

DEVIATION FROM THE PLAN, recorded deliberately: the plan specifies arm28, but arm28's outputs kept
only the retrieval scores, not embeddings or neighbour identities, so neighbour composition cannot
be recovered from it. arm27's fold-0 embeddings are available and use the identical configuration
(k=32, T=0.07, per-finding pooled space, 3,479 reference / 870 query). This is therefore one fold
rather than five. Given arm28 replicated the gain across all five, a fold-0 diagnostic of the
mechanism is a reasonable substitute, but it is one fold and the tertile CIs will be wide.

Language assignment is the exclusive-single-probe rule from Stage 0A, with multi-match and no-match
studies excluded and counted.
"""
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
LANGUAGE_PROBES = {
    "English": r"\btear\b|\bmeniscus\b|\beffusion\b",
    "Spanish_or_Portuguese": r"\brotura\b|\bmenisco\b|\bderrame\b",
    "Italian": r"\blesione\b|\bmenisco\b|\bversamento\b",
    "French": r"\bd.chirure\b|\bm.nisque\b|\b.panchement\b",
    "German": r"\briss\b|\bmeniskus\b|\berguss\b",
    "Turkish": r"\byirtik\b|\bmenisk.s\b|\befuzyon\b",
}
K, T, W, N_BOOT, SEED = 32, 0.07, 0.20, 2000, 20260927


def pct_rank(v):
    return (np.argsort(np.argsort(v, kind="stable"), kind="stable") + 1.0) / len(v)


def aucs(Y, P):
    out = np.full(12, np.nan)
    for j in range(12):
        y = (Y[:, j] > 0.5).astype(int)
        if 0 < y.sum() < len(y):
            out[j] = roc_auc_score(y, P[:, j])
    return out


def language_of(ids):
    """fastText lid.176 assignments from A1 (see Stage 0A for why the probes were replaced)."""
    t = pd.read_csv("kaggle/labelaudit/output/a1_language.csv")
    t["StudyInstanceUID"] = t.StudyInstanceUID.astype(str)
    t["language"] = t.language.fillna("")
    m = dict(zip(t.StudyInstanceUID, t.language))
    return np.array([m.get(sid, "") or "" for sid in ids])


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    e = {k: v for k, v in np.load("kaggle/arm27/output/arm27_embeddings.npz",
                                  allow_pickle=True).items()}
    ids = np.array([str(u) for u in e["ids"]])
    isref = e["is_reference"] == 1
    pool, Yall = e["pool"], e["y"]
    lang = language_of(ids)
    qi, ri = np.where(~isref)[0], np.where(isref)[0]
    print(f"arm27 fold 0: {len(ri)} reference / {len(qi)} query studies | k={K} T={T}")
    known_q = lang[qi] != ""
    print(f"language assigned (exactly one probe): {known_q.sum()}/{len(qi)} queries, "
          f"{(lang[ri] != '').sum()}/{len(ri)} reference")

    # base rate of each language among the reference studies
    rl = lang[ri]
    base = {L: float((rl == L).mean()) for L in set(rl) - {""}}
    print(f"\nreference base rates: " +
          ", ".join(f"{L} {base[L]:.3f}" for L in sorted(base, key=lambda x: -base[x])))

    # ---- 1. neighbour homogeneity, in each finding's own pooled space ----------
    hom = np.full(len(qi), np.nan)
    per_find = np.full((len(qi), 12), np.nan)
    for j in range(12):
        zr = pool[ri, j].astype(np.float64)
        zr /= np.maximum(np.linalg.norm(zr, axis=1, keepdims=True), 1e-12)
        zq = pool[qi, j].astype(np.float64)
        zq /= np.maximum(np.linalg.norm(zq, axis=1, keepdims=True), 1e-12)
        sim = zq @ zr.T
        idx = np.argpartition(-sim, K - 1, axis=1)[:, :K]
        nl = rl[idx]                                    # (n_q, K) neighbour languages
        for a in range(len(qi)):
            L = lang[qi[a]]
            if L == "" or L not in base or base[L] <= 0:
                continue
            per_find[a, j] = (nl[a] == L).mean() / base[L]
    hom = np.nanmean(per_find, axis=1)
    ok = np.isfinite(hom)
    print(f"\nhomogeneity ratio (same-language share of 32 neighbours / that language's base "
          f"rate)\n  over {ok.sum()} queries with a language: mean {np.nanmean(hom):.3f}, "
          f"median {np.nanmedian(hom):.3f}, p10 {np.nanpercentile(hom,10):.3f}, "
          f"p90 {np.nanpercentile(hom,90):.3f}")
    print("  1.0 would mean retrieval is language-indifferent; >1 means it prefers "
          "same-language studies")
    for L in sorted(base, key=lambda x: -base[x]):
        m = (lang[qi] == L) & ok
        if m.sum() >= 20:
            print(f"    {L:<24} n={m.sum():>4}  mean ratio {np.nanmean(hom[m]):.3f}")

    # ---- 2. can the embedding predict language at all? -------------------------
    print(f"\nLANGUAGE PROBE: per-finding pooled embedding -> language, 5-fold logistic "
          f"regression")
    sel = np.where(lang != "")[0]
    ysel = lang[sel]
    keep = np.isin(ysel, [L for L in set(ysel) if (ysel == L).sum() >= 30])
    sel, ysel = sel[keep], ysel[keep]
    maj = pd.Series(ysel).value_counts(normalize=True).iloc[0]
    print(f"  {len(sel)} studies, {len(set(ysel))} classes, majority baseline {maj:.3f}")
    probe = {}
    for j in [LAB.index(x) for x in ("ACL", "Effusion", "Fracture")]:
        X = pool[sel, j].astype(np.float64)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        acc = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, ysel):
            lr = LogisticRegression(max_iter=3000, C=1.0)
            lr.fit(X[tr], ysel[tr])
            acc.append((lr.predict(X[te]) == ysel[te]).mean())
        probe[LAB[j]] = float(np.mean(acc))
        print(f"  {LAB[j]:<12} accuracy {np.mean(acc):.3f}  (baseline {maj:.3f}, "
              f"lift {np.mean(acc) - maj:+.3f})")

    # ---- 3. retrieval increment by homogeneity tertile -------------------------
    ref20 = np.load("kaggle/arm20/output/arm20_f0_noaug_preds.npz", allow_pickle=True)
    order = {u: i for i, u in enumerate(ids[qi])}
    rows = [order[str(u)] for u in ref20["oof_ids"]]
    head = ref20["oof_pred"]
    Yq = Yall[qi][rows]
    homq = hom[rows]
    R = np.full((len(rows), 12), np.nan)
    for j in range(12):
        zr = pool[ri, j].astype(np.float64)
        zr /= np.maximum(np.linalg.norm(zr, axis=1, keepdims=True), 1e-12)
        zq = pool[qi, j].astype(np.float64)[rows]
        zq /= np.maximum(np.linalg.norm(zq, axis=1, keepdims=True), 1e-12)
        sim = zq @ zr.T
        idx = np.argpartition(-sim, K - 1, axis=1)[:, :K]
        s = np.take_along_axis(sim, idx, 1)
        w = np.exp((s - s.max(1, keepdims=True)) / T)
        w /= w.sum(1, keepdims=True)
        R[:, j] = (w * (Yall[ri][:, j] > 0.5)[idx]).sum(1)

    def increment(mask):
        hr = np.stack([pct_rank(head[mask][:, j]) for j in range(12)], 1)
        rr = np.stack([pct_rank(R[mask][:, j]) for j in range(12)], 1)
        a0 = aucs(Yq[mask], hr)
        a1 = aucs(Yq[mask], (1 - W) * hr + W * rr)
        return float(np.nanmean(a1) - np.nanmean(a0))

    good = np.isfinite(homq)
    q1, q2 = np.nanpercentile(homq[good], [33.333, 66.667])
    tert = {"bottom": good & (homq <= q1), "middle": good & (homq > q1) & (homq <= q2),
            "top": good & (homq > q2)}
    print(f"\nRETRIEVAL INCREMENT BY HOMOGENEITY TERTILE (blend weight {W:.0%})")
    print(f"  tertile cuts at {q1:.3f} and {q2:.3f}")
    rng = np.random.default_rng(SEED)
    res = {}
    print(f"  {'tertile':<8} {'n':>5} {'mean ratio':>11} {'increment':>10} {'95% CI':>20}")
    for name, m in tert.items():
        n = int(m.sum())
        inc = increment(m)
        boot = []
        who = np.where(m)[0]
        for _ in range(N_BOOT):
            b = rng.choice(who, len(who), replace=True)
            mm = np.zeros(len(homq), bool);
            hr = np.stack([pct_rank(head[b][:, j]) for j in range(12)], 1)
            rr = np.stack([pct_rank(R[b][:, j]) for j in range(12)], 1)
            a0 = aucs(Yq[b], hr); a1 = aucs(Yq[b], (1 - W) * hr + W * rr)
            boot.append(np.nanmean(a1) - np.nanmean(a0))
        boot = np.array(boot); lo, hi = np.percentile(boot, [2.5, 97.5])
        res[name] = {"n": n, "mean_ratio": float(np.nanmean(homq[m])), "increment": inc,
                     "ci": [float(lo), float(hi)]}
        print(f"  {name:<8} {n:>5} {np.nanmean(homq[m]):>11.3f} {inc:>+10.4f} "
              f"[{lo:>+7.4f},{hi:>+7.4f}]")
    b, t = res["bottom"], res["top"]
    monotone = b["increment"] < res["middle"]["increment"] < t["increment"]
    disjoint = (b["ci"][1] < t["ci"][0]) or (t["ci"][1] < b["ci"][0])
    print(f"\n  monotone increasing: {monotone} | bottom/top CIs non-overlapping: {disjoint}")
    verdict = ("MIMICRY SUPPORTED" if monotone and disjoint else
               "mimicry NOT supported by the tertile test")
    print(f"  -> {verdict}")

    json.dump({"n_reference": int(len(ri)), "n_query": int(len(qi)), "k": K, "temperature": T,
               "reference_base_rates": base,
               "homogeneity_mean": float(np.nanmean(hom)),
               "homogeneity_median": float(np.nanmedian(hom)),
               "per_language_ratio": {L: float(np.nanmean(hom[(lang[qi] == L) & ok]))
                                      for L in base if ((lang[qi] == L) & ok).sum() >= 20},
               "language_probe_accuracy": probe, "majority_baseline": float(maj),
               "tertiles": res, "monotone": bool(monotone), "cis_disjoint": bool(disjoint),
               "verdict": verdict, "seed": SEED, "n_boot": N_BOOT,
               "deviation": "arm27 fold-0 embeddings used; arm28 saved no embeddings"},
              io.open(out / "stage0c_summary.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out / 'stage0c_summary.json'}")
    print(f"\nGATE 0 trigger #3: {'MET' if monotone and disjoint else 'NOT met'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle/labelaudit/output")
