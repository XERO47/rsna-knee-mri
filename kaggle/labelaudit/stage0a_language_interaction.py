"""Stage 0A (RE-RUN with fastText language IDs): is the model's per-finding accuracy conditional on the report's LANGUAGE?

THE HYPOTHESIS. Every member was trained on C, which is LLM-extracted from reports. If extraction
quality varies by language -- different hedging conventions, different abbreviations, different
section structure -- then C carries language-specific error patterns, every member inherits them,
and OOF-vs-C rewards reproducing them. A language x finding interaction in OOF-vs-C accuracy is
the observable signature.

WHAT IS AND IS NOT EVIDENCE HERE. A raw difference "AUC is lower for Turkish on Fracture" proves
nothing: languages differ in case mix, prevalence and acquisition quality, and some findings are
simply harder. The quantity tested removes both main effects:

    D(L,f) = [AUC(L,f) - AUC(all,f)] - mean over f'!=f of [AUC(L,f') - AUC(all,f')]

i.e. how much finding f is unusually good or bad *for this language relative to how that language
does on everything else*. Only the interaction survives.

CONTROLS
  * Bonferroni across every tested cell, since this is a fishing expedition by construction.
  * Slot control on studies with all five acquisition slots present (from geom64's per-slot
    `filled`). If a flag is really about acquisition completeness correlating with site and
    therefore language, it should weaken or flip here.
  * Prevalence is examined separately with a log-linear main-effects model. Descriptive only: a
    prevalence anomaly is not an accuracy anomaly, and conflating them is how this analysis would
    fool itself.

THE MODEL. The strongest genuinely cross-fitted predictor covering all 4,349 studies: arm14c's
five folds (head) rank-averaged with arm28's retrieval, per fold, pooled. Not the fold-0
10-member blend, which exists for 870 studies only and would leave most languages under the
threshold.

LANGUAGE ASSIGNMENT. The probes are verbatim from `references/run_dataset_eda.py`. They overlap by
construction and are NOT language identification, so a study is assigned only when it matches
EXACTLY ONE probe; multi-match and no-match studies are counted and excluded. Stage 0B taught the
relevant lesson here: a hand-built vocabulary can measure itself rather than the corpus, so the
exclusions are reported rather than papered over.
"""
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
SLOTS = [("sag_fs", 0, 18), ("sag_nonfs", 18, 32), ("cor_fs", 32, 44),
         ("cor_nonfs", 44, 52), ("axial", 52, 64)]
MIN_STUDIES, MIN_CLASS, N_BOOT, SEED = 100, 10, 2000, 20260927


def pct_rank(v):
    return (np.argsort(np.argsort(v, kind="stable"), kind="stable") + 1.0) / len(v)


def auc(y, p):
    y = (y > 0.5).astype(int)
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def load_model():
    """Cross-fitted OOF over all 4,349: per fold, rank-mean(head, retrieval); then pool."""
    d = np.load("kaggle/arm28/output/arm28_folds.npz", allow_pickle=True)
    ids, P, Y = [], [], []
    for f in range(5):
        h, r = d[f"fold{f}_head"], d[f"fold{f}_retr"]
        hr = np.stack([pct_rank(h[:, j]) for j in range(12)], 1)
        rr = np.stack([pct_rank(r[:, j]) for j in range(12)], 1)
        P.append((hr + rr) / 2.0)
        Y.append(d[f"fold{f}_y"])
        ids += [str(u) for u in d[f"fold{f}_ids"]]
    return ids, np.concatenate(P), np.concatenate(Y)


def load_language(ids):
    """fastText lid.176 assignments from A1, replacing the regex probes.

    The probes left 40.1% of the corpus unassigned and, worse, missed four whole families
    (Croatian/Serbian, Greek, Bulgarian, Dutch = ~25% of studies). Every language-conditioned
    number in the first run of this script was therefore computed on a corpus that silently
    omitted a quarter of itself. `multi` is returned as 0 because a classifier assigns one
    language; only low-confidence cases are unassigned."""
    t = pd.read_csv("kaggle/labelaudit/output/a1_language.csv")
    t["StudyInstanceUID"] = t.StudyInstanceUID.astype(str)
    t["language"] = t.language.fillna("")
    m = dict(zip(t.StudyInstanceUID, t.language))
    out = {sid: (m.get(sid) or None) for sid in ids}
    return out, 0, sum(1 for v in out.values() if v is None)


def load_slots(ids):
    """All five acquisition slots present, from geom64's per-slot geometry."""
    filled, sid_all = [], []
    for sh in (0, 1):
        g = np.load(f"kaggle/geom64_s{sh}/output/geom64_shard{sh}.npz", allow_pickle=True)
        filled.append(g["filled"].astype(bool))
        sid_all += [str(u) for u in g["ids"]]
    F = np.concatenate(filled)
    at = {u: i for i, u in enumerate(sid_all)}
    full = {}
    for sid in ids:
        if sid not in at:
            full[sid] = False
            continue
        row = F[at[sid]]
        full[sid] = all(row[a:b].any() for _, a, b in SLOTS)
    return full


def interaction(P, Y, mask):
    """D(L,f) for one language subset, against the all-study baseline."""
    base = np.array([auc(Y[:, j], P[:, j]) for j in range(12)])
    sub = np.array([auc(Y[mask][:, j], P[mask][:, j]) for j in range(12)])
    delta = sub - base
    D = np.full(12, np.nan)
    for j in range(12):
        others = np.delete(delta, j)
        if np.isfinite(delta[j]) and np.isfinite(others).sum() >= 6:
            D[j] = delta[j] - np.nanmean(others)
    return D, sub, base


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ids, P, Y = load_model()
    lang, multi, none = load_language(ids)
    full = load_slots(ids)
    L = np.array([lang[s] if lang[s] else "" for s in ids])
    full_arr = np.array([full[s] for s in ids])
    base_macro = np.nanmean([auc(Y[:, j], P[:, j]) for j in range(12)])
    print(f"cross-fitted OOF over {len(ids)} studies, macro AUC {base_macro:.4f} "
          f"(arm14c head + arm28 retrieval, rank-mean per fold)")
    print(f"language: {multi} studies match MULTIPLE probes and {none} match none -- both "
          f"excluded ({100 * (multi + none) / len(ids):.1f}% of the corpus)")
    counts = pd.Series(L[L != ""]).value_counts()
    print(f"\n{'language':<24} {'studies':>8} {'tested':>7}")
    langs = []
    for name, n in counts.items():
        ok = n >= MIN_STUDIES
        langs.append(name) if ok else None
        print(f"{name:<24} {n:>8} {'yes' if ok else 'no (<' + str(MIN_STUDIES) + ')':>7}")
    print(f"all five slots present: {full_arr.sum()}/{len(ids)} "
          f"({100 * full_arr.mean():.1f}%) -- the slot-control subset")

    # which cells are testable at all
    tested = []
    for Lname in langs:
        m = L == Lname
        for j in range(12):
            y = (Y[m][:, j] > 0.5).astype(int)
            if y.sum() >= MIN_CLASS and (len(y) - y.sum()) >= MIN_CLASS:
                tested.append((Lname, j))
            else:
                print(f"  not tested (thin): {Lname} x {LAB[j]} "
                      f"({int(y.sum())} pos / {len(y) - int(y.sum())} neg)")
    m_tests = len(tested)
    alpha = 0.05 / max(m_tests, 1)
    print(f"\n{m_tests} testable cells -> Bonferroni alpha {alpha:.5f} "
          f"({100 * (1 - alpha):.3f}% interval)")

    rng = np.random.default_rng(SEED)
    idx_by_lang = {Lname: np.where(L == Lname)[0] for Lname in langs}
    _inter = {Lname: interaction(P, Y, L == Lname) for Lname in langs}
    point = {Lname: v[0] for Lname, v in _inter.items()}
    # D removes whatever is uniformly hard about a language, which is what makes it a test of
    # an interaction rather than of case mix -- but it also hides the size of the effect a
    # reader cares about, so the raw deficit is reported beside it, descriptively.
    sub_auc = {Lname: v[1] for Lname, v in _inter.items()}
    base_auc = (next(iter(_inter.values()))[2] if _inter else np.full(12, np.nan))

    # bootstrap stratified by language: resample within each language, keep sizes fixed
    boots = {Lname: np.full((N_BOOT, 12), np.nan) for Lname in langs}
    other = np.where(~np.isin(L, langs))[0]
    for b in range(N_BOOT):
        take = [rng.choice(v, len(v), replace=True) for v in idx_by_lang.values()]
        if len(other):
            take.append(rng.choice(other, len(other), replace=True))
        sel = np.concatenate(take)
        Pb, Yb, Lb = P[sel], Y[sel], L[sel]
        bbase = np.array([auc(Yb[:, j], Pb[:, j]) for j in range(12)])
        for Lname in langs:
            mb = Lb == Lname
            sub = np.array([auc(Yb[mb][:, j], Pb[mb][:, j]) for j in range(12)])
            dl = sub - bbase
            for j in range(12):
                oth = np.delete(dl, j)
                if np.isfinite(dl[j]) and np.isfinite(oth).sum() >= 6:
                    boots[Lname][b, j] = dl[j] - np.nanmean(oth)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    print(f"\n{'language':<22} {'finding':<18} {'D':>8} {'Bonferroni interval':>24} "
          f"{'AUC(L)':>8} {'AUC(all)':>9} {'raw':>8} {'flag':>5}")
    flags, rows = [], {}
    for Lname, j in tested:
        col = boots[Lname][:, j]
        col = col[np.isfinite(col)]
        if len(col) < N_BOOT // 2:
            continue
        lo, hi = np.percentile(col, [lo_q, hi_q])
        d = point[Lname][j]
        flagged = (lo > 0) or (hi < 0)
        a_l, a_all = sub_auc[Lname][j], base_auc[j]
        raw = a_l - a_all
        rows[f"{Lname}|{LAB[j]}"] = {"D": float(d), "lo": float(lo), "hi": float(hi),
                                     "auc_language": float(a_l), "auc_all": float(a_all),
                                     "raw_deficit": float(raw), "flagged": bool(flagged)}
        if flagged:
            flags.append((Lname, j, d, lo, hi))
        mark = "  **" if flagged else ""
        print(f"{Lname:<22} {LAB[j]:<18} {d:>+8.4f} [{lo:>+7.4f},{hi:>+7.4f}] "
              f"{a_l:>8.4f} {a_all:>9.4f} {raw:>+8.4f}{mark}")

    print(f"\n{len(flags)} cell(s) flagged at the Bonferroni-adjusted level")
    surviving = []
    if flags:
        print(f"\nSLOT CONTROL (studies with all five slots): a flag stands only if the sign is "
              f"unchanged and |D| >= half the original")
        for Lname, j, d, lo, hi in flags:
            m = (L == Lname) & full_arr
            y = (Y[m][:, j] > 0.5).astype(int)
            if y.sum() < MIN_CLASS or (len(y) - y.sum()) < MIN_CLASS:
                print(f"  {Lname} x {LAB[j]}: control subset too thin "
                      f"({int(y.sum())}/{len(y) - int(y.sum())}) -- cannot confirm")
                continue
            Pc, Yc = P[full_arr], Y[full_arr]
            dc = interaction(Pc, Yc, (L == Lname)[full_arr])[0][j]
            keep = np.sign(dc) == np.sign(d) and abs(dc) >= abs(d) / 2
            print(f"  {Lname} x {LAB[j]}: D {d:+.4f} -> {dc:+.4f} on {int(m.sum())} studies "
                  f"-> {'STANDS' if keep else 'does not survive'}")
            rows[f"{Lname}|{LAB[j]}"]["D_slot_control"] = float(dc)
            rows[f"{Lname}|{LAB[j]}"]["survives_slot_control"] = bool(keep)
            if keep:
                surviving.append((Lname, LAB[j], d, dc,
                                  rows[f"{Lname}|{LAB[j]}"]["raw_deficit"]))

    # prevalence, descriptive only
    print(f"\nPREVALENCE (descriptive, NOT an accuracy claim): log-linear main-effects model, "
          f"adjusted residuals")
    obs = np.array([[(Y[L == Ln][:, j] > 0.5).sum() for j in range(12)] for Ln in langs],
                   float)
    n_l = obs.sum(1, keepdims=True)
    n_f = obs.sum(0, keepdims=True)
    exp = n_l @ n_f / obs.sum()
    resid = (obs - exp) / np.sqrt(exp * (1 - n_l / obs.sum()) * (1 - n_f / obs.sum()))
    big = []
    for a, Ln in enumerate(langs):
        for j in range(12):
            if abs(resid[a, j]) > 3.5:
                big.append((Ln, LAB[j], float(resid[a, j]), int(obs[a, j]), float(exp[a, j])))
    if big:
        for Ln, f, r, o, e in sorted(big, key=lambda x: -abs(x[2])):
            print(f"  {Ln:<22} {f:<18} residual {r:>+6.2f}  observed {o:>4} expected {e:>6.1f}")
    else:
        print("  no cell with |adjusted residual| > 3.5")

    json.dump({"n_studies": len(ids), "base_macro_auc": float(base_macro),
               "languages_tested": langs, "language_counts": {k: int(v) for k, v in counts.items()},
               "excluded_multi_probe": int(multi), "excluded_no_probe": int(none),
               "all_slots_present": int(full_arr.sum()),
               "n_tests": m_tests, "bonferroni_alpha": alpha,
               "cells": rows, "n_flagged": len(flags),
               "surviving_slot_control": [{"language": a, "finding": b, "D": c,
                                           "D_control": d, "raw_deficit": e}
                                          for a, b, c, d, e in surviving],
               "prevalence_outliers": [{"language": a, "finding": b, "adj_residual": c,
                                        "observed": o, "expected": e} for a, b, c, o, e in big],
               "model": "arm14c 5-fold head + arm28 retrieval, rank-mean per fold",
               "seed": SEED, "n_boot": N_BOOT},
              io.open(Path(out_dir) / "stage0a_summary.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {Path(out_dir) / 'stage0a_summary.json'}")
    print(f"\nGATE 0 trigger #1 (interaction surviving slot control): "
          f"{'MET -- ' + str(len(surviving)) + ' cell(s)' if surviving else 'NOT met'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle/labelaudit/output")
