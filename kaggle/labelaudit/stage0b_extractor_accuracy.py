"""Stage 0B: how accurate is C, the report-extracted training label set, against the only
host-annotated labels we have?

WHY THIS IS VALID WHATEVER STEP 0 RESOLVES TO. The competition provides labels for 58 of 4,407
training studies and report text for the rest, so C is entirely third-party LLM extraction
(`stevenleehans/llm_labels_v4_blend`, chosen because it scored 0.8927 macro-AUC against those
58). This script measures C against those 58 host labels. That comparison is about EXTRACTOR
accuracy and does not depend on how the hidden test was labelled -- which is still unresolved
(see the Step 0 note in COORDINATION.md).

A specific external claim to test: a participant's repository states the host ground truth is
image-derived (two MSK radiologists plus an adjudicator, severity-thresholded, borderline cases
graded NEGATIVE) and that report-derived labels agree only about 82%. If that is right we should
see overall agreement near 82% here, and -- because borderline was graded negative -- C's errors
should skew toward FALSE POSITIVES: the report hedges, the extractor says 1, the radiologist
graded 0.

    C(s,f)      the report-extracted label used to train every member
    gold(s,f)   the host's label for the 58, binary

WHAT IT REPORTS
  * per finding: confusion matrix, sensitivity and specificity with Wilson 95% intervals
  * the FP/FN skew, which is the testable signature of a "borderline graded negative" rule
  * every disagreement with study id, language probe, finding, and the report excerpt around
    the finding's own vocabulary, so the direction of each error can be judged by reading

Language uses the EXACT probes from `references/run_dataset_eda.py`, not a new method, so these
counts are comparable with the documented corpus figures. The probes overlap by construction and
are not language identification -- a report can match several, and is reported as such.

58 studies is a small instrument. Wilson intervals are wide and are printed for that reason;
nothing here is a decision on its own, it is input to GATE 0.
"""
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# verbatim from references/run_dataset_eda.py
LANGUAGE_PROBES = {
    "English": r"\btear\b|\bmeniscus\b|\beffusion\b",
    "Spanish_or_Portuguese": r"\brotura\b|\bmenisco\b|\bderrame\b",
    "Italian": r"\blesione\b|\bmenisco\b|\bversamento\b",
    "French": r"\bd.chirure\b|\bm.nisque\b|\b.panchement\b",
    "German": r"\briss\b|\bmeniskus\b|\berguss\b",
    "Turkish": r"\byirtik\b|\bmenisk.s\b|\befuzyon\b",
}

# Vocabulary per finding, used ONLY to locate a report excerpt for human reading. It never
# decides a label. Multilingual because a third of the corpus is not English.
FINDING_TERMS = {
    "ACL": r"acl|anterior cruciate|cruzado anterior|crociato anteriore|crois. ant.rieur|"
           r"vorderen kreuzband|.n .apraz",
    "MCL": r"mcl|medial collateral|colateral medial|collaterale mediale|collat.ral interne|"
           r"innenband|i. yan ba",
    "Medial Meniscus": r"medial meniscus|menisco medial|menisco mediale|m.nisque interne|"
           r"innenmeniskus|i. men",
    "Lateral Meniscus": r"lateral meniscus|menisco lateral|menisco laterale|m.nisque externe|"
           r"au.enmeniskus|d.. men",
    "Medial OA": r"medial compartment|medial osteoarthr|artrosis medial|artrosi mediale|"
           r"gonarthrose interne|medial degenerat",
    "Lateral OA": r"lateral compartment|lateral osteoarthr|artrosis lateral|artrosi laterale|"
           r"gonarthrose externe|lateral degenerat",
    "PF OA": r"patellofemoral|patelofemoral|patellof.moral|retropatell|femoropatel",
    "Effusion": r"effusion|derrame|versamento|.panchement|erguss|efuzyon|s.v. artrit",
    "Synovitis": r"synovit|sinovit|synovial thick|synoviale",
    "Baker's": r"baker|popliteal cyst|quiste popl|cisti popl|kyste popl|poplitealzyste|"
           r"popliteal kist",
    "Contusion": r"contusion|contusi.n|contusione|bone bruise|kontuzyon|marrow o?edema|"
           r".dema .seo|knochenmark",
    "Fracture": r"fractur|fractura|frattura|fraktur|k.r.k|avulsion",
}


def wilson(k, n, z=1.959964):
    """Wilson score interval -- correct at the small n and extreme rates 58 studies produce,
    where a normal-approximation interval would run outside [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def excerpt(text, finding, width=170):
    """A window of report text around the finding's own vocabulary, for human reading only."""
    t = re.sub(r"\s+", " ", str(text or ""))
    m = re.search(FINDING_TERMS[finding], t, re.I)
    if m is None:
        return f"[no vocabulary match] {t[:width]}"
    a = max(0, m.start() - width // 2)
    return ("..." if a else "") + t[a:a + width] + ("..." if a + width < len(t) else "")


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv("data/raw/train.csv")
    raw["StudyInstanceUID"] = raw.StudyInstanceUID.astype(str)
    host = raw[raw[LAB].notna().all(axis=1)].copy()
    src = pd.read_csv("data/external/stevenleehans_rsna-knee-llm-report-labels/"
                      "llm_labels_v4_blend.csv")
    src["StudyInstanceUID"] = src.StudyInstanceUID.astype(str)
    C = host[["StudyInstanceUID"]].merge(src, on="StudyInstanceUID", how="left")
    miss = C[LAB].isna().any(axis=1).sum()
    print(f"host-labelled studies {len(host)} | C resolved for {len(C) - miss}/{len(C)}")
    if miss:
        raise RuntimeError(f"{miss} host studies have no C label; cannot audit them")

    G = host.set_index("StudyInstanceUID")[LAB]
    Cb = C.set_index("StudyInstanceUID")[LAB]
    Cb = Cb.loc[G.index]
    gold = (G.to_numpy() > 0.5).astype(int)
    pred = (Cb.to_numpy() > 0.5).astype(int)

    lang = {}
    for sid, rep in zip(host.StudyInstanceUID, host.Report):
        low = re.sub(r"\s+", " ", str(rep or "")).lower()
        hits = [n for n, pat in LANGUAGE_PROBES.items() if re.search(pat, low)]
        lang[sid] = "+".join(hits) if hits else "none"

    print(f"\nlanguage probes over the 58 (overlapping by construction, not identification):")
    for n in sorted(set(lang.values())):
        print(f"  {n:<32} {sum(1 for v in lang.values() if v == n):>3}")

    print(f"\n{'finding':<18} {'prev':>5} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} "
          f"{'sens (95% Wilson)':>24} {'spec (95% Wilson)':>24} {'acc':>6}")
    rows, tot_fp, tot_fn, tot_err = {}, 0, 0, 0
    for j, f in enumerate(LAB):
        g, p = gold[:, j], pred[:, j]
        tp = int(((g == 1) & (p == 1)).sum()); fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum()); tn = int(((g == 0) & (p == 0)).sum())
        se, sel, seh = wilson(tp, tp + fn)
        sp, spl, sph = wilson(tn, tn + fp)
        acc = (tp + tn) / len(g)
        tot_fp += fp; tot_fn += fn; tot_err += fp + fn
        rows[f] = {"prevalence_gold": int(g.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                   "sens": se, "sens_lo": sel, "sens_hi": seh,
                   "spec": sp, "spec_lo": spl, "spec_hi": sph, "acc": acc}
        thin = " <- thin" if (g.sum() < 10 or (1 - g).sum() < 10) else ""
        print(f"{f:<18} {int(g.sum()):>5} {tp:>3} {fp:>3} {fn:>3} {tn:>3} "
              f"{se:>8.3f} [{sel:.3f},{seh:.3f}] {sp:>8.3f} [{spl:.3f},{sph:.3f}] "
              f"{acc:>6.3f}{thin}")
    n_cells = gold.size
    agree = 1 - tot_err / n_cells
    print(f"\ncell-level agreement C vs host: {agree:.4f} ({n_cells - tot_err}/{n_cells} cells)")
    print(f"  false positives {tot_fp} | false negatives {tot_fn} | "
          f"FP share of errors {tot_fp / max(tot_err, 1):.3f}")
    print(f"  the external claim to test was ~82% agreement with a 'borderline graded negative'"
          f" rule,\n  which predicts agreement near 0.82 AND an FP-skewed error profile")

    # every disagreement, for reading
    dis = []
    rep_of = dict(zip(host.StudyInstanceUID, host.Report))
    for i, sid in enumerate(G.index):
        for j, f in enumerate(LAB):
            if gold[i, j] != pred[i, j]:
                dis.append({"StudyInstanceUID": sid, "language_probe": lang[sid], "finding": f,
                            "host": int(gold[i, j]), "C": int(pred[i, j]),
                            "C_raw": float(Cb.to_numpy()[i, j]),
                            "direction": "FP (C says 1, host 0)" if pred[i, j] == 1
                                         else "FN (C says 0, host 1)",
                            "excerpt": excerpt(rep_of[sid], f)})
    dd = pd.DataFrame(dis)
    dd.to_csv(out / "stage0b_disagreements.csv", index=False)
    print(f"\n{len(dd)} disagreements written to stage0b_disagreements.csv")
    if len(dd):
        print("\nby finding and direction:")
        piv = dd.pivot_table(index="finding", columns="direction", values="StudyInstanceUID",
                             aggfunc="count", fill_value=0)
        print(piv.to_string())
        print("\nby language probe:")
        print(dd.language_probe.value_counts().to_string())
        print("\nfirst 6 disagreements, for reading:")
        for _, r in dd.head(6).iterrows():
            print(f"  [{r.finding}] {r.direction} | {r.language_probe} | C_raw {r.C_raw:.3f}")
            print(f"      {r.excerpt[:150]}")

    json.dump({"n_studies": int(len(G)), "n_cells": int(n_cells),
               "cell_agreement": float(agree), "false_positives": int(tot_fp),
               "false_negatives": int(tot_fn),
               "fp_share_of_errors": float(tot_fp / max(tot_err, 1)),
               "per_finding": rows,
               "language_probe_counts": {k: int(sum(1 for v in lang.values() if v == k))
                                          for k in set(lang.values())},
               "source": "stevenleehans/llm_labels_v4_blend.csv",
               "caveat": "58 studies; Wilson intervals are wide. Input to GATE 0, not a "
                         "decision. Step 0 (host label provenance) remains unresolved."},
              io.open(out / "stage0b_summary.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out / 'stage0b_summary.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle/labelaudit/output")
