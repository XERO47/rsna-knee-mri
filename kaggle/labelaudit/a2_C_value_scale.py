"""A2.1: recover C's own value convention, so Stage 2B can change extraction quality WITHOUT
also changing label softness.

WHY THIS MATTERS FOR 2B. If N's statuses are written onto a plain 0/1 scale while C is soft, then a
2B comparison confounds two changes: better extraction AND harder targets. The fix is to map N onto
C's existing value scale, which first requires knowing what that scale is and what each value means.

A1b gave the clue: 26.6% of all cells sit at 0.025, and Spanish/Portuguese has 38.2% at 0.250 where
English clusters at 0.025. That looks like a three-state convention -- confident positive, hedged or
unmentioned, confident negative -- but "looks like" is not a documented map, and the values may
differ per finding.

WHAT IT DOES
  1. tabulates every distinct value C takes, overall and per finding, with cell shares
  2. CALIBRATES each value against the 58 host-annotated studies: for cells at value v, what
     fraction are host-positive? That turns a guess about meaning into a measurement. A hedged
     value should sit between the confident ones; if 0.25 behaves like 0.025 it is a coverage gap,
     not hedging.
  3. reports the per-language share at each value, which is the A1b signal at value resolution
  4. writes the value -> meaning map that A2.4 will use as N's output scale

Gold-58 is used here ONLY to calibrate the extractor's own value convention, never for model
selection.
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SRC = "data/external/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(SRC)
    src["StudyInstanceUID"] = src.StudyInstanceUID.astype(str)
    raw = pd.read_csv("data/raw/train.csv")
    raw["StudyInstanceUID"] = raw.StudyInstanceUID.astype(str)
    lang = pd.read_csv(out / "a1_language.csv")
    lang["StudyInstanceUID"] = lang.StudyInstanceUID.astype(str)
    lang["language"] = lang.language.fillna("")

    V = src[LAB].to_numpy()
    vals = np.round(V.ravel(), 4)
    cnt = Counter(vals[~np.isnan(vals)])
    print(f"C over {len(src)} studies x 12 findings = {V.size} cells")
    print(f"distinct values: {len(cnt)}\n")
    print(f"{'value':>8} {'cells':>8} {'share':>7}")
    for v, c in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"{v:>8.4f} {c:>8} {c / V.size:>7.3f}")
    main_vals = [v for v, c in sorted(cnt.items(), key=lambda x: -x[1]) if c / V.size > 0.005]

    print(f"\nper finding -- is the convention the same everywhere?")
    print(f"{'finding':<18} {'n distinct':>10}  {'top 4 values (share)':<52}")
    per_find = {}
    for j, f in enumerate(LAB):
        col = np.round(V[:, j], 4)
        cc = Counter(col[~np.isnan(col)])
        top = sorted(cc.items(), key=lambda x: -x[1])[:4]
        per_find[f] = {"n_distinct": len(cc),
                       "top": [[float(v), int(n)] for v, n in top]}
        s = "  ".join(f"{v:.3f}({n / len(col):.2f})" for v, n in top)
        print(f"{f:<18} {len(cc):>10}  {s:<52}")

    # ---- calibration against the 58 host labels ---------------------------------
    host = raw[raw[LAB].notna().all(axis=1)][["StudyInstanceUID"] + LAB]
    H = host.set_index("StudyInstanceUID")[LAB]
    Cg = src.set_index("StudyInstanceUID").loc[H.index, LAB]
    hv, cv = (H.to_numpy() > 0.5).astype(int).ravel(), np.round(Cg.to_numpy(), 4).ravel()
    print(f"\nCALIBRATION on the {len(H)} host-annotated studies: what does each value MEAN?")
    print(f"{'C value':>8} {'cells':>6} {'host-positive rate':>26} {'reading'}")
    cal = {}
    for v in sorted(set(cv)):
        m = cv == v
        if m.sum() < 8:
            continue
        p, lo, hi = wilson(int(hv[m].sum()), int(m.sum()))
        cal[float(v)] = {"cells": int(m.sum()), "host_pos_rate": p, "lo": lo, "hi": hi}
        read = ("confident positive" if p > 0.7 else
                "intermediate" if p > 0.35 else "confident negative")
        print(f"{v:>8.4f} {int(m.sum()):>6} {p:>12.3f} [{lo:.3f},{hi:.3f}] {read}")
    mids = [v for v, d in cal.items() if 0.35 < d["host_pos_rate"] <= 0.7]
    print(f"\n  -> the scale is ordered: low values are host-negative, high values host-positive,"
          f"\n     and {len(mids)} intermediate value(s) sit genuinely in between "
          f"{mids if mids else '(none)'}")

    # ---- per-language share at each main value ----------------------------------
    df = src.merge(lang[["StudyInstanceUID", "language"]], on="StudyInstanceUID", how="left")
    df["language"] = df.language.fillna("")
    langs = [L for L, n in df.language.value_counts().items() if L and n >= 80]
    print(f"\nPER-LANGUAGE share of cells at each main value")
    print(f"{'language':<26} {'n':>5} " + " ".join(f"{v:>8.3f}" for v in main_vals))
    per_lang = {}
    for L in ["CORPUS"] + langs:
        m = np.ones(len(df), bool) if L == "CORPUS" else (df.language == L).to_numpy()
        col = np.round(df.loc[m, LAB].to_numpy(), 4).ravel()
        shares = [float((col == v).mean()) for v in main_vals]
        per_lang[L] = {"n": int(m.sum()),
                       "shares": {str(v): s for v, s in zip(main_vals, shares)}}
        print(f"{L:<26} {int(m.sum()):>5} " + " ".join(f"{s:>8.3f}" for s in shares))

    hedged = sorted(cal, key=lambda v: abs(cal[v]["host_pos_rate"] - 0.5))[0] if cal else None
    scale = {"positive": max(cal, key=lambda v: cal[v]["host_pos_rate"]) if cal else 0.975,
             "hedged": hedged,
             "negative": min(cal, key=lambda v: cal[v]["host_pos_rate"]) if cal else 0.025}
    print(f"\nVALUE SCALE for A2.4 (N's statuses will be written onto these numbers):")
    for k, v in scale.items():
        r = cal.get(v, {}).get("host_pos_rate", float('nan'))
        print(f"  {k:<10} -> {v:.4f}   (host-positive rate {r:.3f})")
    print(f"\n  positive        -> {scale['positive']:.3f}")
    print(f"  uncertain / below-severity-threshold -> {scale['hedged']:.3f}  "
          f"(ranked between, which is all AUC needs)")
    print(f"  explicit negative / not mentioned    -> {scale['negative']:.3f}")

    json.dump({"source": SRC, "n_cells": int(V.size), "n_distinct_values": len(cnt),
               "value_counts": {str(v): int(c) for v, c in cnt.items()},
               "main_values": [float(v) for v in main_vals],
               "per_finding": per_find, "calibration_on_host_58": cal,
               "per_language_value_shares": per_lang,
               "value_scale_for_2B": {k: float(v) for k, v in scale.items()},
               "note": "gold-58 used only to calibrate the extractor's value convention"},
              io.open(out / "a2_C_value_scale.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out / 'a2_C_value_scale.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle/labelaudit/output")
