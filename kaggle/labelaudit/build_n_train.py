# -*- coding: utf-8 -*-
"""A2.4. Build N_train: C, with the consensus disagreements edited and nothing else touched.

THE POINT OF "MINIMAL EDIT". C is a soft, continuous key with 185 distinct values, and every member
of the 0.932 ensemble was fitted against it. Replacing it wholesale with a fresh binary reading would
change two things at once -- which cells are positive AND how confident every cell is -- and a
difference measured afterwards could not be attributed to either. So N_train is C, byte-identical,
except at cells where BOTH independent passes agree with each other and both disagree with C.

Everything about how a cell is edited is fixed in advance:

  * only STABLE disagreements are edited. A single pass disagreeing with C is as likely to be the new
    extractor's error as C's.
  * the new value is written on C'S OWN SCALE, recovered in A2 from C's value distribution and
    calibrated against the 58 host studies:
        explicit positive -> 0.985      (host-positive rate 0.80-0.92 at that value)
        hedged            -> 0.92       (0.44 -- genuinely ambiguous, and C already encodes it so)
        not addressed     -> 0.25       (0.070)
        explicit negative -> 0.01       (0.00-0.05)
    Writing a hard 0 or 1 instead would silently also change the confidence weighting that
    `2*|p-0.5|` gives the loss, which is a second change disguised as one.
  * `invalid` cells are never edited. Absence of a reading is not a reading.
  * no vision-model output enters any decision here. That is arm15, which bought +0.008 on gold and
    +0.001 on the leaderboard.

GATING. This runs only after GATE 1 passes. It refuses to write if the edit fraction exceeds
MAX_EDIT_FRACTION, because an "edit" that rewrites a fifth of the key is a replacement, and that is a
different experiment needing its own pre-registration.
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# C's own value convention (A2), so an edit is expressed in the units C already uses
C_SCALE = {"positive": 0.985, "uncertain": 0.92, "not_mentioned": 0.25, "negative": 0.01}
MAX_EDIT_FRACTION = 0.15


def load_pass(path):
    d = pd.read_csv(path)
    d["StudyInstanceUID"] = d.StudyInstanceUID.astype(str)
    for c in ("status", "acl_substatus"):
        d[c] = d[c].fillna("") if c in d else ""
    return d[["StudyInstanceUID", "finding", "status", "acl_substatus"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", required=True)
    ap.add_argument("--n2", required=True,
                    help="required: a single pass cannot distinguish C's errors from N's")
    ap.add_argument("--labels", default="data/external/stevenleehans_rsna-knee-llm-report-labels/"
                                       "llm_labels_v4_blend.csv")
    ap.add_argument("--language", default="kaggle/labelaudit/output/a1_language.csv")
    ap.add_argument("--out", default="kaggle/labelaudit/output")
    ap.add_argument("--force", action="store_true",
                    help="write even if the edit fraction exceeds the cap (say why in the log)")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extract"))
    try:
        from extract_labels import ACL_SUB_BINARISE as acl_bin
    except Exception:
        acl_bin = None
        print("NOTE: ACL_SUB_BINARISE not importable; ACL uses `status` alone")

    C = pd.read_csv(a.labels)
    C["StudyInstanceUID"] = C.StudyInstanceUID.astype(str)
    N1, N2 = load_pass(a.n1), load_pass(a.n2)
    lang = pd.read_csv(a.language)[["StudyInstanceUID", "language"]]
    lang["StudyInstanceUID"] = lang.StudyInstanceUID.astype(str)

    def to_binary(row):
        st = str(row["status"])
        if st == "invalid":
            return None
        if row["finding"] == "ACL" and acl_bin:
            v = acl_bin.get(str(row.get("acl_substatus", "") or ""))
            if v is not None:
                return int(v)
        if st == "positive":
            return 1
        if st in ("negative", "uncertain", "not_mentioned"):
            return 0
        return None

    m = N1.merge(N2, on=["StudyInstanceUID", "finding"], suffixes=("_1", "_2"))
    m["b1"] = [to_binary({"status": r.status_1, "finding": r.finding,
                          "acl_substatus": r.acl_substatus_1}) for r in m.itertuples()]
    m["b2"] = [to_binary({"status": r.status_2, "finding": r.finding,
                          "acl_substatus": r.acl_substatus_2}) for r in m.itertuples()]
    Cl = C.set_index("StudyInstanceUID")
    m = m[m.StudyInstanceUID.isin(Cl.index)]
    m["c_value"] = [Cl.at[r.StudyInstanceUID, r.finding] for r in m.itertuples()]
    m["c_bin"] = (m.c_value > 0.5).astype(int)
    m["stable"] = m.b1.notna() & m.b2.notna() & (m.b1 == m.b2)
    m["edit"] = m.stable & (m.b1 != m.c_bin)
    # the new value follows N1's STATUS, not merely its binary, so a hedge stays a hedge
    m["new_value"] = [C_SCALE.get(str(r.status_1), np.nan) if r.edit else r.c_value
                      for r in m.itertuples()]

    n_cells = len(m)
    n_edit = int(m.edit.sum())
    frac = n_edit / max(n_cells, 1)
    print(f"cells with both passes present : {n_cells}")
    print(f"stable across the two passes   : {m.stable.mean():.4f}")
    print(f"EDITS                          : {n_edit} ({frac:.4f} of cells)")
    print(f"  N adds a positive C missed   : {int((m.edit & (m.b1 == 1)).sum())}")
    print(f"  N removes a positive C had   : {int((m.edit & (m.b1 == 0)).sum())}")

    per_f = (m[m.edit].groupby("finding").size().reindex(LAB).fillna(0).astype(int))
    tot_f = m.groupby("finding").size().reindex(LAB).fillna(0).astype(int)
    print(f"\n  {'finding':<18}{'edits':>7}{'of':>8}{'rate':>8}")
    for f in LAB:
        r = per_f[f] / tot_f[f] if tot_f[f] else float("nan")
        print(f"  {f:<18}{per_f[f]:>7}{tot_f[f]:>8}{r:>8.4f}")

    ml = m.merge(lang, on="StudyInstanceUID", how="left")
    print(f"\n  {'language':<26}{'edits':>7}{'of':>8}{'rate':>8}")
    per_l = {}
    for lg, s in ml.groupby("language"):
        per_l[lg] = {"edits": int(s.edit.sum()), "cells": int(len(s)),
                     "rate": float(s.edit.mean())}
        print(f"  {str(lg):<26}{per_l[lg]['edits']:>7}{per_l[lg]['cells']:>8}"
              f"{per_l[lg]['rate']:>8.4f}")

    if frac > MAX_EDIT_FRACTION and not a.force:
        raise SystemExit(
            f"\nREFUSING TO WRITE: {frac:.4f} of cells would be edited, above the "
            f"{MAX_EDIT_FRACTION} cap. At this scale N_train is not 'C with corrections', it is a "
            f"different key, and comparing a model trained on it against C would confound which "
            f"cells changed with how confident every cell is. Report this and pre-register the "
            f"replacement as its own experiment, or pass --force with a reason.")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    W = C.copy().set_index("StudyInstanceUID")
    for r in m[m.edit].itertuples():
        if np.isfinite(r.new_value):
            W.at[r.StudyInstanceUID, r.finding] = r.new_value
    W = W.reset_index()
    # the guarantee that makes this a minimal edit, checked rather than asserted
    diff = (W[LAB].to_numpy() != C[LAB].to_numpy())
    n_diff = int(diff.sum())
    if n_diff != n_edit - int((~np.isfinite(m[m.edit].new_value)).sum()):
        print(f"WARNING: {n_diff} cells differ from C but {n_edit} edits were intended "
              f"(some edits may have been no-ops where C already held the target value)")
    print(f"\nN_train differs from C in {n_diff} cells ({n_diff / C[LAB].size:.4f} of the key); "
          f"every other cell is bit-identical")
    W.to_csv(out / "llm_labels_N_train.csv", index=False)
    m[m.edit].to_csv(out / "n_train_edits.csv", index=False)
    json.dump({"source_C": a.labels, "n1": a.n1, "n2": a.n2,
               "value_scale": C_SCALE, "max_edit_fraction": MAX_EDIT_FRACTION,
               "forced": bool(a.force),
               "n_cells": n_cells, "n_edits": n_edit, "edit_fraction": frac,
               "two_pass_stability": float(m.stable.mean()),
               "adds_positive": int((m.edit & (m.b1 == 1)).sum()),
               "removes_positive": int((m.edit & (m.b1 == 0)).sum()),
               "per_finding": {f: int(per_f[f]) for f in LAB}, "per_language": per_l,
               "cells_differing_from_C": n_diff},
              io.open(out / "n_train_summary.json", "w", encoding="utf-8"), indent=1)
    print(f"wrote {out}/llm_labels_N_train.csv and n_train_edits.csv")


if __name__ == "__main__":
    main()
