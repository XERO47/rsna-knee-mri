# -*- coding: utf-8 -*-
"""Stage 1 scoring: everything decided from N, with every rule fixed before N existed.

This file is the pre-registration. It is written and tested before the extraction runs, so that
thresholds, the binarisation and the two predictions cannot be chosen after seeing which choice
gives the interesting answer. That is not a formality here: arm15 bought +0.008 on gold and +0.001
on the leaderboard by letting model output pick which labels to change, and the 49% "no report
mention" figure had to be retracted because the instrument behind it was never checked.

WHAT IT COMPUTES
  A5   Step-0 closure test -- can a competent report reader reach the host's labels at all?
  A2.2 C x N contingency on C's own four-level value convention, not on a binary threshold.
  P1   pre-registered: ACL stable disagreement higher in Turkish/Croatian than English/Dutch.
  P2   pre-registered: Synovitis disagreement is directional by language.
  IND  error independence -- does consensus masking filter C's errors, or just repeat them?
  GATE 1 criteria, including (c), which can stop the whole line.

WHAT IT MAY NOT DO. The 58 host-annotated studies measure extractor accuracy and nothing else. No
model prediction enters any selection here. C is soft and is never flattened to a binary except
where the pre-registered binarisation says so, and that binarisation is stated in one place below.
"""
import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEED = 20260928
N_BOOT = 2000

# C's agreement with the 58, from Stage 0B. Criterion (c) compares N against this number.
C_GOLD_AGREEMENT = 0.8175287356321839

# ---------------------------------------------------------------------------------------------
# PRE-REGISTERED BINARISATION. Stated once, used everywhere.
#
# N: a positive is a positive; uncertain, not_mentioned and negative are all 0. This follows the
# host's apparent convention -- Stage 0B found C's errors are 70% false positives, the signature of
# an annotator who grades a borderline mention negative -- and it was fixed before Stage 1 ran.
# `invalid` is not a label: those cells are dropped, never silently counted as 0.
N_POSITIVE = {"positive"}
N_NEGATIVE = {"negative", "uncertain", "not_mentioned"}

# C's four-level value convention, recovered in A2 from the values themselves and calibrated
# against the 58 (host-positive rate in brackets):
#     <= 0.05   explicit negative              [0.00 - 0.05]
#     ~ 0.25    the report does not address it  [0.070]
#     ~ 0.92    hedged / positive-leaning       [0.44]
#     >= 0.935  explicit positive               [0.80 - 0.92]
# The middle two are NOT collapsed: 0.25 is silence and 0.92 is a hedge, and treating them as one
# "uncertain" bucket would destroy exactly the distinction this stage is testing.
#
# VERIFIED on the 58 before N existed -- host-positive rate per band, with Wilson 95%:
#     negative       194 cells  0.0412  [0.0210, 0.0792]
#     not_addressed  208 cells  0.1394  [0.0989, 0.1931]
#     hedged          99 cells  0.4040  [0.3127, 0.5025]
#     positive       195 cells  0.8359  [0.7775, 0.8813]
# All three adjacent pairs are non-overlapping, so the scale is ordinal and the bands are distinct.
# (0.070 is the rate at the exact value 0.25; the band runs to 0.45 and includes 0.33 and 0.41.)
def c_level(v):
    if v <= 0.05:
        return "negative"
    if v <= 0.45:
        return "not_addressed"
    if v < 0.935:
        return "hedged"
    return "positive"


C_LEVELS = ["negative", "not_addressed", "hedged", "positive"]
# and the binary collapse, used only where a binary is required
def c_binary(v):
    return int(v > 0.5)


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval. Normal approximations are wrong at the rates we care about here."""
    if n == 0:
        return float("nan"), 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n, label=""):
    p, lo, hi = wilson(k, n)
    return f"{label}{k}/{n} = {p:.4f} [{lo:.4f}, {hi:.4f}]"


def load_extract(path):
    """One extraction pass, long format: one row per (study, finding)."""
    d = pd.read_csv(path)
    d["StudyInstanceUID"] = d.StudyInstanceUID.astype(str)
    for col in ("status", "severity", "evidence", "acl_substatus"):
        if col not in d:
            d[col] = ""
        d[col] = d[col].fillna("")
    if "evidence_ok" not in d:
        d["evidence_ok"] = True
    return d


def n_binary(row, acl_binarise):
    """N's status -> {0, 1, None}. None means 'not a label' and is dropped, never counted as 0.

    For ACL the pre-registered sub-status mapping takes precedence where it is decisive, because a
    sprain and a mucoid degeneration are mentions of the ACL but are not tears."""
    st = str(row["status"])
    if st == "invalid":
        return None
    if row["finding"] == "ACL" and acl_binarise:
        sub = str(row.get("acl_substatus", "") or "")
        v = acl_binarise.get(sub)
        if v is not None:
            return int(v)
    if st in N_POSITIVE:
        return 1
    if st in N_NEGATIVE:
        return 0
    return None


def paired_bootstrap(studies, stat, rng):
    """Study-level resampling: cells within a study are correlated, so the study is the unit."""
    uniq = np.array(sorted(set(studies)))
    out = []
    for _ in range(N_BOOT):
        take = rng.choice(uniq, len(uniq), replace=True)
        v = stat(take)
        if v is not None and np.isfinite(v):
            out.append(v)
    return np.array(out)


# =============================================================================================
def a5_closure(N, host, out, acl_bin):
    """A5. Can a competent reader of the REPORT reach the host's labels at all?

    Step 0 -- whether the host's 58 labels were annotated from images or derived from the reports --
    was never resolved; it was stated as an assumption and I earlier mis-attributed it to a host
    clarification. This test closes it from our own data instead of asserting it.

    Restrict to cells where N is as confident as N ever gets: a positive or negative status, with
    evidence that verified as a substring. On those cells a report-derived label set should be
    reachable. A ceiling well below 1 means the host's labels carry information the reports do not.

    THE THRESHOLD IS NOT ARBITRARY. Checked against C before N existed, by restricting C to the
    cells where C itself is most confident: C agrees with the host on 0.8175 of all 696 cells, but
    0.8972 on its explicit cells (<=0.05 or >=0.935), 0.9189 at <=0.025/>=0.97, and 0.9597 at
    <=0.015/>=0.98. So 0.90 is well above what C manages overall and about where C lands when
    confident: clearing it is achievable but not automatic, and N merely matching ~0.897 would mean
    N is no better than C where it counts. (Selecting on C's confidence also selects easy cells,
    which is why A5 selects on N's confidence and verified evidence instead, and scores C on the
    identical cells.)

    PRE-REGISTERED READING, fixed before N existed:
        N agreement >= 0.90  -> the reports do contain the host's information; Step 0 resolves
                                toward report-derived, or at least report-recoverable.
        N agreement <  0.90  -> the host's labels carry information not present in the reports,
                                which is what image annotation looks like. Report-based label
                                repair then has a hard ceiling, and Stage 2 must respect it.
    C is scored on the SAME restricted cells, so the two numbers are comparable rather than
    merely adjacent."""
    print("\n" + "=" * 92)
    print("A5. STEP-0 CLOSURE TEST -- is the host's label reachable from the report?")
    print("=" * 92)
    conf = N[(N.status.isin(["positive", "negative"])) & (N.evidence_ok.astype(bool))]
    rows = []
    for _, r in conf.iterrows():
        g = host.get((r.StudyInstanceUID, r.finding))
        if g is None:
            continue
        nb = n_binary(r, acl_bin)
        if nb is None:
            continue
        rows.append({"study": r.StudyInstanceUID, "finding": r.finding, "gold": g[0],
                     "C": g[1], "N": nb})
    d = pd.DataFrame(rows)
    if d.empty:
        print("  no confident N cells on the 58 host studies -- cannot run the closure test")
        return {"ran": False}
    n_ok = int((d.N == d.gold).sum())
    c_ok = int((d.C.map(c_binary) == d.gold).sum())
    print(f"  confident N cells on the 58: {len(d)} "
          f"({len(d) / (58 * 12):.1%} of all 696 cells)")
    print(f"  {fmt(n_ok, len(d), 'N agreement with host : ')}")
    print(f"  {fmt(c_ok, len(d), 'C agreement with host : ')}  (same cells)")
    verdict = ("reports CONTAIN the host's information -- Step 0 resolves toward report-derived"
               if n_ok / len(d) >= 0.90 else
               "host labels carry information NOT in the reports -- consistent with image "
               "annotation; report-based repair has a ceiling")
    print(f"  PRE-REGISTERED VERDICT (threshold 0.90): {verdict}")
    per = []
    for f in LAB:
        s = d[d.finding == f]
        if len(s) >= 10:
            p, lo, hi = wilson(int((s.N == s.gold).sum()), len(s))
            per.append({"finding": f, "n": len(s), "N_agreement": p, "lo": lo, "hi": hi})
    if per:
        print(f"\n  {'finding':<18}{'cells':>6}{'N agrees':>10}  interval")
        for r in sorted(per, key=lambda x: x["N_agreement"]):
            print(f"  {r['finding']:<18}{r['n']:>6}{r['N_agreement']:>10.4f}  "
                  f"[{r['lo']:.4f}, {r['hi']:.4f}]")
    return {"ran": True, "n_cells": len(d), "N_agreement": n_ok / len(d),
            "C_agreement_same_cells": c_ok / len(d), "threshold": 0.90,
            "verdict": verdict, "per_finding": per}


def a22_contingency(N, C, out, acl_bin):
    """A2.2. Where does N land inside each of C's four value levels?

    C is soft. Collapsing it to a binary before comparing would throw away the only part of C that
    records how sure it was, and 0.25 ('the report does not address this', 19% of all cells) would
    be scored as if it were a confident statement."""
    print("\n" + "=" * 92)
    print("A2.2. C x N CONTINGENCY on C's own four-level value convention")
    print("=" * 92)
    d = N.merge(C, on=["StudyInstanceUID", "finding"], how="inner")
    d["c_level"] = d.C_value.map(c_level)
    d["n_state"] = np.where(d.status.isin(["positive", "negative", "uncertain", "not_mentioned"]),
                            d.status, "invalid")
    tab = pd.crosstab(d.c_level, d.n_state).reindex(index=C_LEVELS).fillna(0).astype(int)
    order = [c for c in ["positive", "uncertain", "not_mentioned", "negative", "invalid"]
             if c in tab.columns]
    tab = tab[order]
    print(f"\n  rows = C level, columns = N status, {len(d)} cells\n")
    print("  " + tab.to_string().replace("\n", "\n  "))
    share = tab.div(tab.sum(axis=1), axis=0).round(3)
    print(f"\n  row-normalised:\n")
    print("  " + share.to_string().replace("\n", "\n  "))
    # the cell that matters most: C says 'the report does not address this', N says it does
    if "not_addressed" in tab.index and "positive" in tab.columns:
        k = int(tab.loc["not_addressed", "positive"])
        n = int(tab.loc["not_addressed"].sum())
        print(f"\n  C 'not addressed' but N reads a POSITIVE: {fmt(k, n)}")
        print("    These are cells where C recorded silence and an independent reader found a")
        print("    statement. They are the strongest candidates for genuine label repair.")
    if "positive" in tab.index:
        k = int(tab.loc["positive", [c for c in ("negative", "not_mentioned") if c in tab.columns]]
                .sum())
        n = int(tab.loc["positive"].sum())
        print(f"  C confident POSITIVE but N reads absent/unmentioned: {fmt(k, n)}")
        print("    Stage 0B says C's errors are 70% false positives; this is that population.")
    return {"table": tab.to_dict(), "row_normalised": share.to_dict(), "n_cells": len(d)}


def stable_disagreement(N1, N2, C, acl_bin):
    """Cells where the two passes AGREE with each other and both differ from C.

    Two passes are the point: a single pass disagreeing with C is as likely to be the new
    extractor's error as C's. Only a disagreement that replicates across independently prompted
    models is evidence about C."""
    a = N1[["StudyInstanceUID", "finding", "status", "acl_substatus"]].copy()
    a["nb"] = [n_binary(r, acl_bin) for _, r in N1.iterrows()]
    b = N2[["StudyInstanceUID", "finding", "status", "acl_substatus"]].copy()
    b["nb"] = [n_binary(r, acl_bin) for _, r in N2.iterrows()]
    d = a.merge(b, on=["StudyInstanceUID", "finding"], suffixes=("_1", "_2"))
    d = d.merge(C, on=["StudyInstanceUID", "finding"], how="inner")
    d = d[d.nb_1.notna() & d.nb_2.notna()]
    d["stable"] = d.nb_1 == d.nb_2
    d["c_bin"] = d.C_value.map(c_binary)
    d["disagree"] = d.stable & (d.nb_1 != d.c_bin)
    # direction, for P2: does N add a positive C missed, or remove one C invented?
    d["direction"] = np.where(~d.disagree, "",
                              np.where(d.nb_1 == 1, "N_adds_positive", "N_removes_positive"))
    return d


def p1_test(D, lang, out):
    """P1, pre-registered: ACL stable disagreement is higher in Turkish and Croatian/Serbian than
    in English and Dutch, with NON-OVERLAPPING Wilson intervals.

    This is what 0A's ACL pattern predicts if the cause is the label. If the cause were the images,
    the disagreement between two report readers and C would not track language this way."""
    print("\n" + "=" * 92)
    print("P1 (PRE-REGISTERED). ACL stable C-vs-N disagreement: Turkish/Croatian > English/Dutch")
    print("=" * 92)
    d = D[D.finding == "ACL"].merge(lang, on="StudyInstanceUID", how="left")
    hi_g = ["Turkish", "Croatian_Serbian_Bosnian"]
    lo_g = ["English", "Dutch"]
    rows = {}
    print(f"\n  {'language':<26}{'cells':>7}{'disagree':>10}  interval")
    for lg in hi_g + lo_g:
        s = d[d.language == lg]
        k, n = int(s.disagree.sum()), len(s)
        p, a, b = wilson(k, n)
        rows[lg] = {"k": k, "n": n, "rate": p, "lo": a, "hi": b}
        print(f"  {lg:<26}{n:>7}{p:>10.4f}  [{a:.4f}, {b:.4f}]")
    kh = sum(rows[g]["k"] for g in hi_g); nh = sum(rows[g]["n"] for g in hi_g)
    kl = sum(rows[g]["k"] for g in lo_g); nl = sum(rows[g]["n"] for g in lo_g)
    ph, ah, bh = wilson(kh, nh)
    pl, al, bl = wilson(kl, nl)
    print(f"\n  pooled Turkish+Croatian : {fmt(kh, nh)}")
    print(f"  pooled English+Dutch    : {fmt(kl, nl)}")
    held = ah > bl
    print(f"\n  P1 {'HOLDS' if held else 'FAILS'}: intervals "
          f"{'do not overlap' if held else 'overlap'} "
          f"({'higher' if ph > pl else 'not higher'} in Turkish/Croatian)")
    if not held:
        print("    P1 failing is evidence AGAINST the label explanation for 0A's ACL pattern.")
    return {"per_language": rows, "pooled_high": [kh, nh, ph], "pooled_low": [kl, nl, pl],
            "holds": bool(held)}


def p2_test(D, lang, out):
    """P2, pre-registered: Synovitis disagreement is DIRECTIONAL by language -- Greek shows
    C-positives that N reads as absent, Bulgarian shows N-positives that C missed.

    Restated in light of the blend inventory, which was established before Stage 1 ran and is
    recorded in EXPERIMENTS.md 10i: C's Synovitis is, for 83.7% of studies, a rescaled copy of C's
    Effusion field (the v2 dataset fills undecided Synovitis cells from Effusion). So C's Synovitis
    POSITIVES come from the ~1 cell in 6 where the report says the word -- and that is a reporting
    convention, not a disease rate. C's Bulgarian Synovitis positive rate is 0.005 against an
    Effusion rate of 0.609, so the missing positives cannot be inherited from Effusion."""
    print("\n" + "=" * 92)
    print("P2 (PRE-REGISTERED). Synovitis disagreement is directional by language")
    print("=" * 92)
    d = D[D.finding == "Synovitis"].merge(lang, on="StudyInstanceUID", how="left")
    print(f"\n  {'language':<26}{'cells':>7}{'N adds pos':>12}{'N removes pos':>15}")
    rows = {}
    for lg, s in d.groupby("language"):
        if len(s) < 20:
            continue
        add = int((s.direction == "N_adds_positive").sum())
        rem = int((s.direction == "N_removes_positive").sum())
        rows[lg] = {"n": len(s), "adds": add, "removes": rem,
                    "add_rate": add / len(s), "remove_rate": rem / len(s)}
        print(f"  {lg:<26}{len(s):>7}{add / len(s):>12.4f}{rem / len(s):>15.4f}")
    gr, bg = rows.get("Greek"), rows.get("Bulgarian")
    held = None
    if gr and bg:
        _, g_lo, g_hi = wilson(gr["removes"], gr["n"])
        _, b_lo, b_hi = wilson(bg["adds"], bg["n"])
        greek_ok = gr["remove_rate"] > gr["add_rate"]
        bulg_ok = bg["add_rate"] > bg["remove_rate"]
        held = bool(greek_ok and bulg_ok)
        print(f"\n  Greek: N removes more than it adds      -> {greek_ok} "
              f"(removes {gr['remove_rate']:.4f} [{g_lo:.4f}, {g_hi:.4f}] vs adds "
              f"{gr['add_rate']:.4f})")
        print(f"  Bulgarian: N adds more than it removes  -> {bulg_ok} "
              f"(adds {bg['add_rate']:.4f} [{b_lo:.4f}, {b_hi:.4f}] vs removes "
              f"{bg['remove_rate']:.4f})")
        print(f"\n  P2 {'HOLDS' if held else 'FAILS'}")
    else:
        print("\n  P2 cannot be evaluated: Greek or Bulgarian has too few cells")
    return {"per_language": rows, "holds": held}


def independence(N1, N2, host, acl_bin, out):
    """Does consensus masking filter C's errors, or does it just repeat them?

    A consensus mask only helps if the two passes fail differently. PRE-REGISTERED: if R_rep, the
    probability that N2 is wrong GIVEN N1 is wrong, exceeds 0.5, the two passes share their errors
    and the consensus mask is not an independent instrument -- Stage 2 does not proceed on it.

    WHAT THIS THRESHOLD CAN DETECT, worked out before it returned a number. Under independence
    R_rep is simply N's own error rate, so 0.5 is a test for SEVERE dependence, not a test of
    independence: at an error rate of 0.18 it needs a 2.8x lift, at 0.30 only 1.7x. It is also not
    sharp -- with ~125 error cells out of 696, a measured 0.50 has a Wilson interval of about
    [0.41, 0.58]. The threshold stays as pre-registered (moving it now is the move this apparatus
    exists to prevent), but R_rep / R_base is the informative statistic and is printed beside it:
    1.0 means fully independent errors. Read a borderline R_rep as "inconclusive, mask not
    established", never as a pass."""
    print("\n" + "=" * 92)
    print("ERROR INDEPENDENCE on the 58 (extractor measurement only)")
    print("=" * 92)
    a = {(r.StudyInstanceUID, r.finding): n_binary(r, acl_bin) for _, r in N1.iterrows()}
    b = {(r.StudyInstanceUID, r.finding): n_binary(r, acl_bin) for _, r in N2.iterrows()}
    w1, w2, both, n = [], [], 0, 0
    for key, (g, cv) in host.items():
        x, y = a.get(key), b.get(key)
        if x is None or y is None:
            continue
        n += 1
        e1, e2 = int(x != g), int(y != g)
        w1.append(e1); w2.append(e2); both += e1 * e2
    if not n:
        print("  no overlapping cells -- cannot measure independence")
        return {"ran": False}
    k1, k2 = sum(w1), sum(w2)
    r_base = k2 / n
    r_rep = both / k1 if k1 else float("nan")
    print(f"  cells scored: {n}")
    print(f"  {fmt(k1, n, 'N1 error rate            : ')}")
    print(f"  {fmt(k2, n, 'N2 error rate            : ')}")
    print(f"  R_base  P(N2 wrong)            = {r_base:.4f}")
    print(f"  {fmt(both, k1, 'R_rep   P(N2 wrong | N1 wrong) = ')}")
    indep = r_rep <= 0.5
    print(f"\n  PRE-REGISTERED: R_rep {'<=' if indep else '>'} 0.5 -> consensus mask "
          f"{'is usable' if indep else 'is NOT filtering C errors; Stage 2 does not proceed on it'}")
    if not np.isnan(r_rep) and r_base > 0:
        print(f"  (lift over independence: R_rep / R_base = {r_rep / r_base:.2f}x; "
              f"1.0 would be fully independent errors)")
    return {"ran": True, "n": n, "N1_error": k1 / n, "N2_error": r_base,
            "R_base": r_base, "R_rep": r_rep, "independent": bool(indep)}


def gate1(N1, host, acl_bin, closure, indep, out):
    """GATE 1. Criterion (c) is the one that can stop this line outright."""
    print("\n" + "=" * 92)
    print("GATE 1")
    print("=" * 92)
    k = n = 0
    rows = {(r.StudyInstanceUID, r.finding): n_binary(r, acl_bin) for _, r in N1.iterrows()}
    for key, (g, cv) in host.items():
        v = rows.get(key)
        if v is None:
            continue
        n += 1
        k += int(v == g)
    p, lo, hi = wilson(k, n)
    print(f"  (c) N1 agreement with the 58 gold studies: {fmt(k, n)}")
    print(f"      C's agreement (Stage 0B)             : {C_GOLD_AGREEMENT:.4f}")
    worse = p < C_GOLD_AGREEMENT
    why = ("replacing a measured extractor with a worse one is not an improvement" if worse
           else "N is at least as good as the label set it would replace")
    print(f"      -> {'STOP' if worse else 'PASS'}: N is "
          f"{'WORSE' if worse else 'not worse'} than C; {why}")
    print(f"  (A5) closure verdict: {closure.get('verdict', 'not run')}")
    if indep.get("ran"):
        print(f"  (IND) consensus mask usable: {indep['independent']}")
    proceed = (not worse) and indep.get("independent", True)
    print(f"\n  GATE 1: {'PROCEED to Stage 2' if proceed else 'STOP -- report and wait'}")
    return {"N1_gold_agreement": p, "N1_gold_lo": lo, "N1_gold_hi": hi,
            "C_gold_agreement": C_GOLD_AGREEMENT, "criterion_c_stop": bool(worse),
            "proceed": bool(proceed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", required=True, help="extract_N1_*.csv")
    ap.add_argument("--n2", default=None, help="extract_N2_*.csv (omit to skip P1/P2/independence)")
    ap.add_argument("--train", default="data/raw/train.csv")
    ap.add_argument("--labels", default="data/external/stevenleehans_rsna-knee-llm-report-labels/"
                                       "llm_labels_v4_blend.csv")
    ap.add_argument("--language", default="kaggle/labelaudit/output/a1_language.csv")
    ap.add_argument("--out", default="kaggle/labelaudit/output")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extract"))
    try:
        from extract_labels import ACL_SUB_BINARISE as acl_bin
    except Exception:
        acl_bin = None
        print("NOTE: extract_labels.ACL_SUB_BINARISE not importable; ACL falls back to `status`")

    N1 = load_extract(a.n1)
    N2 = load_extract(a.n2) if a.n2 else None
    tr = pd.read_csv(a.train); tr["StudyInstanceUID"] = tr.StudyInstanceUID.astype(str)
    Craw = pd.read_csv(a.labels); Craw["StudyInstanceUID"] = Craw.StudyInstanceUID.astype(str)
    lang = pd.read_csv(a.language)[["StudyInstanceUID", "language"]]
    lang["StudyInstanceUID"] = lang.StudyInstanceUID.astype(str)
    C = Craw.melt(id_vars="StudyInstanceUID", value_vars=LAB,
                  var_name="finding", value_name="C_value")
    gold = tr[tr[LAB].notna().all(axis=1)]
    host = {}
    cmap = {(r.StudyInstanceUID, r.finding): r.C_value for _, r in C.iterrows()}
    for _, r in gold.iterrows():
        for f in LAB:
            host[(r.StudyInstanceUID, f)] = (int(r[f] > 0.5),
                                             cmap.get((r.StudyInstanceUID, f), np.nan))
    print(f"N1 {a.n1}: {N1.StudyInstanceUID.nunique()} studies, {len(N1)} cells")
    if N2 is not None:
        print(f"N2 {a.n2}: {N2.StudyInstanceUID.nunique()} studies, {len(N2)} cells")
    print(f"host-annotated studies available: {len(gold)}")
    print(f"N1 status mix: {N1.status.value_counts().to_dict()}")
    print(f"N1 evidence-substring rate: {N1.evidence_ok.astype(bool).mean():.4f}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"n1": a.n1, "n2": a.n2, "seed": SEED,
           "binarisation": {"N_positive": sorted(N_POSITIVE), "N_zero": sorted(N_NEGATIVE),
                            "C_levels": C_LEVELS, "C_binary": "C > 0.5"}}
    res["a5_closure"] = a5_closure(N1, host, out, acl_bin)
    res["a22_contingency"] = a22_contingency(N1, C, out, acl_bin)
    if N2 is not None:
        D = stable_disagreement(N1, N2, C, acl_bin)
        stable = float(D.stable.mean())
        print(f"\ntwo-pass stability (N1 and N2 give the same binary): {stable:.4f} "
              f"over {len(D)} cells")
        res["two_pass_stability"] = stable
        res["p1"] = p1_test(D, lang, out)
        res["p2"] = p2_test(D, lang, out)
        res["independence"] = independence(N1, N2, host, acl_bin, out)
        D.to_csv(out / "stage1_stable_disagreement.csv", index=False)
    else:
        print("\nN2 absent: P1, P2, two-pass stability and the independence check are SKIPPED. "
              "They are not optional -- a single pass cannot distinguish C's errors from N's.")
        res["p1"] = res["p2"] = res["independence"] = {"ran": False}
    res["gate1"] = gate1(N1, host, acl_bin, res["a5_closure"],
                         res.get("independence", {}), out)
    json.dump(res, io.open(out / "stage1_summary.json", "w", encoding="utf-8"),
              indent=1, default=str)
    print(f"\nwrote {out}/stage1_summary.json")


if __name__ == "__main__":
    main()
