# -*- coding: utf-8 -*-
"""Build the blinded human-check sheets for the five language x finding cells 0A flagged.

WHY THESE FIVE. 0A found ACL to be the cross-language axis -- strongly negative in Turkish
(D -0.106) and Croatian/Serbian (-0.112) and strongly positive in Dutch (+0.121) -- plus Greek x
Synovitis (-0.122). A finding genuinely harder to SEE on Turkish scans has no reason to be unusually
easy on Dutch ones, so the question is whether the LABEL differs by language. A human reading the
reports is the only instrument that answers it. Spanish/Portuguese meniscus is included as the
control cell: it was NOT flagged, so agreement there is what "no effect" looks like.

SELECTION, fixed here and not revisited. Within each cell, 10 studies with C >= 0.5 and 10 with
C < 0.5 on that finding, sampled at random under SEED. Balanced rather than prevalence-matched,
because a random 20 Turkish reports would contain ~3 ACL positives and answer nothing; stratifying
on C is a LABEL-based split, not a model-prediction-based one -- no vision-model output touches this
selection, which is the line arm15 crossed.

BLINDING. The sheet the reader fills in does not show C's value, N's status, or which cell the
report came from: those live in a separate key file, joined afterwards by row_id. A reader who can
see the label being audited is not an independent instrument.

REPORT TEXT IS COMPETITION DATA. The sheets contain verbatim report text, so they are written to
kaggle/labelaudit/sheets/ which is gitignored, and must not be committed, uploaded to a dataset, or
sent to any translation or annotation service. For Greek and Bulgarian, glosses must come from a
model running locally or inside a Kaggle kernel -- pass --gloss-from with that output.
"""
import argparse
import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260928
N_PER_SIDE = 10
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# (language, finding, why it is here)
CELLS = [
    ("Turkish", "ACL", "0A D=-0.106, survives slot control"),
    ("Croatian_Serbian_Bosnian", "ACL", "0A D=-0.112, survives slot control"),
    ("Dutch", "ACL", "0A D=+0.121, survives slot control -- the opposite sign"),
    ("Greek", "Synovitis", "0A D=-0.122; C has 33 positives in 321 Greek studies"),
    ("Spanish_or_Portuguese", "Medial Meniscus", "CONTROL cell -- not flagged by 0A"),
]

# Terms that place the finding in the report, so the reader is pointed at the right sentence.
# Deliberately recall-oriented: a missed term costs a reader a few seconds, a wrong one costs nothing
# because the reader sees the whole report anyway. This is NOT used to decide anything.
# NOTE ON GREEK. 305 of the 321 Greek reports write MICRO SIGN U+00B5 where Greek small mu U+03BC
# belongs, and no other language in the corpus contains that character. Patterns below are therefore
# matched against NFKC-normalised text (see norm()), which folds the two. Written with the real mu,
# the Synovitis stem matched 3 Greek reports instead of 44.
TERMS = {
    # Croatian/Serbian/Bosnian: the corpus says "prednjeg kriznog ligamenta" (krizni = cruciate),
    # NOT "ukrizeni", which is what I first guessed -- that guess found the sentence in 1 of 10
    # C-positive reports. Verified against the 131 Croatian C-positive ACL studies.
    # Turkish: the bare "capraz bag" (no "on") appears in a third of the C-positive ACL reports;
    # matching only the full form covered 0.687 of them, adding the bare form covers 1.000.
    "ACL": [r"\bACL\b", r"anterior cruciate", r"\u00d6\u00c7B", r"\u00e7apraz ba[\u011fg]\w*", r"\bVKB\b", r"\bLCA\b",
            r"prednj\w* kri\u017en\w*", r"prednj\w* ukri\u017een\w*", r"kri\u017eni ligament",
            r"voorste kruisband", r"\bVKB\b", r"ligamento cruzado anterior", r"cruzado anterior",
            r"\u03c0\u03c1\u03cc\u03c3\u03b8\u03b9\w* \u03c7\u03b9\u03b1\u03c3\u03c4", r"\u03c7\u03b9\u03b1\u03c3\u03c4\u03bf\u03b5\u03b9\u03b4", r"\u043f\u0440\u0435\u0434\u0435\u043d \u043a\u0440\u044a\u0441\u0442\u043d",
            r"vorderes? Kreuzband", r"ligament crois\u00e9 ant\u00e9rieur",
            # the tear word matters as much as the ligament name
            r"ruptur\w*", r"parcijaln\w*", r"kompletn\w*"],
    "Synovitis": [r"synovit", r"sinovit", r"synovial", r"sinovial", r"\u03c5\u03bc\u03b5\u03bd\u03af\u03c4\u03b9\u03b4",
                  r"\u03c5\u03bc\u03ad\u03bd\w*", r"\u03c3\u03c5\u03bd\u03bf\u03b2\u03b9\u03b1\u03ba\w*", r"\u03b1\u03c1\u03b8\u03c1\u03b9\u03ba\w* \u03c5\u03bc",
                  r"\u0441\u0438\u043d\u043e\u0432\u0438\u0442", r"Synovialitis", r"synovi[i\u00ed]tis"],
    "Medial Meniscus": [r"medial meniscus", r"i[\u00e7c] menisk", r"medijaln\w* meniskus",
                        r"unutra\u0161nj\w* meniskus", r"mediale meniscus",
                        r"menisco (medial|interno)", r"\u03ad\u03c3\u03c9 \u03bc\u03b7\u03bd\u03af\u03c3\u03ba",
                        r"\u0432\u044a\u0442\u0440\u0435\u0448\u043d\u0438\u044f? \u043c\u0435\u043d\u0438\u0441\u043a", r"Innenmeniskus", r"m\u00e9nisque interne"],
}


def norm(s):
    """NFKC first: see the note on Greek above -- without it the Greek patterns miss 93% of hits."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or ""))).strip()


def context(text, finding, width=240):
    """Windows around every term hit, so a reader does not have to scan a 3,000-character report."""
    t = norm(text)
    outs, seen = [], []
    for pat in TERMS.get(finding, []):
        for m in re.finditer(pat, t, re.I):
            a = max(0, m.start() - width // 3)
            b = min(len(t), m.start() + width)
            if any(a < e and s < b for s, e in seen):
                continue
            seen.append((a, b))
            outs.append(("..." if a else "") + t[a:b] + ("..." if b < len(t) else ""))
    return "  ||  ".join(outs[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kaggle/labelaudit/sheets")
    ap.add_argument("--train", default="data/raw/train.csv")
    ap.add_argument("--labels", default="data/external/stevenleehans_rsna-knee-llm-report-labels/"
                                       "llm_labels_v4_blend.csv")
    ap.add_argument("--language", default="kaggle/labelaudit/output/a1_language.csv")
    ap.add_argument("--extract", default=None,
                    help="an extract_N*.csv, to attach N's status/evidence to the KEY (not the sheet)")
    ap.add_argument("--gloss-from", default=None,
                    help="CSV with StudyInstanceUID,gloss produced by a LOCAL or in-kernel model")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(a.train)
    tr["StudyInstanceUID"] = tr.StudyInstanceUID.astype(str)
    host = set(tr[tr[LAB].notna().all(axis=1)].StudyInstanceUID)
    C = pd.read_csv(a.labels)
    C["StudyInstanceUID"] = C.StudyInstanceUID.astype(str)
    lang = pd.read_csv(a.language)
    lang["StudyInstanceUID"] = lang.StudyInstanceUID.astype(str)
    df = (tr[["StudyInstanceUID", "Report"]]
          .merge(C[["StudyInstanceUID"] + LAB], on="StudyInstanceUID", how="left")
          .merge(lang[["StudyInstanceUID", "language"]], on="StudyInstanceUID", how="left"))
    df["language"] = df.language.fillna("unassigned")
    # the 58 host-annotated studies are excluded: they are the extractor-accuracy instrument and
    # must not also become the human-audit sample
    df = df[~df.StudyInstanceUID.isin(host)]
    gloss = {}
    if a.gloss_from:
        g = pd.read_csv(a.gloss_from)
        gloss = dict(zip(g.StudyInstanceUID.astype(str), g.gloss))

    N = None
    if a.extract:
        N = pd.read_csv(a.extract)
        N["StudyInstanceUID"] = N.StudyInstanceUID.astype(str)

    rng = np.random.default_rng(SEED)
    sheet_rows, key_rows, summary = [], [], []
    rid = 0
    for Lname, finding, why in CELLS:
        sub = df[df.language == Lname]
        pos = sub[sub[finding] >= 0.5]
        neg = sub[sub[finding] < 0.5]
        n_pos, n_neg = min(N_PER_SIDE, len(pos)), min(N_PER_SIDE, len(neg))
        take = pd.concat([pos.sample(n_pos, random_state=int(rng.integers(1 << 31))),
                          neg.sample(n_neg, random_state=int(rng.integers(1 << 31)))])
        summary.append({"language": Lname, "finding": finding, "why": why,
                        "available": int(len(sub)), "C_positive_available": int(len(pos)),
                        "sampled_positive": int(n_pos), "sampled_negative": int(n_neg)})
        for _, r in take.iterrows():
            rid += 1
            row_id = f"R{rid:03d}"
            sheet_rows.append({
                "row_id": row_id,
                "finding_to_judge": finding,
                "report": norm(r.Report),
                "english_gloss": gloss.get(r.StudyInstanceUID, ""),
                "relevant_excerpt": context(r.Report, finding),
                # the reader fills these in; nothing else is shown
                "your_verdict_present_absent_unclear": "",
                "your_confidence_1_to_5": "",
                "is_it_a_tear_or_only_signal_change": "",
                "notes": ""})
            k = {"row_id": row_id, "StudyInstanceUID": r.StudyInstanceUID, "language": Lname,
                 "finding": finding, "C_value": float(r[finding]),
                 "C_binary": int(r[finding] >= 0.5)}
            if N is not None:
                m = N[(N.StudyInstanceUID == r.StudyInstanceUID) & (N.finding == finding)]
                if len(m):
                    k["N_status"] = m.status.iloc[0]
                    k["N_severity"] = m.severity.iloc[0]
                    k["N_evidence"] = m.evidence.iloc[0]
                    if finding == "ACL" and "acl_substatus" in m:
                        k["N_acl_substatus"] = m.acl_substatus.iloc[0]
            key_rows.append(k)

    sheets = pd.DataFrame(sheet_rows).sample(frac=1.0, random_state=SEED)   # shuffle across cells
    sheets.to_csv(out / "human_check_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key_rows).to_csv(out / "human_check_key.csv", index=False, encoding="utf-8-sig")
    json.dump({"seed": SEED, "n_per_side": N_PER_SIDE, "cells": summary,
               "n_rows": len(sheets), "gloss_attached": bool(gloss),
               "extract_attached": a.extract or None,
               "warning": "contains verbatim report text -- Competition Data, do not commit or "
                          "upload"},
              io.open(out / "human_check_manifest.json", "w", encoding="utf-8"), indent=1)

    print(f"{len(sheets)} rows over {len(CELLS)} cells -> {out}/human_check_sheet.csv (blinded)")
    print(f"key with C and N -> {out}/human_check_key.csv")
    for s in summary:
        print(f"  {s['language']:<24} {s['finding']:<16} {s['sampled_positive']}+/"
              f"{s['sampled_negative']}-  (of {s['available']} studies, "
              f"{s['C_positive_available']} C-positive)")
    miss = [s for s in summary if s["sampled_positive"] < N_PER_SIDE
            or s["sampled_negative"] < N_PER_SIDE]
    if miss:
        print(f"\nUNDERFILLED cells (too few studies on one side): "
              f"{[(m['language'], m['finding']) for m in miss]}")
    if not gloss:
        print("\nNo English gloss attached. Greek and Bulgarian rows are unreadable without one; "
              "produce it locally or in a Kaggle kernel and pass --gloss-from. Do NOT use an "
              "online translation service: the report text is Competition Data.")


if __name__ == "__main__":
    main()
