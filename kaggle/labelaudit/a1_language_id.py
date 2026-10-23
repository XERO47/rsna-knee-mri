"""A1: proper language identification for all 4,407 reports, replacing the regex probes.

WHY THIS COMES FIRST. Stage 0A had to exclude 40.1% of the corpus -- 372 studies matched several
of the documented EDA probes and 1,371 matched none -- because those probes are three clinical
words per language, overlapping by construction and never intended as language identification.
Worse, Stage 0B showed the same class of instrument failing silently: a hand-built vocabulary can
measure itself rather than the corpus (the retracted 49% figure). Every language-conditioned claim
downstream inherits whatever this step gets wrong, so it is done with a real classifier.

    fastText lid.176.bin, 131.3 MB, sha256 7e69ec5451bc261c...
    176 languages, trained by Facebook Research on Wikipedia/Tatoeba/SETimes.

Licence note: the lid.176 model is CC BY-SA 3.0. It is an ANALYSIS tool -- it reads report text to
group studies for auditing and never enters a trained model or a submission -- so it does not touch
the permissive-only rule that governs backbones.

WHAT IT DOES
  * strips the boilerplate that defeats language ID: bracketed placeholders like [DATE], exam
    headers, measurements and pure punctuation runs. A report that is 60% "[DATE] [TIME] MRI KNEE"
    identifies as English regardless of the body text.
  * classifies the cleaned text, keeping the top-2 labels and their probabilities.
  * assigns a language when the top probability clears a threshold; otherwise leaves it unassigned
    and counts it, rather than forcing a guess.
  * reports agreement with the old regex probes, so the change is auditable rather than asserted.

Reported: coverage against the <5% unassigned target, the language distribution, and every
disagreement class against the old probes.
"""
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "models/lid.176.bin"
MIN_PROB = 0.50
MIN_CHARS = 40
OLD_PROBES = {
    "English": r"\btear\b|\bmeniscus\b|\beffusion\b",
    "Spanish_or_Portuguese": r"\brotura\b|\bmenisco\b|\bderrame\b",
    "Italian": r"\blesione\b|\bmenisco\b|\bversamento\b",
    "French": r"\bd.chirure\b|\bm.nisque\b|\b.panchement\b",
    "German": r"\briss\b|\bmeniskus\b|\berguss\b",
    "Turkish": r"\byirtik\b|\bmenisk.s\b|\befuzyon\b",
}
# fastText codes grouped to the granularity the audit needs. Spanish and Portuguese stay merged
# because the old analysis merged them and because they share the compartment vocabulary that
# matters for the extraction traps (interno/externo).
# The documented EDA named six language families. fastText finds at least ten: Greek (321),
# Bulgarian (220), Croatian/Serbian (~400) and Dutch (153) are ~25% of the corpus and were
# invisible to the regex probes, which is why Stage 0A had to exclude 40% of the studies.
GROUP = {"en": "English", "es": "Spanish_or_Portuguese", "pt": "Spanish_or_Portuguese",
         "tr": "Turkish", "it": "Italian", "fr": "French", "de": "German",
         "el": "Greek", "bg": "Bulgarian", "nl": "Dutch",
         # fastText splits the Serbo-Croatian continuum across four codes, so no single one
         # clears a 0.5 threshold even when the text is plainly Croatian. Grouping them and
         # SUMMING their probabilities is the fix; thresholding the top code alone discarded
         # 403 real reports.
         "sh": "Croatian_Serbian_Bosnian", "hr": "Croatian_Serbian_Bosnian",
         "bs": "Croatian_Serbian_Bosnian", "sr": "Croatian_Serbian_Bosnian",
         "mk": "Macedonian", "ro": "Romanian", "pl": "Polish", "ru": "Russian",
         "cs": "Czech", "hu": "Hungarian", "sl": "Slovenian", "sq": "Albanian"}


def clean(text):
    """Remove what makes clinical reports hard to identify: placeholders, headers, numbers."""
    t = str(text or "")
    t = re.sub(r"\[[^\]]*\]", " ", t)                       # [DATE], [TIME], [NAME]
    t = re.sub(r"(?i)\b(exam|study|patient|mri|mr|date|time|indication|technique|protocol|"
               r"series|sequence|wo|with|without|contrast)\b\s*(type|name|and)?\s*:?", " ", t)
    t = re.sub(r"\d+([.,]\d+)?\s*(mm|cm|ml|mrad|t)?", " ", t)
    t = re.sub(r"[^\w\sÀ-ɏ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    import fasttext
    ft = fasttext.load_model(MODEL)

    raw = pd.read_csv("data/raw/train.csv")
    raw["StudyInstanceUID"] = raw.StudyInstanceUID.astype(str)
    rows = []
    for sid, rep in zip(raw.StudyInstanceUID, raw.Report):
        c = clean(rep)
        if len(c) < MIN_CHARS:
            rows.append({"StudyInstanceUID": sid, "lang_code": "", "prob": 0.0,
                         "lang2": "", "prob2": 0.0, "language": "", "n_chars": len(c),
                         "reason": "too_short_after_cleaning"})
            continue
        labels, probs = ft.predict(c, k=6)
        codes = [l.replace("__label__", "") for l in labels]
        p = [float(x) for x in probs]
        # Probability is accumulated per GROUP, not per code, so a language spread across
        # several codes is not penalised for it.
        acc = {}
        for cc, pp in zip(codes, p):
            acc[GROUP.get(cc, f"other:{cc}")] = acc.get(GROUP.get(cc, f"other:{cc}"), 0.0) + pp
        best = max(acc, key=acc.get)
        top, tp = codes[0], acc[best]
        lang = best if tp >= MIN_PROB else ""
        rows.append({"StudyInstanceUID": sid, "lang_code": top, "prob": tp,
                     "lang2": codes[1] if len(codes) > 1 else "",
                     "prob2": p[1] if len(p) > 1 else 0.0,
                     "language": lang, "group_prob": tp, "n_chars": len(c),
                     "reason": "" if lang else f"low_confidence({tp:.2f})"})
    df = pd.DataFrame(rows)

    unassigned = (df.language == "").sum()
    print(f"{len(df)} reports | unassigned {unassigned} ({100 * unassigned / len(df):.2f}%) "
          f"-- target < 5%")
    print(f"  reasons: " + ", ".join(f"{k} {v}" for k, v in
                                     df[df.language == ""].reason.value_counts().items()))
    print(f"\nlanguage distribution (fastText lid.176, threshold {MIN_PROB}):")
    vc = df.language.value_counts()
    for k, v in vc.items():
        if k:
            print(f"  {k:<26} {v:>5} ({100 * v / len(df):>5.1f}%)")
    print(f"  mean grouped probability {df[df.language != ''].group_prob.mean():.3f}")

    # raw code distribution, so a mis-grouped code is visible rather than hidden in "other:"
    print(f"\ntop fastText codes:")
    for k, v in df.lang_code.value_counts().head(12).items():
        print(f"  {k or '(none)':<8} {v:>5}   -> {GROUP.get(k, 'other/unmapped')}")

    # agreement with the old regex probes
    old = []
    for rep in raw.Report:
        low = re.sub(r"\s+", " ", str(rep or "")).lower()
        hits = [n for n, p in OLD_PROBES.items() if re.search(p, low)]
        old.append(hits[0] if len(hits) == 1 else ("MULTI" if hits else "NONE"))
    df["old_probe"] = old
    both = df[(df.language != "") & (~df.old_probe.isin(["MULTI", "NONE"]))]
    agree = (both.language == both.old_probe).mean()
    print(f"\nvs the old regex probes:")
    print(f"  old assigned exactly one language: "
          f"{(~df.old_probe.isin(['MULTI','NONE'])).sum()} ({100 * (~df.old_probe.isin(['MULTI','NONE'])).mean():.1f}%)")
    print(f"  new assigned a language:           {len(df) - unassigned} "
          f"({100 * (1 - unassigned / len(df)):.1f}%)")
    print(f"  where both assigned, they agree on {100 * agree:.1f}% ({len(both)} studies)")
    dis = both[both.language != both.old_probe]
    if len(dis):
        print(f"  {len(dis)} disagreements, most common:")
        for (o, n), c in dis.groupby(["old_probe", "language"]).size().nlargest(6).items():
            print(f"    old {o:<24} -> new {n:<24} {c:>4}")
    rec = df[df.old_probe.isin(["MULTI", "NONE"]) & (df.language != "")]
    print(f"\n  studies the old probes could NOT assign but fastText can: {len(rec)} "
          f"({100 * len(rec) / len(df):.1f}% of the corpus recovered)")
    for k, v in rec.language.value_counts().head(8).items():
        print(f"    {k:<26} {v:>5}")

    df.to_csv(out / "a1_language.csv", index=False)
    json.dump({"model": MODEL, "sha256_prefix": "7e69ec5451bc261c",
               "min_prob": MIN_PROB, "n_reports": int(len(df)),
               "unassigned": int(unassigned), "unassigned_share": float(unassigned / len(df)),
               "target_met": bool(unassigned / len(df) < 0.05),
               "distribution": {k: int(v) for k, v in vc.items() if k},
               "old_probe_agreement": float(agree),
               "recovered_from_old_unassigned": int(len(rec)),
               "licence_note": "lid.176 is CC BY-SA 3.0; analysis tool only, never in a model "
                               "or submission"},
              io.open(out / "a1_language_summary.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out / 'a1_language.csv'} and a1_language_summary.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle/labelaudit/output")
