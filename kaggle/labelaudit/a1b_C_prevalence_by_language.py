"""A1b: is C blind in the languages Stage 0A never saw?

WHY THIS RUNS BEFORE 0A IS RE-RUN. A1 revealed that ~25% of the corpus is Croatian/Serbian (406),
Greek (321), Bulgarian (220) and Dutch (151) -- families the documented EDA never named and Stage
0A excluded entirely. Two of those use non-Latin scripts. If C's extractor was built and checked on
English reports, it may be close to blind there, and a blind extractor does not produce noisy
labels: it produces ZEROS. Stage 0A already showed the signature at small scale -- Spanish/
Portuguese Medial Meniscus had 1 observed positive against 42.7 expected.

An accuracy analysis cannot be read before this. If C assigns almost no positives for a language,
then AUC for that language is computed against a near-constant target, the interaction statistic
becomes meaningless rather than merely noisy, and "the model is bad at Greek" would be an artefact
of the labels having no signal to be good at.

WHAT IT REPORTS, purely descriptively:
  * C's positive rate per language x finding, binarised at 0.5 (Gate 1's convention)
  * the ratio to the corpus-wide rate for that finding, so a finding that is simply rare is not
    mistaken for a language that is unlabelled
  * cells that are effectively unlabelled: fewer than 5 positives, or a rate below a fifth of the
    corpus rate
  * the mean number of positive findings per study by language -- a report that yields one finding
    on average where English yields four is the clearest sign of extraction failure
  * C's raw score distribution: an extractor that is blind returns its prior, so the fraction of
    cells sitting at a single default value is diagnostic
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
MIN_STUDIES = 80


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    lang = pd.read_csv(out / "a1_language.csv")
    lang["StudyInstanceUID"] = lang.StudyInstanceUID.astype(str)
    lang["language"] = lang.language.fillna("")
    pool = pd.read_csv("data/processed/train_pool.csv")
    pool["StudyInstanceUID"] = pool.StudyInstanceUID.astype(str)
    df = pool.merge(lang[["StudyInstanceUID", "language"]], on="StudyInstanceUID", how="left")
    df["language"] = df.language.fillna("")
    print(f"train_pool studies {len(df)} | with a language {int((df.language != '').sum())}")

    C = (df[LAB].to_numpy() > 0.5).astype(int)
    corpus_rate = C.mean(0)
    langs = [L for L, n in df.language.value_counts().items() if L and n >= MIN_STUDIES]
    new = ["Croatian_Serbian_Bosnian", "Greek", "Bulgarian", "Dutch"]

    print(f"\nC POSITIVE RATE per language x finding (binarised at 0.5). "
          f"Corpus rate in the first row.")
    hdr = f"{'language':<26} {'n':>5} " + " ".join(f"{f[:9]:>9}" for f in LAB)
    print(hdr)
    print(f"{'CORPUS':<26} {len(df):>5} " + " ".join(f"{r:>9.3f}" for r in corpus_rate))
    rows, dead = {}, []
    for L in langs:
        m = (df.language == L).to_numpy()
        r = C[m].mean(0)
        tag = "  <- NEW" if L in new else ""
        print(f"{L:<26} {int(m.sum()):>5} " + " ".join(f"{x:>9.3f}" for x in r) + tag)
        rows[L] = {"n": int(m.sum()),
                   "rate": {LAB[j]: float(r[j]) for j in range(12)},
                   "positives": {LAB[j]: int(C[m][:, j].sum()) for j in range(12)}}
        for j in range(12):
            npos = int(C[m][:, j].sum())
            if npos < 5 or (corpus_rate[j] > 0 and r[j] < corpus_rate[j] / 5):
                dead.append((L, LAB[j], npos, float(r[j]), float(corpus_rate[j])))

    print(f"\nRATIO to the corpus rate (1.00 = same as corpus; <0.20 flagged)")
    print(hdr)
    for L in langs:
        m = (df.language == L).to_numpy()
        r = C[m].mean(0) / np.maximum(corpus_rate, 1e-9)
        cells = " ".join((f"{x:>9.2f}" if x >= 0.2 else f"{'*' + format(x, '.2f'):>9}")
                         for x in r)
        print(f"{L:<26} {int(m.sum()):>5} " + cells + ("  <- NEW" if L in new else ""))

    print(f"\nEFFECTIVELY UNLABELLED CELLS (<5 positives, or rate < 1/5 of corpus): {len(dead)}")
    for L, f, npos, r, cr in sorted(dead, key=lambda x: (x[0], x[1])):
        print(f"  {L:<26} {f:<18} {npos:>4} positives  rate {r:.4f} vs corpus {cr:.3f}")

    print(f"\nMEAN POSITIVE FINDINGS PER STUDY (12 possible)")
    for L in ["CORPUS"] + langs:
        m = np.ones(len(df), bool) if L == "CORPUS" else (df.language == L).to_numpy()
        print(f"  {L:<26} {C[m].sum(1).mean():>5.2f}" + ("  <- NEW" if L in new else ""))

    # a blind extractor returns its prior: look for mass at a single default value
    print(f"\nC RAW SCORE CONCENTRATION -- share of cells at the single most common value")
    raw = df[LAB].to_numpy()
    for L in ["CORPUS"] + langs:
        m = np.ones(len(df), bool) if L == "CORPUS" else (df.language == L).to_numpy()
        v, c = np.unique(np.round(raw[m], 3), return_counts=True)
        top = c.max() / c.sum()
        print(f"  {L:<26} {top:>5.3f} at value {v[c.argmax()]:.3f}" +
              ("  <- NEW" if L in new else ""))

    verdict = ("C IS NOT USABLE AS-IS for the flagged languages: those cells carry no positive "
               "signal, so accuracy statistics on them are meaningless"
               if dead else "no language x finding cell is effectively unlabelled")
    print(f"\n{verdict}")
    json.dump({"n_pool": int(len(df)), "languages": rows,
               "corpus_rate": {LAB[j]: float(corpus_rate[j]) for j in range(12)},
               "effectively_unlabelled": [{"language": a, "finding": b, "positives": c,
                                            "rate": d, "corpus_rate": e}
                                           for a, b, c, d, e in dead],
               "verdict": verdict},
              io.open(out / "a1b_C_prevalence.json", "w", encoding="utf-8"), indent=1)
    print(f"wrote {out / 'a1b_C_prevalence.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle/labelaudit/output")
