# -*- coding: utf-8 -*-
"""Pick N2 between two candidate models by a rule fixed before either has been run.

WHY A RULE AND NOT A LOOK. N2 exists to disagree with N1 in a useful way, and "useful" is exactly the
kind of judgement that, made after seeing the numbers, selects the model that flatters whatever
hypothesis is in favour that day. So the rule is written here first:

    SCORE(model, language) = schema_validity(language) x evidence_substring_rate(language)
    SELECT the model with the highest MINIMUM score over
            Turkish, Croatian_Serbian_Bosnian, Greek, Bulgarian, Dutch
    TIE-BREAK (within 0.02) by human spot-check of 10 reports per language.

The minimum, not the mean. A model that reads English and Spanish beautifully and cannot read
Bulgarian is useless here, because the four newly-identified families are 25% of the corpus and are
where 0A found its interactions. The mean would hide exactly that.

The 58 gold studies are NOT used to choose between the candidates. They measure extractor accuracy,
and a model picked for agreeing with them would make the later accuracy measurement circular.

CANDIDATES: google/gemma-3-12b-it and mistralai/Mistral-Nemo-Instruct-2407. Neither has a usable
quantised build on Kaggle, so both are pulled from pinned Hugging Face revisions with internet
enabled and quantised in-kernel. If the budget does not allow the bake-off, the fallback agreed in
advance is N2 = N1 under an independently phrased prompt, which is then EXCLUDED from the error
independence criterion and from the A5 closure test -- two prompts on one model are not two readers.
"""
import argparse
import io
import json
import math
from pathlib import Path

import pandas as pd

LANGS = ["Turkish", "Croatian_Serbian_Bosnian", "Greek", "Bulgarian", "Dutch"]
TIE_BAND = 0.02


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def score(path, name):
    d = pd.read_csv(path)
    d["StudyInstanceUID"] = d.StudyInstanceUID.astype(str)
    for c in ("status", "language"):
        d[c] = d[c].fillna("")
    if "evidence_ok" not in d:
        raise RuntimeError(f"{path} has no evidence_ok column -- not an extraction output")
    rows = {}
    for lg in LANGS:
        s = d[d.language == lg]
        if s.empty:
            rows[lg] = {"n_cells": 0, "validity": float("nan"), "substring": float("nan"),
                        "score": float("nan")}
            continue
        # schema validity is per STUDY: a study is valid if none of its twelve cells came back
        # invalid, because a single invalid cell makes that study's row unusable for the contingency
        per_study = s.groupby("StudyInstanceUID").status.apply(lambda x: (x != "invalid").all())
        validity = float(per_study.mean())
        claim = s[s.status.isin(["positive", "negative", "uncertain"])]
        substring = float(claim.evidence_ok.astype(bool).mean()) if len(claim) else float("nan")
        _, v_lo, _ = wilson(int(per_study.sum()), len(per_study))
        rows[lg] = {"n_studies": int(len(per_study)), "n_cells": int(len(s)),
                    "validity": validity, "validity_lo": v_lo,
                    "n_claims": int(len(claim)), "substring": substring,
                    "score": validity * (substring if substring == substring else 0.0)}
    finite = [v["score"] for v in rows.values() if v["score"] == v["score"]]
    return {"model": name, "path": str(path), "per_language": rows,
            "min_score": min(finite) if finite else float("nan"),
            "mean_score": sum(finite) / len(finite) if finite else float("nan"),
            "n_languages_scored": len(finite)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="candidate A extraction CSV")
    ap.add_argument("--b", required=True, help="candidate B extraction CSV")
    ap.add_argument("--name-a", default="gemma-3-12b-it")
    ap.add_argument("--name-b", default="Mistral-Nemo-Instruct-2407")
    ap.add_argument("--out", default="kaggle/labelaudit/output")
    args = ap.parse_args()

    A, B = score(args.a, args.name_a), score(args.b, args.name_b)
    print("PRE-REGISTERED RULE: highest MINIMUM of validity x substring over "
          + ", ".join(LANGS) + f"; tie band {TIE_BAND}\n")
    for r in (A, B):
        print(f"{r['model']}")
        print(f"  {'language':<26}{'studies':>8}{'validity':>10}{'substring':>11}{'score':>8}")
        for lg in LANGS:
            v = r["per_language"][lg]
            print(f"  {lg:<26}{v.get('n_studies', 0):>8}{v['validity']:>10.4f}"
                  f"{v['substring']:>11.4f}{v['score']:>8.4f}")
        print(f"  MIN {r['min_score']:.4f} | mean {r['mean_score']:.4f} "
              f"({r['n_languages_scored']}/{len(LANGS)} languages scored)\n")

    gap = A["min_score"] - B["min_score"]
    if abs(gap) < TIE_BAND:
        winner, how = None, (f"TIE: minimum scores differ by {abs(gap):.4f} < {TIE_BAND}. "
                             f"The rule says human spot-check, 10 reports per language, before "
                             f"either is adopted. No automatic winner.")
    else:
        winner = A["model"] if gap > 0 else B["model"]
        how = (f"{winner} wins on the pre-registered minimum "
               f"({max(A['min_score'], B['min_score']):.4f} vs "
               f"{min(A['min_score'], B['min_score']):.4f}).")
    print(how)
    worst = {r["model"]: min(LANGS, key=lambda lg: (r["per_language"][lg]["score"]
                                                    if r["per_language"][lg]["score"] ==
                                                    r["per_language"][lg]["score"] else 9))
             for r in (A, B)}
    print(f"weakest language per candidate: {worst}")
    if winner and max(A["min_score"], B["min_score"]) < 0.80:
        print("\nWARNING: even the winner falls below 0.80 on its weakest language. A second "
              "reader that cannot read a quarter of the corpus cannot support the consensus mask; "
              "report before adopting it.")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({"rule": {"metric": "validity x evidence_substring", "aggregate": "minimum",
                        "languages": LANGS, "tie_band": TIE_BAND,
                        "gold_58_used": False},
               "candidates": [A, B], "winner": winner, "decision": how,
               "weakest_language": worst},
              io.open(out / "n2_bakeoff.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out}/n2_bakeoff.json")


if __name__ == "__main__":
    main()
