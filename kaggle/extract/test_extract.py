# -*- coding: utf-8 -*-
"""Offline tests for the Stage 1 extractor: the parts that can be wrong without the GPU noticing.

The expensive failure mode here is not a crash -- it is a run that completes and produces labels we
then trust. So the tests target the validator and the prompt contract: evidence that is not in the
report must be rejected, every cell dict must carry the same keys (a ragged CSV silently drops
columns), and the ACL sub-status must be constrained to its pre-registered vocabulary.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("EXTRACT_WORK", os.environ.get("TEMP", "."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_labels import (ACL_SUB_BINARISE, LAB, build_prompt, norm, parse_json,  # noqa: E402
                            validate)

REPORT = ("MR KNIE RECHTS. Voorste kruisband: volledige ruptuur met discontinuiteit. "
          "Mediale meniscus: horizontale scheur achterhoorn. Geringe gewrichtsuitzetting. "
          "Geen fractuur.")


def cell(finding, **kw):
    d = {"finding": finding, "status": "not_mentioned", "severity": "unspecified",
         "evidence": "", "compartment": None}
    d.update(kw)
    return d


def full(**over):
    """A complete twelve-finding answer, with specific findings overridden."""
    return {"report_language": "nl",
            "findings": [over.get(f, cell(f)) for f in LAB]}


fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        fails.append(name)


# 1. a good answer survives, and severity / compartment / sub-status are preserved verbatim
obj = full(ACL=cell("ACL", status="positive", severity="large_severe",
                    evidence="volledige ruptuur met discontinuiteit",
                    acl_substatus="complete_tear"),
           **{"Medial Meniscus": cell("Medial Meniscus", status="positive", severity="unspecified",
                                      evidence="horizontale scheur achterhoorn",
                                      compartment="medial"),
              "Effusion": cell("Effusion", status="positive", severity="small_mild",
                               evidence="Geringe gewrichtsuitzetting"),
              "Fracture": cell("Fracture", status="negative", severity="none",
                               evidence="Geen fractuur")})
cells, problems, rl = validate(obj, REPORT)
check("valid answer accepted", cells is not None and not problems, str(problems))
check("severity preserved", cells["Effusion"]["severity"] == "small_mild")
check("compartment preserved", cells["Medial Meniscus"]["compartment"] == "medial")
check("language returned", rl == "nl")
check("ACL sub-status preserved", cells["ACL"]["acl_substatus"] == "complete_tear")

# 2. the anti-confabulation check: evidence that is not in the report must be caught
bad = full(ACL=cell("ACL", status="positive", evidence="complete rupture of the ACL",
                    acl_substatus="complete_tear"))
cells, problems, _ = validate(bad, REPORT)
check("confabulated evidence rejected",
      any("not a substring" in p for p in problems) and not cells["ACL"]["evidence_ok"])

# 2b. ... but whitespace differences must NOT be treated as confabulation
ws = full(ACL=cell("ACL", status="positive", evidence="volledige   ruptuur\nmet discontinuiteit",
                   acl_substatus="complete_tear"))
cells, problems, _ = validate(ws, REPORT)
check("whitespace-normalised evidence accepted", cells["ACL"]["evidence_ok"] and not problems,
      str(problems))

# 3. a missing finding becomes invalid rather than vanishing
miss = {"report_language": "nl", "findings": [cell(f) for f in LAB if f != "Synovitis"]}
cells, problems, _ = validate(miss, REPORT)
check("missing finding -> invalid",
      cells["Synovitis"]["status"] == "invalid" and any("missing" in p for p in problems))

# 4. every cell dict carries identical keys, including the invalid ones: the CSV is built from
#    **cell, so a missing key becomes a silently ragged column
keys = {frozenset(c) for c in cells.values()}
check("cell keys uniform across valid and invalid", len(keys) == 1, str(keys))
check("acl_substatus present on every cell",
      all("acl_substatus" in c for c in cells.values()))
check("acl_substatus blank on non-ACL findings",
      all(cells[f]["acl_substatus"] == "" for f in LAB if f != "ACL"))

# 5. the sub-status vocabulary is closed: a plausible-sounding invention must not pass through
inv = full(ACL=cell("ACL", status="positive", evidence="volledige ruptuur met discontinuiteit",
                    acl_substatus="high_grade_partial"))
cells, problems, _ = validate(inv, REPORT)
check("unknown sub-status -> unspecified",
      cells["ACL"]["acl_substatus"] == "unspecified"
      and any("acl_substatus" in p for p in problems))

# 6. the pre-registered binarisation covers the whole vocabulary stated in the prompt, and only
#    tears map to 1
VOCAB = {"complete_tear", "partial_tear", "sprain_strain", "mucoid_degeneration",
         "reconstructed_graft_intact", "graft_tear", "normal", "not_mentioned"}
check("binarisation covers the prompt vocabulary", VOCAB <= set(ACL_SUB_BINARISE),
      str(VOCAB - set(ACL_SUB_BINARISE)))
check("only tears binarise to 1",
      {k for k, v in ACL_SUB_BINARISE.items() if v == 1}
      == {"complete_tear", "partial_tear", "graft_tear"})
check("unspecified falls back to status", ACL_SUB_BINARISE["unspecified"] is None)

# 7. bad status is flagged, not coerced into something scoreable
bs = full(ACL=cell("ACL", status="probably", evidence="volledige ruptuur met discontinuiteit"))
cells, problems, _ = validate(bs, REPORT)
check("bad status -> invalid", cells["ACL"]["status"] == "invalid"
      and any("bad status" in p for p in problems))

# 8. the model's JSON is never clean: fences, prose and trailing commas all have to parse
check("fenced JSON parses", parse_json('```json\n{"a": 1}\n```') == {"a": 1})
check("trailing comma parses", parse_json('{"a": [1, 2,],}') == {"a": [1, 2]})
check("prose around JSON parses", parse_json('Sure, here it is:\n{"a": 1}\nHope that helps')
      == {"a": 1})
check("unparseable -> None", parse_json("I cannot answer that.") is None)

# 9. the prompt must actually state the sub-status contract, in both the rules and the skeleton
msgs = build_prompt(REPORT)
u = msgs[-1]["content"]
check("prompt states the sub-status rule", "acl_substatus" in u and "mucoid_degeneration" in u)
check("prompt keeps the report verbatim", REPORT in u)
check("prompt still states the twelve findings", all(f in u for f in LAB))
check("prompt names every sub-status value", all(v in u for v in VOCAB))

# 10. norm is what the substring check relies on
check("norm collapses whitespace", norm("  a \n\t b  ") == "a b")
check("norm handles None", norm(None) == "")

# 11. the Greek confusable. 305 of 321 Greek reports write MICRO SIGN U+00B5 where Greek small mu
#     U+03BC belongs, and no other language in the corpus contains the character. The two are
#     visually identical, so the model will often emit the real mu when copying evidence verbatim.
#     Without NFKC on both sides, the substring check rejects it and most Greek cells become
#     invalid -- a total failure of the Greek arm of the audit for a typographic reason.
GREEK = "\u03a5\u03bc\u03b5\u03bd\u03af\u03c4\u03b9\u03b4\u03b1 \u03c4\u03bf\u03c5 "         # with U+03BC
MICRO = GREEK.replace("\u03bc", "\u00b5")                                    # as the corpus writes it
check("micro sign and mu normalise together", norm(MICRO) == norm(GREEK))
greek_report = "MRI \u03b3\u03cc\u03bd\u03b1\u03c4\u03bf\u03c2. " + MICRO + "\u03b1\u03c1\u03b8\u03c1\u03b9\u03ba\u03bf\u03cd \u03c5\u03bc\u03ad\u03bd\u03b1."
obj_g = full(Synovitis=cell("Synovitis", status="positive", severity="unspecified",
                            evidence=GREEK.strip()))
cells_g, problems_g, _ = validate(obj_g, greek_report)
check("mu-spelled evidence accepted against micro-sign report",
      cells_g["Synovitis"]["evidence_ok"] and not problems_g, str(problems_g))

# 12. ... and the check must still be strict about CONTENT, not merely about codepoints
obj_t = full(Synovitis=cell("Synovitis", status="positive", evidence="synovitis of the joint"))
cells_t, problems_t, _ = validate(obj_t, greek_report)
check("translated evidence still rejected",
      not cells_t["Synovitis"]["evidence_ok"]
      and any("not a substring" in x for x in problems_t))

# 12b. the negative-vs-not_mentioned rule must be stated in the prompt, because probe7 showed the
#      model defaulting to "negative" for findings the report never discusses: 270 negatives against
#      11 not_mentioned, 69.6% of those negatives carrying empty evidence, evidence-substring 0.296
#      on negatives against 0.971 on positives. Both binarise to 0, so the labels survive -- but the
#      A2.2 contingency compares C's "not addressed" band against N's status, and it is empty if N
#      never says not_mentioned.
u2 = build_prompt(REPORT)[-1]["content"]
check("prompt distinguishes denial from silence",
      "Silence is not denial" in u2 and "not_mentioned" in u2)
check("prompt says most findings are not_mentioned", "MOST findings in MOST" in u2)

# 12c. a negative with empty evidence must still fail the check, since that is the behaviour the
#      prompt change targets and a silent pass would hide a regression
ne = full(**{"Baker's": cell("Baker's", status="negative", evidence="")})
cells_n, problems_n, _ = validate(ne, REPORT)
check("negative with empty evidence is rejected",
      not cells_n["Baker's"]["evidence_ok"]
      and any("not a substring" in x for x in problems_n))
check("not_mentioned with empty evidence is accepted",
      validate(full(), REPORT)[0]["Baker's"]["evidence_ok"])

# 13. sharding. Two workers must cover every study exactly once: an overlap double-counts a study
#     in the merged corpus and a gap silently shrinks it, and the parent's row-count check is the
#     only thing standing between a partial extraction and a label set we would then trust.
ids = [f"S{i:04d}" for i in range(4407)]
for n_shard in (1, 2, 3, 4):
    parts = [[s for i, s in enumerate(ids) if i % n_shard == k] for k in range(n_shard)]
    union = [x for part in parts for x in part]
    check(f"shards cover every study exactly once (n={n_shard})",
          sorted(union) == sorted(ids) and len(union) == len(set(union)),
          f"{len(union)} vs {len(ids)}, {len(set(union))} unique")
    check(f"shards are balanced (n={n_shard})",
          max(len(x) for x in parts) - min(len(x) for x in parts) <= 1,
          str([len(x) for x in parts]))

print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
