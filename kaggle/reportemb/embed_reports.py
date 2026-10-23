"""reportemb: embed all 4,407 radiology reports once, so the trainer never sees text.

WHY. Every training label we have is an LLM extraction that compresses a whole radiology
report into 12 bits, and those labels are the measured ceiling: our 4-member blend scores
0.9016 against the 58 gold studies while the LABELS themselves score 0.8930. The model
already agrees with truth better than its own training targets do, so fitting those targets
harder cannot help -- which is exactly what arm15 (label refinement, +0.001 LB), arm18
(label masking, -0.0003 OOF) and five aggregation heads each demonstrated.

This is the privileged-information route instead: the report is a TEACHER at training time
and is deleted afterwards.

    TRAIN   MRI -> image encoder -> pathology logits
                              \\-> semantic projection --cosine--> frozen report embedding
    TEST    MRI -> image encoder -> pathology logits

The model being trained is f(MRI) -> y, never f(MRI, report) -> y. The report embedding
appears only in a loss term. At inference there is no text encoder, no report, and nothing
report-derived -- which matters because the hidden test has no Report field at all.

WHAT THIS KERNEL DOES. Encodes each report once with a frozen sentence encoder and writes
a (4407, 384) float32 matrix. The trainer then loads an array; it never imports
sentence-transformers and never needs internet. Doing it here also means the encoder is
downloaded once rather than on every training run.

MODEL. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, license **apache-2.0**
(verified against the HuggingFace model API, 2026-09-25), which satisfies our
permissive-only rule. Multilingual is not optional here: the EDA found vocabulary evidence
of English (1,833 reports), Spanish/Portuguese (550), Turkish (523), Italian (368), French
(81) and German (46). An English-only encoder would silently produce garbage for a third
of the corpus.

CHUNKING. The model's max_seq_length is 128 tokens but reports run to 690 words (median
129, p95 330). Encoding the raw string would truncate most of them. Each report is
therefore split on sentence boundaries into chunks that fit, every chunk is encoded, and
the L2-normalised chunk vectors are averaged and renormalised, so the whole report
contributes.

BUILT-IN VALIDATION. The EDA found 204 studies in 54 groups whose normalised report text
is identical. Those groups MUST come out with cosine ~1.0 between members. If they do not,
the encoder or the chunking is broken and the output should not be used. This is checked
here and the kernel fails loudly rather than writing a bad matrix.
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EXPECT_LICENSE = "apache-2.0"
MAX_TOKENS = 128
BATCH = 128
WORK = Path("/kaggle/working")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_comp():
    for d, dirs, fs in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if "train.csv" in fs:
            return Path(d)
    raise FileNotFoundError("competition data not attached")


def chunks_for(text, tokenizer, limit):
    """Sentence-boundary chunks that each fit under `limit` tokens."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return [""]
    sentences = [s for s in re.split(r"(?<=[.;:!?])\s+", text) if s]
    out, cur = [], ""
    for sent in sentences:
        trial = (cur + " " + sent).strip()
        if len(tokenizer.tokenize(trial)) <= limit - 2 or not cur:
            cur = trial
        else:
            out.append(cur)
            cur = sent
        # a single sentence longer than the limit: hard-split it on words
        while len(tokenizer.tokenize(cur)) > limit - 2:
            words = cur.split()
            half = max(1, len(words) // 2)
            out.append(" ".join(words[:half]))
            cur = " ".join(words[half:])
    if cur:
        out.append(cur)
    return out or [text]


def main():
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    comp = find_comp()
    train = pd.read_csv(comp / "train.csv")
    ids = train.StudyInstanceUID.astype(str).tolist()
    reports = train["Report"].fillna("").tolist()
    log(f"{len(ids)} studies | median report {int(np.median([len(r) for r in reports]))} chars")

    model = SentenceTransformer(MODEL)
    model.max_seq_length = MAX_TOKENS
    tok = model.tokenizer
    dim = model.get_sentence_embedding_dimension()
    log(f"{MODEL} | dim {dim} | max_seq_length {model.max_seq_length}")

    # one flat list of chunks, so the GPU/CPU batch stays full
    flat, owner = [], []
    for i, r in enumerate(reports):
        cs = chunks_for(r, tok, MAX_TOKENS)
        flat.extend(cs)
        owner.extend([i] * len(cs))
    owner = np.asarray(owner)
    per = np.bincount(owner, minlength=len(ids))
    log(f"{len(flat)} chunks | per report: median {int(np.median(per))}, max {int(per.max())}, "
        f"{int((per > 1).sum())} reports needed more than one chunk")

    vecs = model.encode(flat, batch_size=BATCH, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    log(f"encoded in {(time.time() - t0) / 60:.1f} min")

    emb = np.zeros((len(ids), dim), np.float32)
    for i in range(len(ids)):
        v = vecs[owner == i].mean(0)
        n = np.linalg.norm(v)
        emb[i] = v / n if n > 1e-8 else v
    norms = np.linalg.norm(emb, axis=1)
    if not np.all(np.isfinite(emb)):
        raise RuntimeError("non-finite report embedding")
    log(f"embedding norms: min {norms.min():.4f} max {norms.max():.4f} (expect ~1)")

    # ---- validation: identical report text must give identical embeddings ----
    norm_text = [re.sub(r"\s+", " ", str(r or "")).strip().lower() for r in reports]
    groups = {}
    for i, t in enumerate(norm_text):
        groups.setdefault(t, []).append(i)
    dup = {t: g for t, g in groups.items() if len(g) > 1 and t}
    worst, checked = 1.0, 0
    for g in dup.values():
        M = emb[g]
        sims = M @ M.T
        worst = min(worst, float(sims[np.triu_indices(len(g), 1)].min()))
        checked += 1
    log(f"duplicate-text groups: {checked} (expected ~54 from the EDA), "
        f"worst within-group cosine {worst:.6f}")
    if checked and worst < 0.999:
        raise RuntimeError(f"identical reports gave different embeddings (cos {worst:.4f}) "
                           f"-- encoder or chunking is broken; not writing the matrix")

    # a weak sanity signal: unrelated reports should NOT be near-identical
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(ids), 2000), rng.integers(0, len(ids), 2000)
    keep = a != b
    rand_cos = float((emb[a[keep]] * emb[b[keep]]).sum(1).mean())
    log(f"mean cosine between random report pairs: {rand_cos:.4f} "
        f"(near 1.0 would mean the embedding is degenerate)")

    np.save(WORK / "report_emb.npy", emb)
    np.save(WORK / "report_emb_ids.npy", np.asarray(ids, dtype="U"))
    json.dump({"model": MODEL, "license": EXPECT_LICENSE, "dim": int(dim),
               "max_seq_length": MAX_TOKENS, "n": len(ids), "chunks": int(len(flat)),
               "duplicate_groups": checked, "worst_duplicate_cosine": worst,
               "mean_random_pair_cosine": rand_cos,
               "minutes": (time.time() - t0) / 60},
              open(WORK / "report_emb_meta.json", "w"), indent=1)
    log(f"wrote report_emb.npy {emb.shape} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
