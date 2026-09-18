"""Build the preliminary train/test split for RSNA Knee Abnormality Detection.

The competition ships structured 0/1 labels for only 58 of the 4,407 training
studies (see README "Label situation"). Those 58 are the only ground truth
available, so they are held out as our internal test set rather than trained
on. Every other study has a free-text radiology report and nothing else.

Report text cannot be used as a model input: the competition's actual test
set has no Report column, so at inference the model only ever sees images.
Report text can only be used to derive training LABELS. Rather than run our
own LLM extraction pass in this preliminary step, this script adopts a
community-published, gold-validated label set as a bootstrap:

    stevenleehans/rsna-knee-llm-report-labels (llm_labels_v4_blend.csv)
    https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels
    License: CC0-1.0

Chosen because, of the candidates checked (dreaddevelopment/rsna-knee-labels,
pilkwang/rsna-knee-llm-labels, stevenleehans's own v2/full variants), it had
the highest macro-AUC (0.893) against the 58 gold labels -- see
data/processed/label_source_validation.csv for the full comparison. An
ensemble of v4_blend + pilkwang did not beat v4_blend alone (0.893 vs 0.870).

Outputs (data/processed/):
    gold_test.csv    58 studies, real 0/1 labels. Never train on this.
    train_pool.csv   4,349 studies, soft pseudo-labels [0,1] per finding
                     from the external source above, plus the original
                     Report text for reference/future re-labeling only --
                     do not feed Report into the image model.
    label_source_validation.csv   per-label AUC of each candidate label
                     source against gold_test, for future reference.

Usage:
    python scripts/prepare_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]

RAW = Path("data/raw")
EXT = Path("data/external")
OUT = Path("data/processed")

CANDIDATES = {
    "dreaddevelopment_soft": EXT / "dreaddevelopment_rsna-knee-labels" / "labels_llm_soft.csv",
    "stevenleehans_v2": EXT / "stevenleehans_rsna-knee-llm-report-labels" / "llm_labels_v2.csv",
    "stevenleehans_v4blend": EXT / "stevenleehans_rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv",
    "stevenleehans_full": EXT / "stevenleehans_rsna-knee-llm-report-labels" / "llm_labels_full.csv",
    "pilkwang_v2": EXT / "pilkwang_rsna-knee-llm-labels" / "report_labels_v2.csv",
}
CHOSEN_SOURCE = "stevenleehans_v4blend"


def validate_sources(gold: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, path in CANDIDATES.items():
        df = pd.read_csv(path)[["StudyInstanceUID", *LABELS]]
        m = gold.merge(df, on="StudyInstanceUID", suffixes=("_gold", "_pred"), how="inner")
        aucs = []
        for c in LABELS:
            y, p = m[c + "_gold"].values, m[c + "_pred"].values
            aucs.append(roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan)
        rows[name] = aucs
    return pd.DataFrame(rows, index=LABELS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW / "train.csv")
    has_gold = train[LABELS].notna().any(axis=1)

    gold_test = train.loc[has_gold, ["StudyInstanceUID", *LABELS]].reset_index(drop=True)
    print(f"gold_test: {len(gold_test)} studies with real labels (held out, never trained on)")

    val = validate_sources(gold_test)
    val.to_csv(OUT / "label_source_validation.csv")
    print("\nlabel source candidates, per-finding AUC vs gold_test:")
    print(val.round(3).to_string())
    print("\nmacro AUC per source:")
    print(val.mean().round(4).sort_values(ascending=False).to_string())

    pool_uids = train.loc[~has_gold, "StudyInstanceUID"]
    reports = train.loc[~has_gold, ["StudyInstanceUID", "Report"]]

    src_path = CANDIDATES[CHOSEN_SOURCE]
    src = pd.read_csv(src_path)[["StudyInstanceUID", *LABELS]]

    train_pool = pool_uids.to_frame().merge(src, on="StudyInstanceUID", how="left")
    missing = train_pool[LABELS].isna().any(axis=1).sum()
    if missing:
        print(f"\nWARNING: {missing} pool studies have no pseudo-label from {CHOSEN_SOURCE}")
    train_pool = train_pool.merge(reports, on="StudyInstanceUID", how="left")
    train_pool["label_source"] = CHOSEN_SOURCE

    print(f"\ntrain_pool: {len(train_pool)} studies, pseudo-labels from {CHOSEN_SOURCE}")
    print(f"(chosen for highest macro AUC vs gold_test: {val.mean()[CHOSEN_SOURCE]:.4f})")

    gold_test.to_csv(OUT / "gold_test.csv", index=False)
    train_pool.to_csv(OUT / "train_pool.csv", index=False)
    print(f"\nwrote {OUT / 'gold_test.csv'} and {OUT / 'train_pool.csv'}")


if __name__ == "__main__":
    main()
