# RSNA Knee Abnormality Detection

Starter repo for the Kaggle competition:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

- **Task**: multi-label detection of 12 knee findings from DICOM MRI series.
- **Metric**: ROC AUC (macro, across the 12 finding columns)
- **Category**: Research, prize pool $77,000 USD
- **Deadline**: 2026-10-22

## The actual problem

This is not a plain image classifier. `train.csv` has 4,407 studies, each
with a free-text radiology `Report` (multilingual), but structured 0/1
labels for only **58 of them (1.3%)**. `test.csv` has no `Report` column at
all — at inference the model only ever sees images.

So there are two chained sub-problems:

1. **Label mining (text → labels, training-time only).** The `Report` field
   is the only label source for 98.7% of training studies. It can only be
   used to derive training targets, never as a model input, since it doesn't
   exist at test time.
2. **Multi-instance, multi-view vision (images → 12 probabilities).** Each
   study has multiple DICOM series (different planes: sagittal/coronal/axial,
   different sequences: fluid-sensitive/fat-suppressed — see
   `{train,test}_series.csv`), each a stack of slices. The model pools
   evidence across a variable number of 3D volumes per patient into one
   study-level multi-label prediction.

The 12 target columns (`sample_submission.csv`): `ACL`, `MCL`,
`Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`,
`Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.

## Data split (this repo's convention)

The 58 gold-labeled studies are the only ground truth available, so they are
held out as an internal **test set**, never trained on — matching what
established public solutions for this competition do (see Label sourcing
below). Everything else is training data, labeled via mined report text.

```
data/processed/gold_test.csv    58 studies, real 0/1 labels. Never train on this.
data/processed/train_pool.csv   4,349 studies, soft pseudo-labels (0-1) per
                                 finding + original Report (kept for reference/
                                 future re-labeling ONLY — never feed Report
                                 into the image model, test has no Report).
data/processed/label_source_validation.csv   per-finding AUC of each
                                 candidate label source against gold_test.
```

Built by `python scripts/prepare_dataset.py`.

## Label sourcing (preliminary)

Report text was checked on Kaggle's public kernels/datasets for this
competition before building anything from scratch — several teams have
already solved "report → structured label" with an LLM and published the
result, and the top public solutions (0.92-0.935 LB) are built on exactly
this gold-58-as-holdout convention. `scripts/prepare_dataset.py` validates
five candidate label sources against `gold_test` and picks the best:

| source | macro AUC vs gold |
|---|---|
| **stevenleehans/rsna-knee-llm-report-labels (v4_blend)** | **0.893** (chosen) |
| stevenleehans/rsna-knee-llm-report-labels (v2) | 0.887 |
| stevenleehans/rsna-knee-llm-report-labels (full) | 0.878 |
| pilkwang/rsna-knee-llm-labels | 0.870 |
| an ensemble of v4_blend + pilkwang | 0.893 (no better than v4_blend alone) |

All three source datasets are CC0-1.0 and downloaded into `data/external/`
(gitignored). `data/external/pilkwang_rsna-knee-llm-labels/api_labeler.py`
is a full reference implementation worth reading: it prompts `claude-opus-5`
per report with a JSON-schema-constrained YES/NO/UNK + severity verdict for
each of the 12 findings, prompt-cached and batched. It's a solid template if
we later want to run our own extraction pass (needs `ANTHROPIC_API_KEY`;
not run yet — this preliminary pass reuses validated public labels instead).

Known limitation, per the community writeups: report-derived labels run
~82-89% concordant with gold, and the disagreement is systematic — reports
don't encode the severity threshold a real adjudicator applies, so naive
labels skew toward over-calling borderline findings positive. Don't fit
these labels too closely (label smoothing / mixup / early stopping on the
gold set, not on report-teacher validation loss).

## Status

Kaggle account `zeroday19` has accepted the competition rules — downloads
work. DICOM series themselves (`train_series/`, `test_series/`) have not
been pulled yet (large — dataset size scan was still running last time it
was checked; train_series alone is 100,000+ files and tens of GB). Only the
small CSVs plus the community label datasets have been downloaded so far.

## Trained model weights

Checkpoints from the `kaggle/arm*` training runs are too large for git and
are attached instead to the
[`model-weights`](https://github.com/XERO47/rsna-knee-mri/releases/tag/model-weights)
release (deduplicated — files with identical content across `output/`,
`upload/`, and `kaggle_datasets/` copies are uploaded once). Match a release
asset back to how it was produced via the corresponding `kaggle/arm*/train_*.py`
and `kernel-metadata.json`.

## Data files

```
train.csv            StudyInstanceUID, Report, + 12 finding columns (58 populated)
train_series.csv     StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive,
                      Fat_Suppression, Anatomical_Plane
train_series/<study_uid>/<series_uid>/<instance_uid>.dcm
test.csv              StudyInstanceUID only
test_series.csv       same columns as train_series.csv (no Report anywhere)
test_series/<study_uid>/<series_uid>/<instance_uid>.dcm
sample_submission.csv StudyInstanceUID + 12 finding columns
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python scripts/download_data.py --dest data/raw --unzip   # DICOM series (large)
python scripts/prepare_dataset.py                          # builds gold_test / train_pool
```

## Layout

```
configs/            training config (YAML)
data/raw/           downloaded competition data (gitignored)
data/external/      community label datasets pulled from Kaggle (gitignored)
data/processed/     gold_test.csv, train_pool.csv (gitignored, rebuild via script)
models/             checkpoints (gitignored)
scripts/            download_data.py, prepare_dataset.py
src/                dataset, model, training code
notebooks/          EDA / experiments
```

## Training

```bash
python -m src.train --config configs/config.yaml
```

`src/train.py` is a skeleton — the multi-instance-learning vision model
(attention pooling over per-series slice windows, 12-way sigmoid head) still
needs to be built. Reference architectures worth reading before writing it
are in `data/external/` and public kernels (attention-MIL over 2D-CNN or
CoAtNet backbones on 3-slice windows, per-finding attention weights, no
horizontal flip — sagittal knee MRI flips ACL/PCL anatomy).
