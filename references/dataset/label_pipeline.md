# Label pipeline — building gold_test.csv / train_pool.csv

`scripts/prepare_dataset.py` builds the validated split every `kaggle/arm*`
trainer consumes. Outputs land in `data/processed/` (gitignored, regenerable).
Do not substitute raw `train.csv` — its report-derived fields are not this
validated split.

## Inputs it requires

| Path the script hardcodes | Download with |
|---|---|
| `data/raw/train.csv` | `kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data/raw` |
| `data/external/dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv` | `kaggle datasets download -d dreaddevelopment/rsna-knee-labels -p data/external/dreaddevelopment_rsna-knee-labels --unzip` |
| `data/external/stevenleehans_rsna-knee-llm-report-labels/llm_labels_{v2,v4_blend,full}.csv` | `kaggle datasets download -d stevenleehans/rsna-knee-llm-report-labels -p data/external/stevenleehans_rsna-knee-llm-report-labels --unzip` |
| `data/external/pilkwang_rsna-knee-llm-labels/report_labels_v2.csv` | `kaggle datasets download -d pilkwang/rsna-knee-llm-labels -p data/external/pilkwang_rsna-knee-llm-labels --unzip` |

Two gotchas: the local directory name is the Kaggle slug with `/` replaced by
`_`, so the `-p` path must be spelled out explicitly; and the competition
single-file download writes a file literally named `DownloadDataFile` that must
be renamed to `train.csv`. `validate_sources()` reads **all five** candidate
CSVs and will crash if any is absent, even though only the chosen source feeds
`train_pool`. Do not use `scripts/download_data.py` for this — it pulls the
entire competition including the ~530 GiB DICOM tree.

Needs `pandas` + `scikit-learn`, which are not in `.venv_web`:
`uv run --with pandas --with scikit-learn python scripts/prepare_dataset.py`.

## Expected output (verified 2026-09-22)

`gold_test.csv` 58 rows, `train_pool.csv` 4,349 rows, together exactly the
4,407 cache64 study UIDs. Source selection reproduces as:

| source | macro AUC vs gold |
|---|---|
| **stevenleehans_v4blend** (`CHOSEN_SOURCE`) | **0.8927** |
| stevenleehans_v2 | 0.8873 |
| stevenleehans_full | 0.8780 |
| pilkwang_v2 | 0.8700 |
| dreaddevelopment_soft | NaN — no UID overlap with gold; scored but never selectable |

If these numbers drift, an upstream community dataset was revised — do not
silently accept a new `CHOSEN_SOURCE` without re-checking against gold.

## Contract the arm trainers enforce

`load_labels()` in `kaggle/arm*/train_arm*.py` raises unless all of:

- both files carry `StudyInstanceUID` + all 12 label columns, no duplicate UIDs
  (extra columns like `Report`/`label_source` are ignored and harmless);
- **every** gold study resolves in cache64 (`58/58`) — a partial match aborts;
- no NaN/non-finite label for any study it will use. `prepare_dataset.py` only
  *warns* on unlabeled pool studies rather than dropping them, so a source
  change that loses coverage surfaces here as a hard trainer failure;
- gold labels stay strictly binary, and every one of the 12 targets has both
  classes present — `macro_auc()` raises `"Gold split cannot compute AUC for
  targets: ..."` otherwise. Current gold positives range from 9/58 (MCL) to
  35/58 (Effusion), so all 12 are computable.

`train_pool.csv` is 5.7 MB, of which ~4.8 MB is the free-text `Report` column.
The trainers never read it and it must never become a model input (the
competition test set has no `Report`); it is retained only for re-labeling.

## Uploading to Kaggle

Staging metadata lives at `kaggle_datasets/knee-labels/dataset-metadata.json`
(id `zeroday19/rsna-knee-labels-validated`). Per repo convention the CSVs are
not committed — copy them in at upload time:

```bash
cp data/processed/{gold_test,train_pool,label_source_validation}.csv kaggle_datasets/knee-labels/
kaggle datasets create -p kaggle_datasets/knee-labels    # or: version -m "..."
```

The arm trainers locate these by filename anywhere under `/kaggle/input`, so
attaching this dataset alongside the cache/ROI dataset is sufficient — no path
configuration needed.
