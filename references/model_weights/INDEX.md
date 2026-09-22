# Model weights — AI routing guide

Read this when a task needs to load, warm-start from, or reference a trained
checkpoint (`.pt`) from any `kaggle/arm*` or `kaggle/knee_dino` run. All
checkpoints are too large for git. They may be supplied by the
[`model-weights`](https://github.com/XERO47/rsna-knee-mri/releases/tag/model-weights)
GitHub release or by a local ignored copy under `models/`; the current Arm C
warm start is `models/arm13_best.pt`.

## Asset → producing script

Each asset's `arm<N>` prefix maps to `kaggle/arm<N>/train_arm<N>.py` (read
that file and its sibling `kernel-metadata.json` for exact architecture,
hyperparameters, and what the checkpoint suffix means for that arm — e.g.
`_swa`, `_best`, `_last`, `_topk_ep<N>`, or a fold/head tag like `_f0_attn`).

| Asset | Size | Produced by |
|---|---|---|
| `arm1_best.pt`, `arm1_final.pt` | 45 MB | `kaggle/arm1/train_arm1.py` |
| `arm2_scratch_best.pt`, `arm2_ssl_best.pt` | 86 MB | `kaggle/arm2/train_arm2.py` |
| `arm6_best.pt` | 97 MB | `kaggle/arm6/train_arm6.py` |
| `arm13_best.pt` | 293 MB | `kaggle/arm13/train_arm13.py` — this IS the usable SWA-equivalent checkpoint; the run's weight-averaging step OOM'd, so `arm13_best.pt` (the best single epoch) is what warm-starts later arms, not a separate `_swa` file |
| `arm14_f0_attn_swa.pt`, `arm14_f0_seq_swa.pt` | ~200 MB each | `kaggle/arm14/train_arm14.py` (fold 0, attention vs. sequential pooling head variants) |
| `arm15_best.pt`, `arm15_last.pt`, `arm15_swa.pt`, `arm15_topk_ep{6,7,9}.pt` | 199–596 MB | `kaggle/arm15/train_arm15.py` |
| `arm16_f0_base_swa.pt`, `arm16_f0_canon_swa.pt` | ~199 MB each | `kaggle/arm16/train_arm16.py` (the original Kaggle-kernel version; `train_arm16_kaggle.py` is the standalone cell rewrite of the same recipe) |
| `arm18_f0_base_swa.pt`, `arm18_f0_mask_swa.pt` | 199 MB each | `kaggle/arm18/train_arm18.py` |
| `arm19_f0_aug_swa.pt`, `arm19_f0_noaug_swa.pt` | 199 MB each | `kaggle/arm19/train_arm19.py` |
| `arm20_f0_aug_swa.pt`, `arm20_f0_noaug_swa.pt` | 199 MB each | `kaggle/arm20/train_arm20.py` |
| `knee_dino_checkpoint_v3_collapsed.pt`, `_v5_easy_evalbn_collapsed.pt`, `_v6_easy_trainbn_stable.pt`, `_v7_curriculum.pt` | 182 MB each | `kaggle/knee_dino/train_knee_dino.py` (SSL pretraining, see `references/archive` / COORDINATION.md for the collapse/curriculum history behind each variant name) |

Files with identical content across `output/`, `upload/`, and
`kaggle_datasets/` copies were deduplicated before upload — the release has
one copy per unique checkpoint, not one per arm run.

## Downloading an asset

```bash
gh release download model-weights --repo XERO47/rsna-knee-mri \
  --pattern 'arm13_best.pt' --dir models/
# or without gh:
curl -L -o models/arm13_best.pt \
  https://github.com/XERO47/rsna-knee-mri/releases/download/model-weights/arm13_best.pt
```

For a Kaggle run, upload the needed `.pt` file(s) as a Kaggle dataset (or
attach directly if already uploaded) rather than downloading the whole
release — a single checkpoint is 45 MB–596 MB; the full release is ~5 GB.

## What this release does NOT contain

No label files (`gold_test.csv`, `train_pool.csv`) — those are built locally
by `python scripts/prepare_dataset.py` (gitignored under `data/processed/`,
see repo root `README.md`) and must be uploaded to Kaggle as a separate
dataset input; they are not part of `model-weights`.
