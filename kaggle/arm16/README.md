# Arm 16 — Arm 13 plus anatomy-specific ROI residuals

Arm 16 now preserves the strongest Arm 13 recipe as its global path:

- CoAtNet-RMLP-2 at 384 px;
- the cache64 wide-span global bag;
- twelve finding-specific global attention queries;
- a required Arm 13 best/SWA checkpoint warm start.

Five weak or localization-sensitive findings additionally receive a residual
from the validated local ROI cache: MCL, Lateral Meniscus, Lateral OA, PF OA,
and Synovitis. Each residual affects only its matching logit. New ROI readouts
start at exactly zero, so the initialized combined model is exactly the loaded
Arm 13 model.

The standalone Kaggle entry point is `train_arm16_kaggle.py`. It contains the
dataset, model, training, validation, W&B, checkpoint, resume, and final-model
selection code and can be executed as one notebook cell.

## Default high-probability recipe

The defaults are intentionally conservative:

- all five ROI caches are required and IDs must exactly match cache64;
- an Arm 13 checkpoint is required unless scratch mode is explicitly enabled;
- epochs 0–1 freeze Arm 13 and train only ROI heads/gates;
- epochs 2–9 jointly fine-tune with learning rates `1e-5` backbone, `3e-5`
  global head, and `3e-4` ROI heads;
- the joint phase adds an auxiliary global loss on the five ROI targets to
  resist degradation of the proven global signal;
- training samples two of four valid windows per ROI target; validation uses
  all four;
- batch size 2 with two-step accumulation gives effective batch size 4;
- two visible GPUs are used automatically through `DataParallel`;
- validation uses 24 global windows each epoch and 62 for the selected final
  model;
- the best three validation epochs are retained and their weight average is
  selected only when it is at least as good as the best single checkpoint.

## Required Kaggle inputs

1. Cache64: `vols_shard0/1.npy`, `ids_shard0/1.npy`, and
   `masks_shard0/1.npy`.
2. The five merged ROI arrays and masks plus `roi_ids.npy`, or the equivalent
   two-shard ROI format.
3. `train_pool.csv` and `gold_test.csv` with all twelve label columns.
4. `arm13_swa.pt` when it exists; otherwise use `arm13_best.pt`. The checkpoint
   must contain the CoAtNet-384 `backbone`, `norm`, `att`, `clsW`, and `clsb`
   weights and matching `arch`/`res` metadata when those fields are present.

Run `check_roi_cache.py` before training. The 57 GiB local upload archive does
not currently contain the two label CSVs or an Arm 13 checkpoint; attach those
as an additional small Kaggle dataset.

## Kaggle launch cell

```python
import os

os.environ["ARM16_INIT_CHECKPOINT"] = "arm13_swa.pt"
os.environ["WANDB_PROJECT"] = "rsna-knee-arm16"
os.environ["WANDB_RUN_NAME"] = "arm16-arm13-roi-seed42"

exec(open("/kaggle/input/YOUR-CODE-DATASET/train_arm16_kaggle.py").read())
```

The file finder searches all attached Kaggle inputs, so the checkpoint may be
supplied by basename. Set the W&B API key through Kaggle Secrets. Do not change
the default architecture or resolution for the primary run.

If memory is insufficient on the selected accelerator, first set
`ARM16_BATCH_SIZE=1` and `ARM16_ACCUM_STEPS=4`. This preserves the effective
batch size. Do not reduce resolution or silently switch the backbone.

## Resume

Every two epochs the W&B artifact contains a complete training checkpoint
(model, optimizer, scheduler, AMP scaler, history, RNG state, and config) plus
the current best model-only candidate. Download the whole artifact directory,
not only the full checkpoint, and run:

```python
os.environ["ARM16_RESUME"] = "/kaggle/input/RESUME-DATASET/arm16_checkpoint_ep003.pt"
exec(open("/kaggle/input/YOUR-CODE-DATASET/train_arm16_kaggle.py").read())
```

`ARM16_RESUME` takes precedence over `ARM16_INIT_CHECKPOINT`.

## Outputs and interpretation

- `arm16_last.pt`: latest resumable state;
- `arm16_checkpoint_epNNN.pt`: artifact-uploaded resumable states;
- `arm16_candidate_epNNN.pt`: validation-selected model-only candidates;
- `arm16_selected_model.pt`: the final inference model selected between the
  best single epoch and top-k average, evaluated at 62 global windows.

W&B records both `gold/macro_auc` (global + ROI) and
`gold/global_macro_auc` (the global branch alone). The difference and the five
per-target `*_roi_gain` metrics show whether local evidence is genuinely
helping rather than whether the shared backbone merely changed. A run should
not replace Arm 13 unless the combined model improves macro AUC, most targeted
findings improve, and the seven untouched findings remain stable.

## Package model and tests

`model.py` provides the reusable architecture and `test_model.py` covers
target isolation, missing-ROI no-ops, checkpoint mapping, and exact zero-impact
ROI initialization. Run in an environment containing PyTorch:

```bash
cd kaggle/arm16
python -m unittest -v test_model.py
```
