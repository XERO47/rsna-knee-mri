# Arm C ROI — Arm 13 plus anatomy-specific ROI residuals

Arm C ROI preserves the strongest Arm 13 recipe as its global path:

- CoAtNet-RMLP-2 at 384 px;
- the cache64 wide-span global bag;
- twelve finding-specific global attention queries;
- a required Arm 13 `arm13_best.pt` warm start. The Arm13 SWA averaging step
  OOMed, so there is no separate `arm13_swa.pt` dependency.

Five weak or localization-sensitive findings additionally receive a residual
from the validated local ROI cache: MCL, Lateral Meniscus, Lateral OA, PF OA,
and Synovitis. Each residual affects only its matching logit. New ROI readouts
start at exactly zero, so the initialized combined model is exactly the loaded
Arm 13 model.

The standalone Kaggle entry point is `train_armc_roi_kaggle.py`. It contains the
dataset, model, training, validation, W&B, checkpoint, resume, and final-model
selection code. Run it as a file so spawned DDP ranks can import the same
on-disk module.

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
- batch size 2 per GPU with one step per batch gives effective batch size 4
  on two ranks;
- two visible GPUs are used automatically through one-process-per-GPU DDP;
- the five ROI target bags are packed into one chunked encoder pass rather
  than five small passes; masks, heads, gates, and losses are unchanged;
- data loading stays inside each rank (`WORKERS = 0`) because forking loader
  workers after CUDA/NCCL/W&B initialization can deadlock the first batch;
- validation uses 24 global windows each epoch and 62 for the selected final
  model;
- the best three validation epochs are retained and their weight average is
  selected only when it is at least as good as the best single checkpoint.
- joint fine-tuning uses the already-measured eager checkpoint stages `(0,1)`:
  3.088 seconds/batch and 13.95 GiB reserved on two Kaggle T4s. Launch-time
  probes and compilation have been removed.

## Required Kaggle inputs

1. Cache64: `vols_shard0/1.npy`, `ids_shard0/1.npy`, and
   `masks_shard0/1.npy`.
2. The five merged ROI arrays and masks plus `roi_ids.npy`, or the equivalent
   two-shard ROI format.
3. `train_pool.csv` and `gold_test.csv` with all twelve label columns.
4. `arm13_best.pt`. Arm13's SWA step OOMed, so the recovered best epoch is the
   only validated warm-start checkpoint. The checkpoint
   must contain the CoAtNet-384 `backbone`, `norm`, `att`, `clsW`, and `clsb`
   weights and matching `arch`/`res` metadata when those fields are present.

Run `check_roi_cache.py` before training. The 57 GiB local upload archive does
not currently contain the two label CSVs or an Arm 13 checkpoint; attach those
as an additional small Kaggle dataset.

## Kaggle launch cell

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

```python
!python -u /kaggle/input/YOUR-CODE-DATASET/train_armc_roi_kaggle.py
```

The file finder searches all attached Kaggle inputs, so the checkpoint may be
supplied by basename. Never paste the W&B key into the trainer. Do not change
the default architecture or resolution for the primary run.

Do not change batch size, resolution, windows, or epochs to work around probe
failure. The preflight tries progressively stronger checkpointing and refuses
to launch if no policy has safe T4 headroom; inspect its measured speed and
memory before changing the model recipe.

## Resume

Every two epochs the W&B artifact contains a complete training checkpoint
(model, optimizer, scheduler, AMP scaler, history, RNG state, and config) plus
the current best model-only candidate. Download the whole artifact directory,
not only the full checkpoint, and run:

```python
RESUME = "/kaggle/input/RESUME-DATASET/armc_roi_checkpoint_ep003.pt"
```

Set `RESUME` in the on-disk script's configuration block before launching it;
it takes precedence over `INIT_CHECKPOINT`.

To continue the already-running recipe under the optimized trainer, wait for
the current epoch to finish and copy/upload its complete `armc_roi_last.pt`,
stop the old process, set `RESUME` to that file, and launch this trainer. The
real ranks start immediately with the fixed `(0,1)` policy and restore the
optimizer, scheduler, scaler, history, epoch, and RNG state. Keep the same
two-GPU world size.

## Outputs and interpretation

- `armc_roi_last.pt`: latest resumable state;
- `armc_roi_checkpoint_epNNN.pt`: artifact-uploaded resumable states;
- `armc_roi_candidate_epNNN.pt`: validation-selected model-only candidates;
- `armc_roi_selected_model.pt`: the final inference model selected between the
  best single epoch and top-k average, evaluated at 62 global windows.

W&B records both `gold/macro_auc` (global + ROI) and
`gold/global_macro_auc` (the global branch alone). The difference and the five
per-target `*_roi_gain` metrics show whether local evidence is genuinely
helping rather than whether the shared backbone merely changed. A run should
not replace Arm 13 unless the combined model improves macro AUC, most targeted
findings improve, and the seven untouched findings remain stable.

## Package model and tests

`model.py` provides the reusable architecture. `test_model.py` additionally
compares packed and legacy ROI logits and gradients, including masks and empty
bags, and verifies the encoder-call reduction. `test_ddp_smoke.py` launches
two CPU/Gloo ranks through loader iteration, packed gathered baseline
evaluation, the resume-compatible standard AdamW path, one optimizer step, and an exact
weight-synchronization check. Run in an environment containing the trainer's
dependencies:

```bash
cd kaggle/armc_roi
python -m unittest -v test_model.py test_ddp_smoke.py
```
