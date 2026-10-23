# Arm C ROI training policy

Source: the archived Arm16 training reference (training policy and reasoning,
validation and selection contract, checkpoint and recovery contract). Read
together with `architecture.md` in this folder.

## Training policy and reasoning

The primary run uses these defaults:

| Setting | Default | Reason |
|---|---:|---|
| Backbone/resolution | CoAtNet-RMLP-2 / 384 | Exact Arm 13 global model |
| Arm 13 warm start | required | Avoid relearning the strongest global solution |
| Epochs | 10 | Match the proven training horizon |
| Frozen global epochs | 2 | Let random ROI heads find signal without damaging Arm 13 |
| Global train/eval/final windows | 12 / 24 / 62 | Cheap training, stable epoch scoring, strongest final inference |
| ROI train/eval windows | 2 / all 4 | Fit Kaggle memory while exposing all cached views over epochs |
| Batch / accumulation | 2 per GPU / 1 | Effective batch 4 on two ranks, which is what the LRs were set for |
| Parallelism | DDP, one rank per GPU | Two T4s halve the epoch; see "Throughput" below |
| Backbone LR | `1e-5` | Conservative shared-feature adaptation |
| Global-head LR | `3e-5` | Preserve established attention/classification weights |
| ROI-head LR | `3e-4` | New modules need substantially faster learning |
| Auxiliary global loss | 0.25 on ROI targets | Discourage the shared encoder/global branch from forgetting |
| Checkpoint cadence | every 2 epochs | Full resumability without per-epoch upload overhead |
| Model selection | best single vs top-3 average | Reuse Arm 13's variance-reduction idea |

During the first two epochs, loss is computed only over the five ROI targets
because every other trainable parameter is frozen. During joint fine-tuning,
the primary BCE covers all twelve combined logits and an additional weighted
BCE protects the five global-only logits. The seven non-ROI findings already
receive their normal loss through unchanged combined logits.

No horizontal flipping is performed. Left/right identity and anatomical side
are meaningful, and the ROI cache deliberately preserves physical orientation
(see `../roi_cache/laterality_rules.md`).

## Throughput

An Arm C ROI epoch is intrinsically heavier than an Arm 13 one. Arm 13 encoded
12 global windows per study; this arm encodes 12 global plus five targets times
`ROI_K_TRAIN` windows, so 22 images per study at `ROI_K_TRAIN = 2` — about
1.8x the forward work per study before any other difference. A single-GPU Arm C
run against a two-GPU Arm 13 run is therefore roughly 3.7x slower per epoch,
which is arithmetic, not a fault to hunt for.

The optimized implementation recovers throughput without changing the recipe:

- **DDP, one rank per GPU.** `nn.DataParallel` is not an option here:
  `torch.autocast` state is thread-local and would not reach the replicas
  running in side threads, so the 384 px CoAtNet would silently fall back to
  fp32 and exhaust the card. Ranks are spawned, never forked.
- **uint8 to the GPU.** Datasets emit windows at cache resolution as uint8;
  `ArmC.prepare` scales, normalises, and resizes to `RES` on-device. Doing it
  in the loader cost four times the PCIe traffic and put interpolation on a
  four-vCPU box that also has to feed two ranks.
- **Pack all five ROI targets.** The global bag remains a separate encoder
  call. The five `[B,2,3,336,336]` ROI bags are concatenated to
  `[B,10,3,336,336]`, encoded in chunks of at most 24 images, then split back
  before the unchanged target-specific masks, poolers, gates, and residuals.
  Masked zero windows are deliberately encoded: eliminating five
  tensor-to-Python `valid.any()` synchronizations and five tiny backbone calls
  is faster at the cache's high coverage.
- **Avoid hot-loop host synchronization.** Loss and gradient diagnostics stay
  as GPU tensors and are converted to Python only at the configured logging
  cadence and epoch end. CUDA events profile data wait, transfer,
  forward/backward, and optimizer time every 100 batches.
- **Cheaper DDP plumbing with resume-safe optimization.** DDP uses gradient
  bucket views (and batched gradient copies when the installed PyTorch exposes
  them), and gradient clipping uses the foreach implementation. AdamW remains
  on the standard implementation used to create the existing checkpoints;
  changing implementation while loading their groups breaks AMP stepping.
- **Use the measured T4 policy.** A completed two-rank joint-finetune probe
  established eager checkpoint stages `(0,1)` at 3.088 seconds/batch and
  13.95 GiB reserved. No checkpointing and `(0,)` OOMed. Compilation reached
  2.519 seconds/batch but required nearly twelve minutes to compile and rose
  to 14.13 GiB, above the safety ceiling, so production uses eager `(0,1)`
  directly and does not repeat the probes at launch.

Peak memory during `roi_only` is therefore not a guide to what the run needs:
those epochs store no backbone activations and use a small fraction of the
card. The binding constraint is the joint phase. The measured `(0,1)` policy
is fixed in checkpoint config and W&B summary; watch `system/gpu_peak_gb`
across the phase boundary as a production cross-check.

## Validation and selection contract

Before epoch 0 the trainer evaluates the loaded model and requires the
combined macro AUC to equal its global-only macro AUC. This detects
accidental nonzero ROI initialization.

Every epoch logs:

- combined and global-only macro AUC;
- combined per-target AUC;
- combined-minus-global ROI gain for each of the five local targets;
- ROI gate values;
- phase, learning rates, optimizer-step count, loss, and elapsed time;
- peak GPU memory (`system/gpu_peak_gb`) and the epoch's last clipped
  gradient norm (`train/grad_norm_last`).

Between epoch rows the trainer also emits a `step/*` series on its own
`step/global_step` x-axis every `LOG_EVERY_STEPS` optimizer steps: windowed
loss, gradient norm, AMP loss scale, and ROI-head LR. These are diagnostics
for watching a run in flight, not part of the selection contract. tqdm bars
(`PROGRESS`) cover the training pass and every evaluation; both passes are
otherwise silent for long stretches, which has previously been mistaken for a
hang.

Every 100 production batches, the same series receives low-overhead
`data_wait_ms`, `transfer_ms`, `forward_backward_ms`, `optimizer_ms`, and
`total_profiled_ms` CUDA-event timings. They identify the remaining bottleneck
without synchronizing every training batch.

Under DDP each rank scores a disjoint stride of Gold and the predictions are
all-gathered in rank order before any AUC is computed, so every rank holds the
identical matrix and reaches the same selection verdict without broadcasting a
decision. Only rank 0 prints, draws progress bars, writes checkpoints, or
talks to W&B.

Epoch selection uses the same deterministic 24-window Gold evaluation for all
checkpoints. The best three candidate weights are retained. At the end, the
top-three average is compared with the best single candidate on that same
protocol; the better one becomes `armc_roi_selected_model.pt`. The selected model
is then reported once with 62 global windows and all four ROI windows.

Do not claim success from macro AUC alone. A primary run is a credible Arm 13
upgrade only when:

1. combined macro AUC exceeds the warm-start baseline;
2. at least three of the five ROI targets improve and the gains make clinical
   sense;
3. the seven non-ROI findings do not show a broad regression;
4. the result is reproduced with at least one additional seed or supported by
   leaderboard behavior;
5. Arm 13 remains available for a rank/probability blend.

The highest-probability final submission is likely a blend of Arm 13 and a
successful Arm C ROI run rather than discarding Arm 13.

## Checkpoint and recovery contract

`armc_roi_last.pt` and every uploaded `armc_roi_checkpoint_epNNN.pt` contain model,
optimizer, scheduler, AMP scaler, history, RNG states, configuration, and last
metrics. `world_size` is part of the resume contract: the number of ranks sets
how long the per-rank loader is and therefore the whole OneCycle horizon, so a
checkpoint cannot be resumed onto a different GPU count. W&B artifacts also include the current best candidate. Download the
entire artifact directory for recovery so historical top-k selection remains
available. The fixed `RESUME` configuration takes precedence over `INIT_CHECKPOINT`.

The final artifact contains:

- the final resumable state (which remains the last training state);
- all surviving top-k model-only candidates;
- `armc_roi_selected_model.pt`, the inference checkpoint.

Keeping the resumable state separate from the selected inference weights is
intentional: optimizer state belongs to the last chronological training
state, not necessarily to a validation-selected or averaged model.

The throughput fields added to configuration are intentionally outside the
resume compatibility tuple. An existing same-recipe, same-world-size
`armc_roi_last.pt` can therefore resume under packed execution and the fixed
checkpoint policy without changing the optimizer, scheduler, epoch,
data split, or parameter names.
