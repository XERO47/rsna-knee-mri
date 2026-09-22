# Arm 16 training reference

Last updated: 2026-09-21

This is the durable context for the model in `kaggle/arm16`. Read it together
with `ROI_CACHE_REFERENCE.md` before changing architecture, data loading,
training, validation, checkpointing, or inference.

## Decision and evidence

Arm 13 is the global baseline to preserve. It uses CoAtNet-RMLP-2 at 384 px on
the wide-span cache64 corpus. Its recorded best per-epoch Gold macro AUC is
`0.8909467`. The five findings selected for local ROI evidence average only
`0.81546` on that record, compared with `0.94487` for the seven remaining
findings:

| ROI target | Arm 13 recorded AUC |
|---|---:|
| MCL | 0.9410 |
| Lateral Meniscus | 0.8037 |
| Lateral OA | 0.7389 |
| PF OA | 0.8314 |
| Synovitis | 0.7622 |

This supports local anatomical evidence, but it does not support replacing the
global backbone. The former Arm 16 default—ConvNeXt-small at 336 px—was an Arm
12/15 derivative rather than a strict Arm 13 extension. The primary recipe is
therefore now **Arm 13 plus bounded target-specific ROI residuals**.

The local Gold set contains only 58 studies, so small AUC movements are noisy.
Arm 12's recorded top-three average (`0.89333`) and Arm 13's recorded best
epoch (`0.89095`) are effectively close on this set. Leaderboard performance,
paired predictions, repeated seeds, and blending remain necessary evidence.

## Architecture contract

Global path:

```text
cache64 windows -> shared CoAtNet-RMLP-2 @384 -> LayerNorm
                -> 12 query-attention pools -> 12 logits
```

Local path for MCL, Lateral Meniscus, Lateral OA, PF OA, and Synovitis:

```text
four anatomy-specific cached windows
    -> same CoAtNet encoder
    -> target-specific gated-attention pool
    -> scalar residual
    -> sigmoid(beta) * residual added only to matching global logit
```

The ROI readout weights and biases initialize to zero. Consequently, after an
Arm 13 checkpoint is loaded, the global+ROI logits are bit-identical to the
global logits before training. The beta parameters initialize at `-2`, but a
nonzero gate cannot change predictions while the residual readout is zero.

Missing windows remain masked and contribute exactly zero. ROI evidence can
never directly change a different target's logit.

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
| Batch / accumulation | 2 / 2 | Effective batch 4; automatically uses two GPUs when present |
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
are meaningful, and the ROI cache deliberately preserves physical orientation.

## Data contract

The trainer requires:

- cache64 global arrays, IDs, and masks for both original shards;
- all five ROI arrays and masks, either merged or sharded;
- ROI IDs exactly equal to cache64 IDs (merged order is shard 0 then shard 1);
- `train_pool.csv` and `gold_test.csv` with the twelve canonical columns;
- an Arm 13 `arm13_swa.pt` or `arm13_best.pt` checkpoint.

The merged ROI arrays remain stored at 336 px and are deterministically resized
to 384 px by the dataset because the shared Arm 13 encoder requires 384. The
physical crop content is unchanged. The cache builder, synovitis 25 mm policy,
coverage results, provenance, and merge evidence are documented in
`ROI_CACHE_REFERENCE.md`.

The local `roi_work/armc_kaggle_upload.zip` contains cache64, merged ROI cache,
the cache checker, and the standalone trainer. It does not contain the two
label CSVs or an Arm 13 checkpoint. Those must be attached separately.

## Validation and selection contract

Before epoch 0 the trainer evaluates the loaded model and requires the
combined macro AUC to equal its global-only macro AUC. This detects accidental
nonzero ROI initialization.

Every epoch logs:

- combined and global-only macro AUC;
- combined per-target AUC;
- combined-minus-global ROI gain for each of the five local targets;
- ROI gate values;
- phase, learning rates, optimizer-step count, loss, and elapsed time.

Epoch selection uses the same deterministic 24-window Gold evaluation for all
checkpoints. The best three candidate weights are retained. At the end, the
top-three average is compared with the best single candidate on that same
protocol; the better one becomes `arm16_selected_model.pt`. The selected model
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
successful Arm 16 rather than discarding Arm 13.

## Checkpoint and recovery contract

`arm16_last.pt` and every uploaded `arm16_checkpoint_epNNN.pt` contain model,
optimizer, scheduler, AMP scaler, history, RNG states, configuration, and last
metrics. W&B artifacts also include the current best candidate. Download the
entire artifact directory for recovery so historical top-k selection remains
available. `ARM16_RESUME` takes precedence over `ARM16_INIT_CHECKPOINT`.

The final artifact contains:

- the final resumable state (which remains the last training state);
- all surviving top-k model-only candidates;
- `arm16_selected_model.pt`, the inference checkpoint.

Keeping the resumable state separate from the selected inference weights is
intentional: optimizer state belongs to the last chronological training state,
not necessarily to a validation-selected or averaged model.

## File and symbol map

### `kaggle/arm16/train_arm16_kaggle.py`

Self-contained Kaggle program.

| Symbol | Input | Output/responsibility |
|---|---|---|
| `find_file`, `require_file` | basename | safely locate attached files without walking raw DICOM trees |
| `Corpus` | cache64 arrays | memory-mapped global study slices |
| `load_index` | cache IDs/masks | unique canonical ID order and masks |
| `load_labels` | canonical IDs | disjoint training/Gold IDs and 12-label vectors |
| `OptionalROICache` | ROI arrays, masks, IDs | exact-alignment validation and memory-mapped target bags; all targets are required in the primary run |
| `StudyDataset` | IDs, masks, cache, window counts | normalized global/ROI bags; stochastic train sampling and deterministic evaluation |
| `QueryMIL` | global features/mask | twelve global logits |
| `ROIPooler` | one target's local features/mask | one zero-initialized scalar residual |
| `ArmC` | global and local bags | combined logits and optional global-only logits |
| `evaluate` | model/Gold loader | combined and global macro/per-target AUCs |
| `load_global_initialization` | model/Arm 13 checkpoint | validated exact global weight mapping |
| `set_training_phase` | model/epoch | ROI-only or joint trainability policy |
| `checkpoint_payload`, `restore_rng_state` | run state | complete interruption-safe save/restore |
| `save_top_candidate`, `average_candidates` | validation history/model states | bounded top-k model selection |
| `log_checkpoint` | W&B run/payload | complete artifact plus selected model files |
| `main` | environment configuration | end-to-end train, validate, select, and upload workflow |

### `kaggle/arm16/model.py`

Reusable architecture with validation, masked attention, optional diagnostic
outputs, chunked encoding, and Arm 13 checkpoint mapping. It is not imported by
the standalone Kaggle trainer; keep shared behavioral contracts synchronized.

### `kaggle/arm16/test_model.py`

CPU unit tests for tensor shapes, target isolation, empty/missing ROI no-ops,
exact fresh-ROI no-op behavior, global checkpoint mapping, masking, and loss
utilities.

### `kaggle/arm16/check_roi_cache.py`

Read-only Kaggle preflight for either merged or sharded ROI layouts. Run it
before allocating a GPU session to training.

## Intentional non-default experiments

Scratch training is an ablation, not the primary run. It requires both
`ARM16_ALLOW_SCRATCH=1` and `ARM16_PRETRAINED=1`. Switching to ConvNeXt-336 is
also a separate Arm 12/15-based experiment. Neither result should be labeled a
strict Arm 13 upgrade.
