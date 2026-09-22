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
are meaningful, and the ROI cache deliberately preserves physical orientation
(see `../roi_cache/laterality_rules.md`).

## Validation and selection contract

Before epoch 0 the trainer evaluates the loaded model and requires the
combined macro AUC to equal its global-only macro AUC. This detects
accidental nonzero ROI initialization.

Every epoch logs:

- combined and global-only macro AUC;
- combined per-target AUC;
- combined-minus-global ROI gain for each of the five local targets;
- ROI gate values;
- phase, learning rates, optimizer-step count, loss, and elapsed time.

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
metrics. W&B artifacts also include the current best candidate. Download the
entire artifact directory for recovery so historical top-k selection remains
available. The fixed `RESUME` configuration takes precedence over `INIT_CHECKPOINT`.

The final artifact contains:

- the final resumable state (which remains the last training state);
- all surviving top-k model-only candidates;
- `armc_roi_selected_model.pt`, the inference checkpoint.

Keeping the resumable state separate from the selected inference weights is
intentional: optimizer state belongs to the last chronological training
state, not necessarily to a validation-selected or averaged model.
