# Arm C ROI architecture contract

Source: the archived Arm16 training reference (decision and evidence, architecture
contract, file/symbol map). Last updated 2026-09-21. Read together with
`../roi_cache/laterality_rules.md` before changing architecture, data
loading, training, validation, checkpointing, or inference.

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
global backbone. The former Arm C ROI default — ConvNeXt-small at 336 px — was an
Arm 12/15 derivative rather than a strict Arm 13 extension. The primary
recipe is therefore now **Arm 13 plus bounded target-specific ROI residuals**.

The local Gold set contains only 58 studies, so small AUC movements are noisy.
Arm 12's recorded top-three average (`0.89333`) and Arm 13's recorded best
epoch (`0.89095`) are effectively close on this set. Leaderboard performance,
paired predictions, repeated seeds, and blending remain necessary evidence.

## Global path

```text
cache64 windows -> shared CoAtNet-RMLP-2 @384 -> LayerNorm
                -> 12 query-attention pools -> 12 logits
```

## Local path

For MCL, Lateral Meniscus, Lateral OA, PF OA, and Synovitis:

```text
four anatomy-specific cached windows
    -> pack the five fixed-shape target bags
    -> one chunked call to the same CoAtNet encoder
    -> split features back by target
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

## Data contract

The trainer requires:

- cache64 global arrays, IDs, and masks for both original shards;
- all five ROI arrays and masks, either merged or sharded;
- ROI IDs exactly equal to cache64 IDs (merged order is shard 0 then shard 1);
- `train_pool.csv` and `gold_test.csv` with the twelve canonical columns;
- the recovered Arm 13 `arm13_best.pt` checkpoint. The Arm13 SWA step OOMed;
  do not require a nonexistent `arm13_swa.pt` file.

The merged ROI arrays remain stored at 336 px and are deterministically
resized to 384 px by the dataset because the shared Arm 13 encoder requires
384. The physical crop content is unchanged. The cache builder, synovitis
25 mm policy, coverage results, provenance, and merge evidence are documented
in `../roi_cache/laterality_rules.md`.

The legacy `roi_work/armc_kaggle_upload.zip` contains cache64 and the merged
ROI cache, but its embedded code copy predates the Arm C rename/checkpoint-name
cleanup. Use the current small `arm13_training_bundle.zip` for code plus
`arm13_best.pt`, and attach the two label CSVs separately. The large cache
archive itself does not need to be rebuilt.

## File and symbol map

### `kaggle/armc_roi/train_armc_roi_kaggle.py`

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
| `ArmC.encode_roi_bags` | five fixed-shape ROI bags | one packed/chunked encoder pass, split back into target feature bags without changing masks or residual heads |
| `evaluate` | model/Gold loader | combined and global macro/per-target AUCs |
| `load_global_initialization` | model/Arm 13 checkpoint | validated exact global weight mapping |
| `set_training_phase` | model/epoch | ROI-only or joint trainability policy |
| `CHECKPOINT_STAGES` | measured two-T4 policy | fixed `(0,1)` stage checkpointing established by the completed joint-finetune probes; no launch-time reprobe |
| `checkpoint_payload`, `restore_rng_state` | run state | complete interruption-safe save/restore |
| `save_top_candidate`, `average_candidates` | validation history/model states | bounded top-k model selection |
| `log_checkpoint` | W&B run/payload | complete artifact plus selected model files |
| `train_worker` | rank, world size, TCP port | the actual end-to-end train/validate/select/upload workflow; runs once per DDP rank |
| `main` | environment configuration | launcher only: counts GPUs, resolves which on-disk copy of this file to spawn, then hands off to `train_worker` via `run_children`; touches no CUDA, data, or W&B itself |
| `resolve_launch_module`, `find_all_files`, `is_ddp_capable` | `SELF_BASENAME`/`SELF_PATH` | find every on-disk copy of this file and pick the first one that actually defines `train_worker`, skipping stale pre-DDP uploads by content, not just by name |
| `run_children`, `spawn_canary` | target fn, rank count | `mp.start_processes` with a hard deadline; the canary proves child processes can start at all before the real ranks run for hours |
| `wrap_for_training` | model, rank, world | builds a fresh `DistributedDataParallel` wrapper at each phase boundary, `static_graph=True` |
| `gather_rows`, `reduce_mean`, `barrier`, `NullRun` | rank-local tensors/values | cross-rank aggregation primitives; `NullRun` is the no-op W&B stand-in on every rank but 0 |
| `heartbeat`, `start_stall_watchdog`, `make_loader_kwargs` | — | phase-specific stall detection plus the zero-worker loader policy that avoids forking a CUDA/NCCL/W&B rank at first iteration |

See `launch.md`'s "DDP: what went wrong and how it was fixed" for the
reasoning and incident history behind the DDP-specific symbols above.

### `kaggle/armc_roi/model.py`

Reusable architecture with validation, masked attention, optional diagnostic
outputs, packed ROI encoding, chunked execution, and Arm 13 checkpoint mapping. It is not imported
by the standalone Kaggle trainer; keep shared behavioral contracts
synchronized.

### `kaggle/armc_roi/test_model.py`

CPU unit tests for tensor shapes, target isolation, empty/missing ROI no-ops,
exact fresh-ROI no-op behavior, packed-vs-legacy logits and gradients, encoder
call count, global checkpoint mapping, masking, and loss utilities.

### `kaggle/armc_roi/check_roi_cache.py`

Read-only Kaggle preflight for either merged or sharded ROI layouts. Run it
before allocating a GPU session to training.

## Intentional non-default experiments

Scratch training is an ablation, not the primary run. It requires both
the fixed `ALLOW_SCRATCH = True` and `PRETRAINED = True` configuration values.
Switching to ConvNeXt-336 is
also a separate Arm 12/15-based experiment. Neither result should be labeled a
strict Arm 13 upgrade.
