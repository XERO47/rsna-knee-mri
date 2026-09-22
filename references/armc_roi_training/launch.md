# Arm C ROI launch and file contract

This is the operational companion to `architecture.md` and
`training_policy.md`. Read it when assembling or running the Kaggle job.

## Required inputs

Attach the cache dataset containing these basenames anywhere below
`/kaggle/input`:

```text
vols_shard0.npy, vols_shard1.npy
ids_shard0.npy, ids_shard1.npy
masks_shard0.npy, masks_shard1.npy
roi_ids.npy
roi_mcl.npy, roi_mcl_masks.npy
roi_lateral_meniscus.npy, roi_lateral_meniscus_masks.npy
roi_lateral_oa.npy, roi_lateral_oa_masks.npy
roi_pf_oa.npy, roi_pf_oa_masks.npy
roi_synovitis.npy, roi_synovitis_masks.npy
```

`roi_meta.json`, `roi_priors.json`, and provenance are useful audit artifacts;
the trainer's required alignment/data path does not consume them. The checker
accepts either `roi_provenance.csv.gz` or the uncompressed legacy
`roi_provenance.csv`.

The trainer searches recursively by basename, so folders such as
`roi_work/cache64/` and `roi_work/cache_merged/` are safe. It requires exact
equality between `roi_ids.npy` and `ids_shard0.npy` followed by
`ids_shard1.npy`; row counts alone are insufficient. The merged arrays must
be `uint8 [N, 4, 3, 336, 336]`, and masks must be binary `uint8 [N, 4]`.

Attach a label dataset containing `gold_test.csv` and `train_pool.csv`. Both
need `StudyInstanceUID` plus the twelve canonical columns listed in
`../dataset/label_pipeline.md`. Do not substitute the competition `train.csv`.

Attach or extract the warm-start checkpoint `arm13_best.pt`. This is Arm13's
recovered best epoch and the validated warm start. Arm13's SWA averaging step
OOMed, so `arm13_swa.pt` is not required or expected.

## Recommended run order

1. Attach cache, labels, and checkpoint datasets.
2. Run `check_roi_cache.py`; inspect the merged-ID PASS and all five targets.
3. Extract the small code bundle if it was uploaded as a ZIP.
4. Run `train_armc_roi_kaggle.py` **as a file**, not as pasted source.

Example:

```python
!unzip -q /kaggle/input/YOUR-CODE-DATASET/arm13_training_bundle.zip \
    -d /kaggle/working/arm_training_bundle
!python /kaggle/working/arm_training_bundle/train_armc_roi_kaggle.py
```

Running it as a file is what makes multi-GPU possible. `start_method="spawn"`
does not clone the parent process; it starts a brand-new interpreter with none
of the parent's memory, and its only way to reach `train_worker` is to load
the file again from disk. Concretely, on a 2-GPU box the module body executes
in four separate processes: the parent (`__name__ == "__main__"`, which runs
`main()` and does nothing else — no CUDA, no data, no W&B), a 3-minute spawn
canary, and the two DDP ranks. In every child `__name__` is `"__mp_main__"`,
so `if __name__ == "__main__": main()` does not fire there and there is no
recursion — the children run straight to the `train_worker(rank, world, port)`
call multiprocessing hands them. This is also why module-level setup
(`NCCL_P2P_DISABLE`, `PYTORCH_NVML_BASED_CUDA_CHECK`) sits above `import torch`
at the very top of the file: it has to apply before each rank's own CUDA/NCCL
initialization, not just the parent's.

If the file was pasted into a notebook cell instead of run from disk, write it
out first and run the saved copy:

```python
%%writefile /kaggle/working/train_armc_roi_kaggle.py
"""Standalone Kaggle trainer for Arm C ROI ...
...paste the entire script here; %%writefile must be the cell's first line...
```

```python
!python -u /kaggle/working/train_armc_roi_kaggle.py
```

(`-u` unbuffers stdout so the log and tqdm bars stream live.) Whatever was
pasted into the `%%writefile` cell becomes what the ranks read — re-run that
cell before re-running training if a config constant changes; editing any
other cell has no effect on the ranks.

The older `exec(open(...).read())` form still works too: the script looks for
its own basename under `/kaggle/working` and `/kaggle/input` and launches from
the copy it finds there (which is why a `%%writefile`'d copy already covers
this path). If no on-disk copy exists it prints why and continues on a single
GPU, which roughly doubles the wall time, rather than failing outright.

Set `WORLD_SIZE = 1` to force the single-process path, for example when
debugging a crash whose traceback is easier to read without rank interleaving.

Only rank 0 talks to W&B — every other rank gets a no-op stand-in
(`NullRun`) — and the parent launcher process never touches W&B at all, so a
2-GPU run produces exactly one W&B run, not two. Its logged metrics are
already reduced across ranks (`train/loss` by mean, Gold AUCs by gathering
every rank's disjoint predictions before scoring), so they describe the whole
batch and the whole Gold split rather than one rank's half. `WANDB_RUN_NAME`
is a fixed string per file edit, so restarting a run under a changed config
(e.g. switching to DDP, which the checkpoint's resume contract refuses to
resume across — see Outputs and recovery) will show up as a second entry
under the same display name with a different run ID; give it a new
`WANDB_RUN_NAME` before relaunching, and stop the old Kaggle session so it
isn't burning GPU quota alongside the new one.

The fixed configuration uses CoAtNet-RMLP-2 at 384 px, freezes the global
path for two epochs, and writes resumable outputs under
`/kaggle/working/armc_roi/`. W&B artifacts are uploaded every two epochs.

For an OOM, edit `ROI_K_TRAIN = 1` first: it drops the per-study image count
from 22 to 17 and costs only window variety, which the epochs recover anyway.
Only then reach for `BATCH_SIZE = 1` with `ACCUM_STEPS = 2`, which preserves
the effective batch of 4. Do not silently change the backbone or resolution.
Expect the OOM at the `roi_only` to `joint_finetune` boundary rather than at
startup — see the throughput section of `training_policy.md` for why a low
peak in the first two epochs means nothing.

## Outputs and recovery

The important outputs are `armc_roi_last.pt` (resume),
`armc_roi_checkpoint_epNNN.pt` (artifact checkpoints), and
`armc_roi_selected_model.pt` (inference). Download the complete W&B artifact
directory when resuming; the selected model alone is not a resumable state.
Resume onto the same number of GPUs the checkpoint was written on: `world_size`
is checked against the saved configuration and a mismatch is refused, because
it would change the OneCycle horizon mid-schedule.

## Separate Arm16 experiment

`kaggle/arm16/train_arm16.py` is a different orientation-canonicalization
experiment. It is not Arm C ROI, does not consume the ROI arrays, and requires
`kaggle/orient/output/orient.npz` in addition to cache64 and labels. Keep its
outputs and conclusions separate from this recipe.
