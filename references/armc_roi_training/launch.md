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
of the parent's memory, and its only way to reach a worker function is to load
the file again from disk. On a 2-GPU box this includes the parent
(`__name__ == "__main__"`, which runs `main()` and does nothing else — no
model, data, or W&B), a 3-minute spawn canary, and the two training ranks. In
every child `__name__` is `"__mp_main__"`, so `if __name__ == "__main__":
main()` does not fire there and there is no recursion — multiprocessing calls
`train_worker(...)` directly. This is also why module-level setup
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
its own basename under `/kaggle/working` first, then `/kaggle/input`, and
launches from the first copy that defines `train_worker` (which is why a
`%%writefile`'d copy already covers this path). Kaggle inputs routinely hold
older uploads of the same file — e.g. a pre-DDP copy inside
`arm13_training_bundle/` — so copies without `train_worker` are skipped by
content, never imported, and named in the log as ignored. The log line
`ranks will load <path>` says which copy the ranks actually run. To remove
any doubt, set `SELF_PATH` to the exact file. If no DDP-capable copy exists it
lists every copy it considered and continues on a single GPU, which roughly
doubles the wall time, rather than failing outright.

Set `WORLD_SIZE = 1` to force the single-process path, for example when
debugging a crash whose traceback is easier to read without rank interleaving.

Healthy DDP startup prints `ranks will load <path>`, `spawn canary passed;
using proven eager checkpoint stages (0, 1)`, `launching 2 DDP ranks`,
`process group up (nccl); testing a collective`, then `collective ok: all
ranks connected`. If the log stops between the last two, the NCCL
transport itself is broken; nothing else has run yet. Each rank binds its GPU
before the process group exists and passes it as `device_id`; without that,
NCCL guesses the device lazily on the first collective, and two ranks landing
on GPU 0 hang the first all-gather indefinitely.

Only rank 0 draws progress bars, so a frozen bar cannot by itself distinguish
a blocked input read, a slow forward, a hung rank 1, or a collective waiting
on a dead peer. Kaggle's `!` output can also keep showing a cleared
(`leave=False`) bar long after it finished. After `STALL_DUMP_SECONDS`
(default 300) without progress, every rank prints `[stall] rank N ... during
'<phase>'` followed by every thread's stack. The phase names distinguish
waiting for a loader batch, transferring it, the GPU forward/backward, and
prediction gathering. A stack ending in `all_gather_object`/`barrier` means it
is waiting on the other rank. A stall labelled `barrier: waiting for rank 0
to finish checkpoint/W&B work` during an upload is expected, not a hang.

Do not write the W&B key into the file; every copy of the file (Kaggle
datasets included) would carry it. Store it as a Kaggle Secret and export it
before launching; the spawned ranks inherit the environment:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

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

The performance probes have already been completed on two Kaggle T4s. Eager
checkpoint stages `(0,1)` completed at 3.088 seconds/batch and 13.95 GiB
reserved; no checkpointing and `(0,)` OOMed. The compiled candidate reached
2.519 seconds/batch but consumed 14.13 GiB and spent nearly twelve minutes
compiling, so it was rejected. Launch now uses eager `(0,1)` directly while
preserving batch size 2 per rank, effective batch 4, all windows and epochs,
losses, learning rates, and selection policy.

## DDP: what went wrong and how it was fixed

This section is a chronological incident record for future agents debugging
a stuck or failing DDP run. Read it before re-deriving any of this from the
traceback; each entry is symptom -> cause -> fix, in the order these were hit
converting the trainer from single-GPU/`nn.DataParallel` to multi-GPU DDP.

### `nn.DataParallel` crash and its two latent hazards

**Symptom:** `TypeError: ArmC.forward() missing 2 required positional
arguments` from a DataParallel replica thread, batch size below device count.
**Cause:** `forward` originally took a non-tensor `return_global: bool`
kwarg. DataParallel replicates a scalar kwarg to every device but only
scatters tensor inputs into as many chunks as the batch size allows; with
batch 1 on 2 GPUs, the second replica got the replicated kwarg and zero
positional arguments. **Fix:** removed the kwarg; `forward` always returns
`(combined_logits, global_logits)`. **Do not reintroduce `DataParallel`** even
once this crash is fixed: `torch.autocast` state is thread-local and does not
reach DataParallel's side-thread replicas, so fp16 training would silently
fall back to fp32 there and OOM the 384 px CoAtNet; gradient checkpointing in
those side threads is likewise unreliable. Use DDP, one rank per GPU, always.

### Stale-copy resolution: DDP silently downgrading to one GPU

**Symptom:** log says `N GPUs visible but train_armc_roi_kaggle.py was not
found on disk`, even though the file was uploaded and is visible in the
Kaggle input file list. **Cause:** Kaggle inputs commonly hold more than one
copy of this file — an old upload inside `arm13_training_bundle/`, a new one
in a separate dataset. The original file search returned the *first* basename
match by walk order, which was the pre-DDP copy with no `train_worker`.
**Fix:** `resolve_launch_module` now collects every copy (`find_all_files`),
filters to ones whose *text* defines `train_worker` (`is_ddp_capable` — text,
not import, so a stale copy's top-level code never executes), and uses the
first survivor. The log always says which file was picked
(`ranks will load <path>`) and names every copy it skipped. If still in
doubt, set `SELF_PATH` to the exact file. **A second bug found while fixing
this:** when the script is `exec()`-ed, `train_worker` ends up defined inside
the notebook kernel's own `__main__`, and the original check trusted any
`__main__` with a real `__file__`. That would have let ranks be spawned from
the kernel launcher itself. Fixed by checking `__main__`'s file content the
same way (`is_ddp_capable`), not just its existence.

### NCCL device binding order: startup hang at the first collective

**Symptom:** log stops cleanly after `spawn canary passed; launching N DDP
ranks`, before any training output, for many minutes — indistinguishable at a
glance from a slow warm-start checkpoint load. **Cause:** `dist.init_process_group`
was called before `torch.cuda.set_device(rank)`. NCCL then builds its
communicator lazily on the first real collective and has to guess which GPU
each rank owns; if both ranks guess GPU 0 (a common failure on a fresh
runtime), that first collective — inside the training loop, e.g. the
predictions `all_gather_object` in `evaluate` — hangs indefinitely with no
error. **Fix:** bind `torch.cuda.set_device(rank)` *before* creating the
process group, and pass `device_id=device` into `init_process_group` so the
communicator is built eagerly at group-init time, not lazily at first use.
Immediately after group init, do one throwaway `dist.all_reduce` and check its
result equals `world` — this turns a silent multi-hour hang into an
immediate, specific failure (`all_reduce returned X, expected N`) at the exact
line the transport broke. In the real training-rank portion of a healthy log,
`launching 2 DDP ranks` is followed by `process group up (nccl); testing a
collective` and `collective ok: all ranks connected`. If the log stops between
the latter two lines, this is the failure mode to suspect first.

### Stall watchdog: distinguishing "slow" from "hung" from anywhere in the log

**Problem, not yet a bug:** only rank 0 draws tqdm bars (the others would
interleave garbage into one shared Kaggle log stream), so a frozen bar cannot
tell a stuck DataLoader worker apart from a hung peer rank or a dead
collective — they all just look like silence. Kaggle's `!` cell output can
also keep displaying an already-cleared (`leave=False`) bar well after it
finished, making a *healthy* run look stalled. **Fix:** `heartbeat()` is
called at every meaningful step (waiting for a batch, transfer, forward or
backward, gather, and barrier boundaries) and a background thread per rank
(`start_stall_watchdog`) dumps every thread's stack via `faulthandler` after
`STALL_DUMP_SECONDS` (default 300) without a heartbeat. The watchdog can still
signal child processes if workers are enabled for a future experiment, but
the supported DDP configuration uses none. Read the resulting `[stall]`
label and bottom frame: `all_gather_object`/`dist.barrier` means the rank is
waiting on another rank. A stall labelled `barrier: waiting for rank 0
to finish checkpoint/W&B work` is the expected wait during an upload, not a
hang — `barrier()` labels it explicitly before blocking so it is not
mistaken for one. Verified locally by injecting a synthetic 25 s stall into a
loader worker with `STALL_DUMP_SECONDS = 8`: both ranks reported the stall
independently and the dump pointed straight at `_get_data`.

### Loader fork deadlock at `warm-start baseline: 0/29`

**Symptom:** NCCL startup and its test collective both pass, the Arm 13 weights
load, and the log then remains at the first warm-start validation batch.
**Cause:** each DDP rank used `DataLoader(num_workers=2)`. On Linux those
workers default to `fork`; their first creation happened only when validation
was iterated, after CUDA, NCCL, and (on rank 0) W&B had already created native
threads. Forking that state can leave a loader child blocked before it returns
its first batch. **Fix:** `WORKERS = 0` keeps mmap slicing inside each rank.
The costly normalization and resize already run on the GPU, and the two ranks
still load independently, so reliability is worth more than speculative CPU
prefetch. Worker-only DataLoader options are omitted when the count is zero,
and ranks synchronize immediately before the baseline evaluation. Do not
raise `WORKERS` again without changing the mmap dataset to reopen its files
lazily in explicitly spawned workers.

### Joint-finetune throughput regression at 3.82 seconds/batch

**Symptom:** frozen epochs complete in about 21 minutes, but the first joint
epoch rises to roughly 70 minutes at 8.6 GB/GPU. **Cause:** joint backward
recomputed every checkpointed backbone stage; each batch also invoked CoAtNet
once globally plus once for each of five ROI targets, performed five
GPU-to-Python validity checks, copied gradients into DDP buckets, and used the
generic AdamW path. **Fix:** keep the global call separate but pack the five
fixed-shape ROI bags into one chunked encoder execution; encode masked zero
slots and apply the unchanged masks after splitting features; defer scalar
diagnostics to their logging cadence; use DDP gradient bucket views, foreach
clipping, and the checkpoint-compatible standard AdamW. The completed probe
fixed eager checkpoint stages `(0,1)` as the production policy. These execution changes do
not alter examples, windows, effective batch, loss, LR schedule, parameter
names, validation, or model selection.

### Resume failure: `RNG state must be a torch.ByteTensor`

**Symptom:** the real ranks pass NCCL setup and load the resume checkpoint,
then fail in `torch.set_rng_state` before the first resumed epoch. **Cause:**
loading the whole checkpoint with `map_location=device` moved the saved CPU RNG
byte tensor onto CUDA; the CPU RNG setter refuses CUDA tensors. **Fix:** load
the checkpoint on CPU, let `load_state_dict` copy model and optimizer values to
their devices, explicitly normalize RNG tensors back to CPU before restoration,
restore only the current rank's CUDA generator, and release the checkpoint
payload afterward. This also avoids retaining a checkpoint-sized duplicate on
GPU during training.

### Resume failure: `Expected grad_scale and found_inf to be None`

**Symptom:** resume succeeds and joint fine-tuning reaches the first optimizer
step, then AdamW's `_multi_tensor_adam` raises an assertion about AMP scale
tensors. **Cause:** the trainer constructed fused AdamW, but loading an older
standard-AdamW checkpoint replaced its parameter-group `fused`/`foreach`
settings. AMP still treated the optimizer as fused while Adam dispatched to
the foreach implementation. **Fix:** construct the same standard AdamW used by
the checkpoint. Packed ROI execution and selective checkpointing provide the
material speedup; optimizer implementation consistency protects resumability.

### W&B API key committed in plaintext

**Symptom:** a real `WANDB_API_KEY` value was found hardcoded in the
CONFIGURATION block of a copy of this file that had already been uploaded to
a Kaggle dataset and pasted into a chat transcript. **Why this is worse than
a normal secret leak:** every re-upload, every `%%writefile`'d copy, and every
copy an agent pastes back to a user for review carries the key forward
verbatim — there is no single place to redact it after the fact. **Fix:**
`WANDB_API_KEY` in the file now defaults to `""`; `init_wandb` falls back to
the `WANDB_API_KEY` environment variable. Load it from a Kaggle Secret at the
top of the notebook, never into the file:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

If a key was ever pasted into a file, chat, or committed anywhere, rotate it
at wandb.ai/authorize — do not assume redacting the file afterward is
sufficient, since copies (Kaggle dataset versions, git history, chat
transcripts) may already hold it.

## Outputs and recovery

The important outputs are `armc_roi_last.pt` (resume),
`armc_roi_checkpoint_epNNN.pt` (artifact checkpoints), and
`armc_roi_selected_model.pt` (inference). Download the complete W&B artifact
directory when resuming; the selected model alone is not a resumable state.
Resume onto the same number of GPUs the checkpoint was written on: `world_size`
is checked against the saved configuration and a mismatch is refused, because
it would change the OneCycle horizon mid-schedule.

An older checkpoint from the same two-rank recipe can resume under the
optimized execution path. Wait until its current epoch has completed and
`armc_roi_last.pt` is safely written, stop that process, attach the complete
checkpoint (or latest complete W&B artifact), set `RESUME`, and relaunch. The
new performance fields are not part of the compatibility tuple, while the
architecture, parameters, optimizer groups, scheduler horizon, and training
data contract are unchanged.

## Separate Arm16 experiment

`kaggle/arm16/train_arm16.py` is a different orientation-canonicalization
experiment. It is not Arm C ROI, does not consume the ROI arrays, and requires
`kaggle/orient/output/orient.npz` in addition to cache64 and labels. Keep its
outputs and conclusions separate from this recipe.
