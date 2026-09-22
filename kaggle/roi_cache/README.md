# Arm C DICOM-to-ROI cache

This directory contains the deterministic, label-free ROI preprocessing path
for Arm C. It uses DICOM patient-space geometry, millimetre crops, explicit
window masks, exact cache-ID alignment, and per-window provenance.

Read `../../references/ROI_CACHE_REFERENCE.md` before modifying the pipeline.
It records the EDA-driven decisions, complete data contracts, and a summary of
every file, class, and function.

## Important EDA-driven change

Only 2,202 of 4,407 training studies (49.97%) have decisive DICOM laterality.
The old design masked MCL, lateral-meniscus, and lateral-OA windows when side
was unknown and then required 95% known laterality during validation. That
workflow could never pass on this dataset.

The production contract now remains `[N,4,3,336,336]` while avoiding guessed
laterality:

- MCL, lateral meniscus, and lateral OA: one centered triplet for the left-knee
  hypothesis and one for the right-knee hypothesis in each of two planes;
- PF OA: two overlapping triplets in each plane;
- synovitis: two distinct, in-bounds triplets whose middle slices are nearest
  the suprapatellar landmark in each plane.  Its sagittal 100 x 100 mm crop is
  shifted 25 mm inferiorly (leaving a 25 mm superior margin), following
  radiologist review; acquisition-truncated sagittal windows remain masked;
- the target-specific attention pooler chooses useful windows;
- known laterality is retained in provenance for auditing, not required to
  generate the cache.

This keeps the cache size and Arm C loader unchanged and makes all studies
eligible. Do not replace the paired candidates with patient-X sign or a simple
report regex: local audits reached only 97.18% and 98.22% agreement,
respectively, among decisive cases, which is not safe for side-specific crops.

## Files

- `roi_geometry.py`: split-agnostic geometry and deterministic selection
  primitives.
- `build_roi_cache.py`: selection, prior fitting, cache building, provenance,
  smoke rendering, and validation CLI.
- `annotate_roi_pilot.py`: resumable local Matplotlib annotation window.
- `run_local_pilot.py`: one-command local pilot selection, annotation, prior
  fit, smoke render, and single-shard cache build.
- `test_roi_geometry.py` and `test_roi_builder.py`: regression tests.

## Local order of operations

The examples assume the repository is `rsna-knee-mri` and the raw files are in
its sibling `data-root` directory.

### 1. Run tests

```bash
cd rsna-knee-mri
python -m unittest discover -v -s kaggle/roi_cache -p 'test_*.py'
```

### 2. Run the complete local pilot

```bash
python kaggle/roi_cache/run_local_pilot.py \
  --competition-root ../data-root \
  --work-dir roi_work/pilot \
  --annotator your-name \
  --workers 4
```

The script resumes the already-generated selection in `roi_work/pilot`, opens
a normal desktop Matplotlib window, and shows one landmark task at a time. Use
the slice slider or left/right keys, click the landmark, verify the red marker,
and press Enter or **Save + next**. `A`/`D` move between tasks.

Every click is saved atomically. Closing the window early is safe; rerun the
same command to resume at the first incomplete task. After all 80 landmarks are
saved, the same process automatically fits the LOO priors, renders
`roi_smoke.png`, and builds the complete 20-study cache into one local shard at
`roi_work/pilot/cache`.

The manifest is deterministically balanced: 10 left/10 right, 16 complete/4
imperfect acquisitions, and one study from each vendor family for which trusted
laterality exists. Click and save all four landmarks for each study.

### 3. Inspect the pilot result

Open `roi_work/pilot/roi_smoke.png` and inspect
`roi_work/pilot/pilot_priors.json`. The priors must say `"accepted": true`.
The pilot cache is approximately 130 MiB and is intentionally a single shard;
it does not require cache64 IDs.

Do not expand the cohort until leave-one-study-out coverage passes and the
reprojections/crops have been visually reviewed.

### 4. Expand to the frozen 100-study cohort

```bash
python kaggle/roi_cache/build_roi_cache.py select-cohort \
  --competition-root ../data-root \
  --pilot-manifest roi_work/pilot/pilot_manifest.csv \
  --output-dir roi_work/annotations
```

This retains the pilot, adds 80 studies, balances left/right 50/50, retains
80 complete and 20 imperfect acquisitions, and freezes `prior_partition` as
80 fit and 20 audit studies before annotation. Annotate
`roi_work/annotations/roi_annotations.csv` with the same UI.

### Exception: controlled cohort revision

If an annotated study proves clinically unsuitable because its DICOM metadata
selected the wrong reference view, do **not** overwrite the manifest or the
annotation database. Prepare a separate, review-only revision instead:

```bash
python kaggle/roi_cache/prepare_roi_revision.py \
  --competition-root ../data-root \
  --manifest roi_work/annotations/annotation_manifest.csv \
  --output-dir roi_work/annotations/revisions/final-r1-provisional \
  --patient-number 39 \
  --qc-count 20
```

It preserves frozen cohort counts, chooses a deterministic label-free
replacement whose reference `SeriesDescription` direction tokens agree with
the selected planes, and produces a 20-case visual-QC list. It does not edit
the old SQLite database and it is not an approval to train. A radiologist must
approve that list and annotate the replacement before a new revision database
is created. See `references/ROI_CACHE_REFERENCE.md` for the current patient-39
decision and activation order.

### 5. Fit final priors

```bash
python kaggle/roi_cache/build_roi_cache.py fit-priors \
  --phase final \
  --competition-root ../data-root \
  --annotations roi_work/annotations/roi_annotations.csv \
  --manifest roi_work/annotations/annotation_manifest.csv \
  --output roi_work/roi_priors.json
```

The final command refuses to reshuffle the frozen 80/20 partition and rejects
held-out target coverage below 95% or any medial/lateral mirror error.

### 6. Smoke-test real pixels

```bash
python kaggle/roi_cache/build_roi_cache.py smoke \
  --competition-root ../data-root \
  --dicom-path ../data-root/train_series/<study>/<series>/<sop>.dcm \
  --priors roi_work/roi_priors.json \
  --output roi_work/roi_smoke.png
```

Review the source context, both side hypotheses, all five targets, masks, and
failure reasons. The command also requires a byte-identical second build.

### 7. Build two shards

The authoritative `ids_shard0.npy` and `ids_shard1.npy` from cache64 are still
required. Run each shard independently:

```bash
python kaggle/roi_cache/build_roi_cache.py build \
  --shard 0 \
  --competition-root ../data-root \
  --ids-path /path/to/ids_shard0.npy \
  --priors roi_work/roi_priors.json \
  --output-dir roi_work/cache
```

Repeat with shard 1. Each shard is approximately 13.9 GiB before provenance
and filesystem overhead; keep at least 16 GiB free per shard. The full cache is
approximately 27.8 GiB, so it cannot fit into one 20 GiB Kaggle output job.

### 8. Validate before training

```bash
python kaggle/roi_cache/build_roi_cache.py validate \
  --cache-dir roi_work/cache \
  --ids-dir /path/to/cache64_ids
```

Validation requires exact IDs, accepted final priors, exact shapes/dtypes,
 binary masks, 20 provenance rows per study, correct paired/target-declared window
strategies, monotonic physical triplets, at least 90% two-window coverage for
every target across all studies, and p95 padding no greater than 20%.

Do not start Arm C training until this command passes and the held-out visual
audit is complete.
