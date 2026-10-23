# References index — AI routing guide

AI INSTRUCTIONS: This repository's reference knowledge is split into shallow,
domain-specific folders so you only load the context your current task
needs. Do not read every file in this tree. Instead:

1. Identify which domain your task touches from the table below.
2. Read that domain's `INDEX.md` first — it maps each file in the folder to
   the specific rules/constants/symbols it contains.
3. Read only the specific file(s) in that folder that your task needs.
4. If your task spans multiple domains (e.g. changing a crop size that
   affects both the ROI cache and Arm C ROI training), read each relevant
   domain's `INDEX.md` and follow its pointers — do not guess cross-domain
   content from memory.

Never read files under `archive/` unless you are specifically auditing the
original monolithic documents or verifying that a split preserved content —
they are superseded by the domain folders below and may drift out of date.

## Domain routing table

| If your task involves... | Read |
|---|---|
| The FastAPI/Uvicorn server, the browser annotation UI, SQLite schema, coordinate-mapping/click logic, authentication, exports, or anything under `kaggle/roi_cache/web_annotator/` | `read_file('references/web_annotator/INDEX.md')` |
| ROI crop geometry, the laterality/paired-candidate contract, target windows (MCL/lateral meniscus/lateral OA/PF OA/synovitis), `roi_geometry.py`, `build_roi_cache.py`, `annotate_roi_pilot.py`, or cache build/mirror-gate/laterality incidents | `read_file('references/roi_cache/INDEX.md')` |
| The Arm C ROI model architecture, ROI residual heads, training hyperparameters/schedule, checkpoint/selection logic, launch inputs, or `kaggle/armc_roi/` | `read_file('references/armc_roi_training/INDEX.md')`, then `launch.md` for execution |
| The raw DICOM corpus structure, `train.csv`/`train_series.csv` schema, label prevalence, laterality/report/geometry EDA findings, `run_dataset_eda.py`, or building `gold_test.csv`/`train_pool.csv` via `scripts/prepare_dataset.py` | `read_file('references/dataset/INDEX.md')` |
| Loading/warm-starting from a trained `.pt` checkpoint, which `kaggle/arm*` run produced it, or the `model-weights` GitHub release | `read_file('references/model_weights/INDEX.md')` |
| Renting a RunPod GPU (when Kaggle's weekly GPU quota is exhausted), SSH access, staging the Kaggle image caches onto the pod, running training there, or any RunPod/Kaggle-CLI/aria2 staging issue | `read_file('references/runpod/RUNBOOK.md')` |

## Cross-domain notes

- The web annotator (`web_annotator/`) exists to produce the landmark
  annotations that `roi_cache/` consumes to fit crop priors — read both when
  changing the annotation schema or landmark set.
- `roi_cache/` produces the cache that `armc_roi_training/` consumes — read both
  when changing crop size, window count, or cache array shape.
- `dataset/` is the ground truth for corpus facts (laterality availability,
  slot coverage, geometry) that both `web_annotator/` and `roi_cache/` build
  on — read it first if a downstream rule looks surprising.

Do not summarize domain knowledge in this file. If a fact belongs in a
sub-index or a chunk file, put it there and keep this file purely
navigational.
