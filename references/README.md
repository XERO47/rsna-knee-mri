# RSNA knee data references

This directory is the durable starting point for any future work on this
dataset. Read [INDEX.md](INDEX.md) first — it routes you to the right
domain sub-folder instead of loading everything at once.

## Contents

This tree is index-driven and organized by domain rather than as flat
monolithic files:

- `INDEX.md` — root AI routing guide. Read this first.
- `web_annotator/` — the Mac-hosted browser annotation application: contract
  (`spec.md`), UI/coordinate-mapping (`frontend.md`), and FastAPI/SQLite
  engineering reference (`backend.md`). The original contract supports two
  independent readers; the current operating decision is one shared account
  used jointly by both radiologists.
- `roi_cache/` — Arm C ROI cache: laterality/window/crop contract and
  chronological correction history (`laterality_rules.md`), plus per-file
  function/class inventories (`file_contracts.md`).
- `armc_roi_training/` — Arm C ROI model: architecture contract (`architecture.md`)
  and training/validation/checkpoint policy (`training_policy.md`).
- `dataset/` — raw corpus structure and findings (`dataset_structure.md`) and
  reproducible EDA methodology (`eda_rules.md`).
- `archive/` — the original five monolithic reference documents, superseded
  by the domain folders above. Kept for audit only; do not read for current
  guidance.
- `roi_cache_audit.json` — machine-readable selection, laterality, cache-size,
  and implementation-validation results for the ROI review.
- `run_dataset_eda.py` — deterministic, read-only EDA. It checks every CSV row,
  every series directory, one representative header from every series, a
  deterministic slice-level geometry sample, and a deterministic pixel decode
  sample.
- `eda_outputs/summary.json` — machine-readable headline results.
- `eda_outputs/*.csv[.gz]` — evidence tables behind the reference document.
- `eda_outputs/plots/` — generated overview figures.

The attached third-party notebook was used only as a checklist of possible EDA
topics. Every finding in this folder was recomputed from the local data.

## Reproduce

From the repository's parent directory:

```bash
python rsna-knee-mri/references/run_dataset_eda.py --data-root data-root
```

Dependencies are `numpy`, `pandas`, `pydicom`, `matplotlib`, and `seaborn`.
The project requirements already list the first four core data dependencies;
install `seaborn` only if plots are wanted, or pass `--skip-plots`.

Defaults inspect all 24,371 training-series headers. For a quick code smoke
test, use `--dicom-series-limit 200 --geometry-series-sample 10
--pixel-samples 5 --output-dir /tmp/rsna-eda-smoke`. Do not treat a limited run
as a replacement for the committed full-corpus results.

## Interpretation rules

1. `train_series.csv` is the source of the competition-provided plane and
   fluid-sensitivity labels; DICOM geometry remains the source of physical
   ordering and projection.
2. Never sort slices by filename. SOP UID filenames are not spatially ordered.
3. The 4,349 rows without target values are unlabeled in `train.csv`; reports
   are inputs, not automatically trustworthy labels.
4. Statistics about the 12 targets use only 58 fully labeled studies and must
   be described as uncertain.
5. The local test set has only three studies. It validates code paths, not test
   distribution or leaderboard generalization.
6. Any future refresh should rerun the EDA and compare `summary.json` before
   replacing these conclusions.
