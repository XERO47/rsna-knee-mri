# Dataset domain index

Files in this folder, split from `DATASET_REFERENCE.md` (archived at
`../archive/`), with methodology cross-referenced from `run_dataset_eda.py`
(kept in place at `../run_dataset_eda.py`, not moved).

| File | Read this for |
|---|---|
| `dataset_structure.md` | Directory/CSV layout (`train.csv`, `train_series.csv`, DICOM tree), corpus size (4,407 training studies, 530 GiB), the supervision constraint (only 58/4,407 studies fully labeled — never zero-fill missing targets), series/slot coverage percentages, per-target prevalence with Wilson intervals, report-language heterogeneity, DICOM geometry/physical-scale ranges, scanner/vendor/field-strength breakdown, and the laterality availability table (49.97% decisive). Read this for *what the corpus looks like* and *what numbers to cite*. |
| `eda_rules.md` | The reproducible EDA methodology: the `SEED=20260919` deterministic hash-sort sampling rule (`deterministic_subset`), the mandatory physical slice-ordering formula (IOP/IPP projection, never filename order), the exact Wilson-interval function used for label prevalence, the integrity/schema checks the script performs, the generated-output evidence map, and when to re-run the script. Read this before writing new EDA code or trusting a claim about "how a number in `dataset_structure.md` was computed." |

`dataset_structure.md` is findings; `eda_rules.md` is method. If a number
looks stale, re-run `run_dataset_eda.py` per `eda_rules.md` before trusting
`dataset_structure.md`.
