# ROI cache domain index

Files in this folder, split from `ROI_CACHE_REFERENCE.md` (archived at
`../archive/`).

| File | Read this for |
|---|---|
| `laterality_rules.md` | Why the laterality contract uses paired L/R candidates instead of trusting DICOM side; the four-window strategy per target type (centered-triplet vs. overlapping-triplet); the frozen crop-size table (mm, per target); EDA alignment (physical slice ordering, missing-slot masking); annotation cohort composition (pilot 20, final 100); and the full chronological record of the final-cohort mirror/laterality incidents (patient-39 sequence-direction exception, `final-r1` revision, mirror-gate failure and fix, left-knee joint-line correction workflow, cache64 recovery, synovitis 25 mm policy, merged Kaggle dataset). Read this for *why a crop or side decision looks the way it does* and for cache build/output file locations. |
| `file_contracts.md` | Per-file contracts and complete function/class inventories for `roi_geometry.py`, `build_roi_cache.py`, `annotate_roi_pilot.py`, and `run_local_pilot.py`; the end-to-end data-flow diagram; output array/provenance schemas; and the test inventory. Read this to find *which function does what* before editing or calling into the ROI cache pipeline. |

`laterality_rules.md` and `file_contracts.md` cross-reference each other (e.g.
`SliceGeometry.patient_to_pixel`'s fix is explained in `laterality_rules.md`'s
sibling doc `../web_annotator/backend.md`, and listed in `file_contracts.md`'s
inventory) — read both when making a geometry or crop-policy change.
