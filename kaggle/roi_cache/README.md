# Arm C DICOM-to-ROI cache

This directory implements the gated preprocessing workflow for Arm C. It does
not use pathology labels, does not infer laterality from image appearance, and
does not silently substitute an unrelated acquisition plane.

## Files

- `roi_geometry.py`: split-agnostic DICOM geometry, canonical axes, crop
  projection, padding, triplets, and deterministic series selection.
- `build_roi_cache.py`: pilot selection, prior fitting/auditing, smoke render,
  two-shard cache build, provenance, determinism checks, and final validation.
- `annotate_roi_pilot.py`: Kaggle notebook slider/click annotation interface.
- `test_roi_geometry.py`: pure geometry and alignment regression tests.

The required Kaggle inputs are the competition directory containing
`train_series.csv` and `train_series/`, plus the cache64 dataset containing
`ids_shard0.npy` and `ids_shard1.npy`.

## Gate 1: twenty-study pilot

Select 16 complete and four imperfect acquisitions deterministically without
reading `train.csv` or any pathology label:

```bash
python build_roi_cache.py select-pilot \
  --competition-root /kaggle/input/competitions/rsna-knee-abnormality-detection \
  --output-dir /kaggle/working/roi_pilot
```

In a notebook, annotate the generated template:

```python
from annotate_roi_pilot import ROIAnnotator

annotator = ROIAnnotator(
    competition_root="/kaggle/input/competitions/rsna-knee-abnormality-detection",
    manifest_csv="/kaggle/working/roi_pilot/pilot_manifest.csv",
    annotations_csv="/kaggle/working/roi_pilot/pilot_annotations.csv",
    annotator="your-name",
)
annotator.show()
```

Every click is immediately converted to patient coordinates and reprojected
as a red cross. Only rows saved with `qc_status=pass` can be fitted.

Run leave-one-study-out coverage:

```bash
python build_roi_cache.py fit-priors \
  --phase pilot \
  --annotations /kaggle/working/roi_pilot/pilot_annotations.csv \
  --output /kaggle/working/roi_pilot/pilot_priors.json
```

The command fails unless minimum target coverage is at least 90%, laterality
availability is at least 95%, and there are zero medial/lateral mirror errors.
Visual superior/inferior and anterior/posterior review remains a human gate.

Build all twenty pilot studies before expanding the annotation set:

```bash
python build_roi_cache.py build-pilot \
  --manifest /kaggle/working/roi_pilot/pilot_manifest.csv \
  --priors /kaggle/working/roi_pilot/pilot_priors.json \
  --output-dir /kaggle/working/roi_pilot/cache
```

This uses the production builder and writes the same tensor/provenance schema
for the pilot IDs, including deterministic three-study rebuild checks. Review
the annotation UI reprojections and pilot crops before proceeding to 100.

## Gate 2: final 80/20 prior fit

Expand the same annotation schema to exactly 100 complete studies, then run:

```bash
python build_roi_cache.py fit-priors \
  --phase final \
  --annotations /kaggle/input/arm-c-annotations/roi_annotations.csv \
  --output /kaggle/working/roi_priors.json
```

Seed `20260919` freezes the first 80 shuffled studies as the fit set and the
next 20 as the held-out audit. The exact IDs are serialized. Final priors are
rejected below 95% held-out coverage.

## Smoke test and full build

Run the supplied example path before a shard build:

```bash
python build_roi_cache.py smoke \
  --dicom-path /kaggle/input/competitions/rsna-knee-abnormality-detection/train_series/1.2.826.0.1.3680043.8.498.10004945927472656027199792075652399585/1.2.826.0.1.3680043.8.498.27845969797953115400959653384511165237/1.2.826.0.1.3680043.8.498.11113037817423835631916468081726445529.dcm \
  --priors /kaggle/input/arm-c-priors/roi_priors.json \
  --output /kaggle/working/roi_smoke.png
```

The smoke command builds the study twice, requires identical pixels/masks, and
renders source context plus all five four-window bags.

Run the full cache as two separate CPU notebooks/jobs:

```bash
python build_roi_cache.py build --shard 0 \
  --priors /kaggle/input/arm-c-priors/roi_priors.json \
  --output-dir /kaggle/working
```

```bash
python build_roi_cache.py build --shard 1 \
  --priors /kaggle/input/arm-c-priors/roi_priors.json \
  --output-dir /kaggle/working
```

Each job reads its authoritative `ids_shard<S>.npy`, writes five
`uint8 [N,4,3,336,336]` memmaps and five binary `[N,4]` mask memmaps, and
rebuilds three seeded studies to require byte-identical results. It also writes
`roi_ids`, compressed per-window provenance, metadata, and the exact priors.

After combining both shard outputs in one Kaggle dataset, validate:

```bash
python build_roi_cache.py validate \
  --cache-dir /kaggle/input/arm-c-roi-cache \
  --ids-dir /kaggle/input/rsna-cache64
```

Validation requires exact ID equality, final accepted priors, exact shapes and
dtypes, binary masks, 20 provenance rows per study, strictly ordered triplet
positions, at least 90% two-window coverage among eligible studies, at least
95% known laterality, and no more than 20% 95th-percentile padding.

Run geometry tests in a Kaggle notebook image with the dependencies installed:

```bash
python -m unittest -v test_roi_geometry.py test_roi_builder.py
```

Do not start Arm C training until `validate` passes. The trainer independently
compares every `roi_ids_shard<S>.npy` against its cache64 ID vector and refuses
misaligned data.
