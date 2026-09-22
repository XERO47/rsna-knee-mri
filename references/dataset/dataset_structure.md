# RSNA knee MRI dataset: structure, corpus size, and labels

Source: `DATASET_REFERENCE.md`. Validated on 2026-09-19 against the local
`data-root/` copy. This document is a reviewed interpretation of the
generated evidence in `eda_outputs/`; it is not a description copied from a
model notebook. See `eda_rules.md` for the reproducible EDA methodology that
produced these numbers.

## Executive summary

The local dataset is a 530.63 GiB DICOM collection with 4,407 training
studies, 24,371 training series, and 819,078 training images. The local test
bundle is only three studies, 15 series, and 557 images (0.56 GiB). CSV
identifiers, directory identifiers, and representative DICOM identifiers
reconcile exactly.

The central constraint is supervision: every study has a radiology report,
but only 58 studies (1.32%) have all 12 binary targets populated in
`train.csv`. The other 4,349 studies have no target values at all; there are
no partially labeled rows. Report-derived labels are therefore weak labels
unless their extraction procedure is independently validated. Prevalence,
co-occurrence, threshold tuning, and validation scores computed from the 58
labeled studies have high uncertainty.

The imaging data is heterogeneous but unusually clean at the container and
geometry levels. All 24,371 representative training headers were readable;
all declared series exist on disk; every representative header has pixel
spacing, Image Orientation Patient (IOP), and Image Position Patient (IPP);
and the declared plane agrees with the dominant IOP plane for every series.
All 315 series in the full slice-level audit were geometrically projectable
and stable within 1°. However, zero of those 315 series were spatially
monotonic when sorted by filename. Physical sorting from IOP/IPP is
mandatory.

Laterality is the main metadata limitation for anatomy-specific ROIs. Only
2,202 of 4,407 studies (49.97%) have a decisive `L` or `R` majority from
`ImageLaterality`/`Laterality`; 2,204 are unknown and one conflicts/ties. A
pipeline that requires DICOM laterality for medial/lateral ROIs cannot reach a
95% availability target on this dataset without another validated source. See
`../roi_cache/laterality_rules.md` for how the ROI cache pipeline works around
this.

## Directory and table structure

```text
data-root/
├── train.csv
├── train_series.csv
├── train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
├── test.csv
├── test_series.csv
├── test_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
└── sample_submission.csv
```

The macOS `.DS_Store` files found around the root are not data. There are no
empty series directories and no non-DICOM files inside training series.

### `train.csv`

One row per study (4,407 rows, no duplicate study IDs):

- `StudyInstanceUID`: string join key.
- `Report`: non-empty free-text radiology report.
- 12 nullable binary targets: `ACL`, `MCL`, `Medial Meniscus`, `Lateral
  Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`,
  `Baker's`, `Contusion`, and `Fracture`.

All observed target values are exactly 0 or 1. Exactly 58 rows have all
targets; 4,349 have none. **Do not silently fill missing targets with zero.**

### `train_series.csv`

One row per series (24,371 rows, no duplicate series IDs):

- `StudyInstanceUID` and `SeriesInstanceUID`: join/path keys.
- `Fluid_Sensitive`: binary sequence label.
- `Fat_Suppression`: binary sequence label.
- `Anatomical_Plane`: `Axial`, `Coronal`, or `Sagittal`.

In this release `Fluid_Sensitive` and `Fat_Suppression` are identical on
every row. Preserve the original fields in data contracts, but do not count
them as two independent sources of information.

### Test and submission files

`test.csv` contains three study IDs. `test_series.csv` contains 15 series,
and `sample_submission.csv` contains the same three IDs plus the 12 targets
in the correct order. This is adequate for end-to-end inference testing only.
It is far too small for conclusions about the hidden-test distribution.

## Corpus size and integrity

| Split | Studies | Series | DICOM files | Bytes | GiB |
|---|---:|---:|---:|---:|---:|
| Train | 4,407 | 24,371 | 819,078 | 569,155,361,080 | 530.07 |
| Local test | 3 | 15 | 557 | 599,962,984 | 0.56 |
| Total | 4,410 | 24,386 | 819,635 | 569,755,324,064 | 530.63 |

Integrity checks passed:

- CSV study IDs and filesystem study directories agree.
- CSV series IDs and filesystem series directories agree.
- representative DICOM Study/Series UIDs agree with their directory names;
- all UIDs use dotted numeric syntax;
- there are no empty series and no representative-header read failures;
- all 79 deterministically sampled pixel arrays decoded successfully.

All series use MR modality, Explicit VR Little Endian transfer syntax
(`1.2.840.10008.1.2.1`), and `MONOCHROME2`. A robust loader should still
retain modality-LUT, transfer-syntax, and `MONOCHROME1` handling for future
data.

## Study and series composition

A training study contains 3–14 series (median 5, mean 5.53; p95 9). Plane
counts are 9,864 sagittal, 8,609 coronal, and 5,898 axial series.

| Slot | Series | Studies covered | Coverage |
|---|---:|---:|---:|
| Axial, non-fluid-sensitive | 1,179 | 857 | 19.45% |
| Axial, fluid-sensitive | 4,719 | 4,407 | 100.00% |
| Coronal, non-fluid-sensitive | 3,985 | 3,406 | 77.29% |
| Coronal, fluid-sensitive | 4,624 | 4,248 | 96.39% |
| Sagittal, non-fluid-sensitive | 5,197 | 4,266 | 96.80% |
| Sagittal, fluid-sensitive | 4,667 | 4,150 | 94.17% |

Absence must be represented by an explicit mask. Substituting an unrelated
plane or silently treating a missing slot as a real zero image leaks
acquisition patterns and creates ambiguous supervision. Axial
non-fluid-sensitive imaging is particularly sparse and should not be a hard
requirement.

Series depth ranges from 11 to 320 images (median 30, p95 45, p99 160). The
long right tail is concentrated in some axial and sagittal acquisitions and
is consistent with thin-slice/3D protocols. Slot-specific medians are 29–32
slices; see `series_depth_by_slot.csv` before choosing a fixed slice sampler.
Across all series, 52 have fewer than 12 images, 838 have fewer than 18, and
4,247 have fewer than 24. Sampling code must specify whether it masks, pads,
or resamples short series and how it preserves focal abnormalities in long
series.

## Labels: what is known and what is not

The table below uses only the 58 fully labeled studies. The interval is the
95% Wilson binomial interval and shows why point estimates should not be
treated as stable class priors. See `eda_rules.md` for the exact Wilson
interval formula used.

| Target | Positive / 58 | Prevalence | 95% interval |
|---|---:|---:|---:|
| ACL | 24 | 41.4% | 29.6–54.2% |
| MCL | 9 | 15.5% | 8.4–26.9% |
| Medial Meniscus | 26 | 44.8% | 32.7–57.5% |
| Lateral Meniscus | 23 | 39.7% | 28.1–52.5% |
| Medial OA | 15 | 25.9% | 16.3–38.4% |
| Lateral OA | 11 | 19.0% | 10.9–30.9% |
| PF OA | 21 | 36.2% | 25.1–49.1% |
| Effusion | 35 | 60.3% | 47.5–71.9% |
| Synovitis | 27 | 46.6% | 34.3–59.2% |
| Baker's | 12 | 20.7% | 12.3–32.8% |
| Contusion | 19 | 32.8% | 22.1–45.6% |
| Fracture | 18 | 31.0% | 20.6–43.8% |

Exploratory phi correlations show plausible shared structure, including
Effusion–Synovitis (+0.40), Medial Meniscus–Medial OA (+0.42), and Lateral
Meniscus–Lateral OA (+0.42). The largest observed association is Medial
OA–Baker's (+0.48). These are hypotheses, not reliable dependencies: with
only 58 rows, a few studies can move a coefficient substantially. Do not use
these values alone to hard-code loss weights, calibration groups, or
architecture.

### Validation implications

- A five-fold split gives only about 11–12 gold studies per validation fold.
  Per-target AUC may be undefined or extremely noisy if a fold lacks one
  class.
- Use iterative/multilabel stratification where possible and verify both
  classes exist for every scored target in every fold.
- Report bootstrap confidence intervals and out-of-fold predictions, not one
  lucky split.
- Keep gold and report-derived labels distinguishable. Never overwrite the
  original nullable target columns.

## Reports

All 4,407 studies have a non-empty report. Median length is 129 words (p05
28, p95 330; range 8–690), or 977 characters (p05 205, p95 2,453). Reports
are multilingual and vary greatly in formatting: the median is two lines,
while the p75 is 22 lines.

Vocabulary fingerprints found overlapping evidence of English (1,833
reports), Spanish/Portuguese (550), Italian (368), French (81), German (46),
and Turkish (523). These regex counts overlap and are deliberately not called
language labels. They establish only that an English-only parser would be
inadequate.

Whitespace-normalized, lower-cased report hashes place 204 studies in 54
duplicate-text groups; the largest group has 37 studies. Identical reports
can be templates rather than duplicate patients, but they can still leak
report wording across random folds. Grouping or auditing by normalized
report hash is recommended for any report-supervised experiment.

Safe handling rules:

- keep the raw report immutable;
- store extracted weak labels separately with parser version and confidence;
- validate each target and language family against the 58 gold studies;
- handle negation, uncertainty, laterality, and normal findings explicitly;
- split before fitting vocabularies, pseudo-label calibrators, or thresholds;
- never present report-language fingerprints as a validated classifier.

## DICOM geometry and physical scale

Representative headers from every training series have `PixelSpacing`, IOP,
and IPP. The competition plane agrees with the IOP-derived dominant plane in
all 24,371 cases.

In-plane spacing ranges from 0.0728 to 1.1719 mm/pixel; median is 0.3125, p05
is about 0.167, and p95 is 0.5625. Image matrices range from 160 to 1,280
rows and 160 to 1,444 columns, with 512×512 most common. Despite that pixel
variation, physical FOV is tighter: median 160×160 mm, approximately 140 mm
at p05, and 189–190 mm at p95. Outliers span roughly 70–361 mm.

Slice thickness ranges from 0.6 to 6.0 mm (median 3.0, p95 4.0).
`SpacingBetweenSlices` is present in 22,792 of 24,371 series and has median
3.5 mm and p95 5.02 mm. It must not replace position-derived spacing when
IPP is available.

The deterministic full-slice audit covered 300 training and all 15
local-test series:

- all 315 had projectable IOP/IPP on every slice;
- all 315 varied in orientation by no more than 1°;
- none had slice positions duplicated within 0.1 mm;
- median physical slice step was 3.50 mm;
- zero were physically monotonic under lexicographic filename order.

The last result is expected because filenames are SOP UIDs, not sequence
numbers. See `eda_rules.md` for the required physical slice-ordering and
pixel-to-patient projection formulas.

Physical-mm cropping/resampling is therefore necessary. A fixed pixel crop
would cover very different anatomy across scanners.

## Scanner and protocol heterogeneity

Raw manufacturer strings contain aliases, so normalization is necessary.

| Normalized vendor family | Series | Share |
|---|---:|---:|
| Siemens | 10,186 | 41.8% |
| Philips | 7,611 | 31.2% |
| GE | 5,101 | 20.9% |
| Canon/Toshiba | 1,343 | 5.5% |
| Fujifilm/Hitachi | 130 | 0.5% |

The most frequent scanner models include Ingenia (3,802 series), Aera
(2,409), MAGNETOM Vida (2,399), Achieva dStream (2,217), and MAGNETOM Avanto
fit (1,377). Field strength is 1.5 T for 14,108 series, 3 T for 8,922,
missing for 1,206, 1.16 T for 130, and 1 T for five.

Protocol parameters also vary: representative repetition times span
4.6–8,822 ms and echo times 0.016–132 ms. Those ranges mix sequence families,
so raw intensity is not comparable across series. Normalize per series after
applying rescale/modality information, and include scanner/protocol
diversity when forming validation folds or pilot annotation samples.

`BodyPartExamined` is noisy: 19,360 series say `KNEE`, 2,322 are missing,
2,265 say `EXTREMITY`, and small counts contain unrelated values such as
`WRIST`, `ANKLE`, or `HEAD`. Because directory/UID/plane geometry is
otherwise consistent, this field should not be used to delete series.

Patient age and study date are absent from all inspected representative
headers. Patient sex is populated on many series, but includes `O` and
missing values; it is series-level repeated metadata and should not be
interpreted as a patient count without first aggregating and checking
within-study consistency.

## Laterality and ROI consequences

Study-level majority aggregation over `ImageLaterality` and `Laterality`
gives:

| Status | Studies | Share |
|---|---:|---:|
| Right | 1,110 | 25.19% |
| Left | 1,092 | 24.78% |
| Unknown | 2,204 | 50.01% |
| Conflict/tie | 1 | 0.02% |

Therefore the earlier goal of at least 95% metadata-derived laterality is not
attainable as written. Before building laterality-dependent medial/lateral
ROI caches, choose and validate one of these approaches:

1. extract laterality from multilingual report/series text and audit it
   against the 2,202 DICOM-known studies;
2. annotate or train a dedicated orientation/laterality classifier with a
   held-out error audit;
3. redesign medial/lateral targets to use paired symmetric candidates and a
   mask/attention mechanism that does not require a side label.

Do not infer laterality ad hoc from image display orientation. DICOM patient
coordinates, acquisition orientation, and viewer presentation are distinct.
The ROI cache pipeline chose approach 3; see `../roi_cache/laterality_rules.md`.

## Recommended data contract

Every derived sample or cache row should retain:

- `StudyInstanceUID`, `SeriesInstanceUID`, and ordered SOP UIDs;
- declared plane and fluid/fat-suppression flags;
- physical slice positions and the ordering method;
- pixel spacing, crop center/extent in millimetres, padding fraction, and
  output resolution;
- manufacturer/model/field strength and normalized vendor family;
- laterality value, source, confidence, and conflict status;
- validity mask plus an explicit failure reason;
- source-data fingerprint and preprocessing-code version.

For every sharded cache, save an ID array and require exact element-wise ID
equality in the loader. Matching only row counts is unsafe.

## Practical checks before training

1. Reconcile study and series IDs against the committed EDA outputs.
2. Split at study level and group duplicate report hashes where reports or
   report-derived labels are used.
3. Verify both target classes exist in each gold validation fold.
4. Sort slices by IPP projected onto the IOP normal; never by filename.
5. Apply modality rescale, handle photometric inversion defensively, and use
   deterministic per-series intensity normalization.
6. Crop/resample in millimetres and propagate missing-slot/window masks.
7. Track performance by vendor, field strength, plane availability, spacing,
   and laterality-known status — not only overall AUC.
8. Do not treat the three-study local test split as distribution evidence.

## Limits of this EDA

- Pixel content was decoded for 79 deterministic samples, not visually
  reviewed across the entire corpus.
- Full slice geometry was audited for 315 series rather than all 24,386;
  every representative series header was still inspected.
- Target statistics have only 58 labeled observations.
- Language fingerprints are overlapping regex probes, not validated language
  identification.
- DICOM metadata can be anonymized, missing, or semantically noisy even when
  syntactically valid.
- The local test set is a three-study code fixture and may not describe the
  scoring test set.

Re-run `run_dataset_eda.py` after any data replacement, extraction change, or
competition refresh. Compare `summary.json`, the mismatch table, and
laterality coverage before reusing existing caches or assumptions. See
`eda_rules.md` for how to run it.
