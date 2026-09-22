# ROI cache: laterality contract, target windows, and crop sizes

Source: `ROI_CACHE_REFERENCE.md` (decision summary, frozen target contract,
alignment with EDA, annotation cohort findings, and the chronological
laterality/mirror-correction incidents). Validated on 2026-09-19 against the
local `data-root` corpus described in `../dataset/dataset_structure.md`.

## Decision summary

The geometry, physical-mm crop, intensity, sharding, and provenance approach is
sound. The original laterality contract was not: DICOM laterality is decisive
for only 2,202/4,407 studies (49.97%), while the old validator required 95% and
zero-filled three targets for every unknown-side study. It would have discarded
about half of the MCL, lateral-meniscus, and lateral-OA signal and could never
pass its own acceptance gate.

The corrected approach keeps four windows and the current Arm C contract:

| Target type | Window 0 | Window 1 | Window 2 | Window 3 |
|---|---|---|---|---|
| MCL / lateral meniscus / lateral OA | plane 1, left hypothesis, centered triplet | plane 1, right hypothesis, centered triplet | plane 2, left hypothesis, centered triplet | plane 2, right hypothesis, centered triplet |
| PF OA / synovitis | plane 1, lower overlapping triplet | plane 1, upper overlapping triplet | plane 2, lower overlapping triplet | plane 2, upper overlapping triplet |

The target-specific gated-attention pooler can suppress the anatomically wrong
side. The strategy is identical for known- and unknown-side studies, preventing
metadata availability from changing tensor semantics. It costs no additional
storage and requires no trainer shape change.

This is the best current path because two cheaper inferred-side alternatives
failed strict local audits:

- median DICOM patient-X sign agreed with known side in 97.18% of 2,202
  known-side studies;
- a multilingual report regex was decisive on only 38.28% of known-side
  studies and was 98.22% accurate when decisive.

Neither error rate is acceptable for silently choosing a medial/lateral crop.
A laterality classifier should replace paired candidates only after an
independent scanner-stratified audit supports a zero-mirror-error requirement.

## Frozen target contract

| Target | Landmark | Crop (width × height) | Slot 1 | Slot 2 | Same-plane sequence fallback |
|---|---|---:|---|---|---|
| MCL | medial joint line | 90 × 120 mm | coronal fluid | axial fluid | no |
| Lateral meniscus | lateral joint line | 80 × 80 mm | sagittal non-fluid | coronal non-fluid | no |
| Lateral OA | lateral joint line | 85 × 75 mm | coronal non-fluid | sagittal non-fluid | no |
| PF OA | patellofemoral center | 95 × 85 mm | axial fluid preferred | sagittal fluid preferred | yes |
| Synovitis | suprapatellar recess | 100 × 100 mm | axial fluid | sagittal fluid | no |

Coronal width/height is medial-lateral/superior-inferior; sagittal is
anterior-posterior/superior-inferior; axial is medial-lateral/anterior-posterior.
No target ever substitutes an unrelated plane and no crop is mirrored.

`LANDMARKS` (in `roi_geometry.py`) fixes four annotation names. `TARGET_SPECS`
is the single source for target name, landmark, crop mm, side dependence,
fallback, and the two slots above.

## Alignment with the EDA

- Slices are sorted by IPP projected onto one series-level IOP normal. Filename
  order is never used; 0/315 audited filename-sorted series were physically
  monotonic.
- Every declared study series contributes to the coordinate-wise median study
  origin, so the origin does not shift merely because one target slot is absent.
- Crops are specified in millimetres because spacing ranges from 0.0728 to
  1.1719 mm/pixel.
- Missing acquisitions become zero windows plus false masks. Coronal non-fluid
  coverage is only 77.29%; the other used slots cover roughly 94–100%.
- No axial non-fluid slot is required; it exists in only 19.45% of studies.
- Per-series modality-LUT processing and percentile normalization address
  heterogeneous scanners/protocols. MONOCHROME1 handling remains defensive.
- Provenance retains geometry, laterality source, candidate side, physical
  center, crop bounds, padding, intensity limits, and ordered SOPs.
- Cache IDs come from cache64 and must match element-for-element.

## Annotation cohort findings

The former novelty-maximizing selector over-sampled 1.16 T Fujifilm/Hitachi:
4/20 pilot studies despite only 130/24,371 series. It also treated vendor
aliases as independent vendors.

The replacement normalizes vendors, aggregates study measurements with
medians, balances side exactly, includes short/imperfect cases, and matches
the eligible population distribution. The audited deterministic selections
are:

| Cohort | Side | Acquisition coverage | Vendor composition | Prior split |
|---|---|---|---|---|
| Pilot, 20 | 10 L / 10 R | 16 complete / 4 imperfect | 13 Siemens, 5 Philips, 1 GE, 1 Fujifilm/Hitachi | leave-one-study-out |
| Final, 100 | 50 L / 50 R | 80 complete / 20 imperfect | 73 Siemens, 25 Philips, 1 GE, 1 Fujifilm/Hitachi | 80 fit / 20 frozen audit |

The final 100-study cohort is the beneficial expansion. Do not enlarge the
first pilot before its visual and 90% leave-one-out gate passes; its purpose
is to catch systematic mistakes cheaply. After it passes, `select-cohort`
retains all 20 and adds 80. More than 100 landmark studies is not justified
until the held-out dispersion/coverage shows that fixed priors are the
limiting error.

Canon/Toshiba is absent from annotation because none of its studies has
trusted DICOM laterality, required to fit canonical medial/lateral offsets.
It is not excluded from cache generation: paired-side candidates give those
studies all five ROI bags. Include them in the post-prior visual stress audit.

## Chronological laterality/mirror-correction history

### 2026-09-21 final-cohort sequence-direction exception

The annotation selector's historical rule, "first matching `Anatomical_Plane`
and `Fluid_Sensitive` row", is not by itself sufficient for clinical reference
view suitability. A post-annotation audit of the 300 unique reference series
selected for the frozen final cohort compared the CSV plane with an
unambiguous directional token in DICOM `SeriesDescription` where available.
Of 241 series with such a token, four conflicted. Three were patient 39: the
app's claimed coronal medial/lateral reference was named `LT_t2_tse_sag`, and
the claimed axial/sagittal references were likewise inconsistent with the
visual sequence orientation. The radiologist correctly identified the
displayed medial-joint-line view as unsuitable and did not annotate it.

Do not use patient 39's forced selector output to fit priors or build crops.
The 99 other completed patients are not automatically invalidated: the same
audit found 237/241 unambiguous selected series aligned, and their saved
provenance can be retained pending targeted visual QC. The fourth conflict is
patient 1's ACL-oblique PF reference and must be reviewed separately.

This is a data-quality exception, not a reason to silently ignore an `unable`
row or reshuffle the cohort. Before final-prior fitting, create a controlled
manifest revision that replaces/repairs patient 39 with a visually approved,
metadata-and-description-consistent candidate while preserving the intended
50/50 laterality, 80/20 acquisition, and 80/20 prior partition balance. The
replacement and all changed IDs must be recorded in provenance and the web
annotation cohort must be regenerated/reannotated only for the changed study.
For future annotation UI work, offer all same-plane candidate series and make
the radiologist's explicit series choice part of the saved annotation record;
never auto-accept a first metadata match without this QC.

#### Controlled `final-r1` preparation (2026-09-21)

`kaggle/roi_cache/prepare_roi_revision.py` is the safe preparation step for
this exception. It is deliberately non-destructive: it only reads the old
manifest/DICOM headers and writes a separate provisional bundle. It never
edits the existing `final` SQLite database, so the 99 saved studies remain the
unchanged source record until human QC approves the revision.

The current generated bundle is
`roi_work/annotations/revisions/final-r1-provisional/`. It replaces only
original patient 39 (`...41698150452794914665945455397334605778`) with
`...11014091506864318694851337553197477906`. The replacement is label-free
selected and preserves left laterality, complete preferred-plane coverage,
Siemens manufacturer bin, 1.5 T field-strength bin, and the `fit` partition.
No unused study matched every protocol bin exactly, so the record explicitly
documents the permitted depth-bin relaxation rather than concealing it.

Its three required reference series are directionally self-consistent in both
the selector metadata and `SeriesDescription`: coronal `t1_tse_cor_320`, axial
`t2_pd_fs_TRA`, and sagittal `pd_tse_fs_sag_320`. This is an automated
screening check, not a clinical approval.

The bundle contains:

- `annotation_manifest.csv`: the provisional 100-study revision.
- `roi_annotations.csv`: a new blank 400-landmark template, never merged into
  the original annotation file.
- `annotation_ids.npy`: a safe Unicode (not Python-object) array with exactly
  the revised manifest row order.
- `replacement_decision.json`: the replacement, preserved constraints,
  selected DICOM-series facts, and outstanding human actions.
- `visual_qc_manifest.csv`: 20 targeted reviews: patient 1 (the remaining
  ACL-oblique conflict), 18 stratified retained cases, and the replacement.

Before activation, radiologists must visually approve all rows in
`visual_qc_manifest.csv`, confirm the replacement's three views, and annotate
its four landmarks. Only then may a *new* revision database import the 99
accepted annotations and add the replacement; never mutate the original
database or silently re-use patient 39. Then export canonical annotations, fit
final priors, pass the coverage/mirror gates, and build the cache.

After the stated visual QC was completed, `migrate_roi_revision.py` created
the isolated `roi_work/web_annotator_final_r1/annotator.sqlite3` database. It
copied exactly 396 passing rows (99 × 4) from the immutable original database,
removed the unsuitable old patient 39, and left exactly four incomplete tasks
for the approved replacement. The migration is recorded in
`roi_work/web_annotator_final_r1/revision_migration.json`; it never writes to
the source database. Launch this revision — not the old `final` database —
with:

```bash
python run_web_annotator.py --phase final --cohort final-r1 \
  --work-dir roi_work/web_annotator_final_r1 \
  --manifest roi_work/annotations/revisions/final-r1-provisional/annotation_manifest.csv \
  --template roi_work/annotations/revisions/final-r1-provisional/roi_annotations.csv
```

The app therefore reports one patient remaining: the replacement at the
original patient-39 display position. Once all four tasks are saved, run the
canonical export from this revision database; do not export from `final`.

The replacement was annotated and the server was cleanly stopped. Direct
database verification recorded 400 total rows, 400 `pass`, zero `incomplete`,
zero `unable`, and 100 studies with all four landmarks. The revision is now
ready for canonical export and final-prior fitting; the original `final`
database remains an untouched historical record.

### `final-r1` export and prior-gate result (2026-09-21)

Canonical export completed successfully from the `final-r1` database. It wrote
`roi_work/annotations/revisions/final-r1/roi_annotations_canonical.csv` with
400 rows after revalidating every saved pixel/SOP point against the DICOM
geometry. Its metadata sidecar and a pre-export SQLite backup are present.

Final prior fitting was then executed with that export and the revised frozen
manifest. It **failed closed**: `roi_priors.json` exists as a diagnostic
record but has `accepted: false`, so it must not be supplied to a cache
build. The coverage result is not the problem: every target has 1.000
held-out coverage and audit laterality availability is 1.000. The failure is
exactly 10 mirror errors among the 20 audit studies.

`mirror_gate_diagnostic.csv` gives the full evidence. The issue is systematic,
not limited to the replacement: all 10 left-side audit studies and all 40
left-side fit studies have `medial_joint_line.patient_x_mm >=
lateral_joint_line.patient_x_mm`; all 50 right-side studies satisfy the
current expected patient-X relation. Under the current documented LPS
convention (`medial axis = -X` for left and `+X` for right), this is a mirror
violation. Cache generation, smoke rendering, and training are deliberately
blocked while the project resolves whether these are left-knee annotations
made with an unlabelled/mirrored display, a cohort laterality-label issue, or
an erroneous canonical-side convention. Do not bypass or relax the gate.

The authoritative cache64 `ids_shard0.npy` and `ids_shard1.npy` files were
also not found under the repository or local competition data root, so a full
cache build has a second independent dependency even after the mirror issue
is resolved.

### Left-knee joint-line correction workflow (2026-09-21)

The correction app now exposes the DICOM laterality and display orientation at
the top of every task. For a coronal joint-line task it states, for example,
`LEFT KNEE. In this displayed coronal image, MEDIAL is on the LEFT side.` The
instruction is calculated from that exact series' DICOM patient-coordinate
mapping; it is withheld rather than guessed for a non-coronal or too-oblique
image. This guidance is specifically for `medial_joint_line` and
`lateral_joint_line`; PF and suprapatellar tasks are centreline landmarks. See
`../web_annotator/frontend.md` for the UI contract.

`queue_left_joint_line_corrections.py` created a pre-change SQLite backup and
reopened exactly the 100 left-knee joint-line tasks (50 studies × 2). The
other 300 landmark records are untouched. Reopening never destroys the old
point: it is retained in the append-only `annotation_events` history and the
app renders it amber while the candidate correction is red. On save, the new
point becomes the live green point and the previous point remains auditable.
The queue report is
`roi_work/web_annotator_final_r1/left_joint_line_correction_queue.json`.

The live verification immediately after queueing was 300 `pass`, 100
`incomplete`, zero `unable`, and all 100 incomplete tasks had a recoverable
historical marker. Do not use the earlier canonical export or rejected priors
after these corrections. Once the 100 tasks are saved, repeat canonical export
and final-prior fitting from `final-r1`; only an accepted new priors JSON may
proceed to smoke/cache construction.

### Post-correction gates (2026-09-21)

All queued corrections were saved before the server exited: direct SQLite
verification found 400 `pass`, zero `incomplete`, and zero `unable` rows, with
100 passing rows for each landmark. A new canonical export then succeeded and
replaced the earlier pre-correction export at
`roi_work/annotations/revisions/final-r1/roi_annotations_canonical.csv`.

The repeated final-prior fit is accepted. Its output is
`roi_work/annotations/revisions/final-r1/roi_priors.json` and records the
frozen 80 fit / 20 audit IDs, 1.000 laterality availability, 1.000 audit
coverage for MCL, lateral meniscus, lateral OA, PF OA, and synovitis, and zero
mirror errors. This supersedes the earlier rejected diagnostic priors file.

The deterministic real-pixel smoke build also passed and wrote
`roi_work/annotations/revisions/final-r1/roi_smoke.png`. It was built twice
from one real study and its arrays/masks were byte-identical. The smoke sheet
may contain black windows labelled invalid when a selected triplet crosses a
series boundary or its physical crop exceeds the 20% padding limit; this is
the intentional zero-fill/mask failure contract, not a silent image failure.
Full-shard validation will measure whether these events remain below the
cohort-wide coverage gate.

The next runnable stage is cache construction, but it still requires the
authoritative cache64 `ids_shard0.npy` and `ids_shard1.npy` vectors. They were
not found under the repository or local competition data root. Do not invent
or regenerate them from `train.csv`: Arm C requires exact equality with the
existing global-cache row order.

### Cache64 recovery and rebuild contract (2026-09-21)

The repository contains the missing global-cache creator:
`kaggle/cache64/build_cache64.py` for shard 0 and the historical shard-1 copy
under `kaggle/cache64_s1/`. The actual cache64 rule is fully deterministic and
does not depend on labels: read `train.csv` in its existing row order and take
alternating rows, `rows[0::2]` for shard 0 and `rows[1::2]` for shard 1. The
global cache includes all 4,407 studies; gold studies are later excluded by
the training split, not from the cache.

`build_cache64.py` was updated without changing its pixel recipe so it accepts
`--competition-root`, `--output-dir`, `--shard`, `--workers`, and `--ids-only`.
It still uses the verified 64×336 max-span recipe: first unused matching
series, physical slice ordering, modality LUT, MONOCHROME1 handling, 2–98%
normalization, 140 mm centre crop, and `INTER_AREA` resize. The full build
writes `vols_shard<S>.npy`, `masks_shard<S>.npy`, `ids_shard<S>.npy`, and a
metadata/determinism report.

Local recovery is feasible on the current volume: 181 GiB was free when the
work began. The two global shards are about 31.8 GB total and the two ROI
shards about 29.8 GB total, leaving headroom for raw data and metadata. An
IDs-only run produced 2,204 shard-0 and 2,203 shard-1 UIDs and independently
verified exact equality with the respective alternating `train.csv` rows. The
full shard-0 global cache build was then started in `roi_work/cache64/`;
never run the ROI builder against IDs alone because Arm C also requires the
matching global volumes and masks.

`test_roi_revision.py` covers direction-token handling, explicit conflict
rejection, and QC-list composition. The preparation run also asserts 100
unique IDs, 50/50 laterality, 80/20 fit/audit, 80/20 complete/imperfect
coverage, a 400-row template, and exact manifest/ID-array equality.

## Local build status — 2026-09-21

The missing cache was not an irrecoverable external artifact. The repository
contains the deterministic builders and the raw competition data is available
locally under `../data-root`.

### Completed global-context cache64

`kaggle/cache64/build_cache64.py` was made usable from a local command line
without changing its image recipe. Its authoritative row order is the raw
`train.csv` alternating split: shard 0 is `rows[0::2]` (2,204 studies), shard
1 is `rows[1::2]` (2,203 studies). The resulting files are:

```text
roi_work/cache64/vols_shard0.npy     [2204,64,336,336] uint8
roi_work/cache64/masks_shard0.npy    [2204,64] uint8
roi_work/cache64/ids_shard0.npy
roi_work/cache64/meta_shard0.json
roi_work/cache64/vols_shard1.npy     [2203,64,336,336] uint8
roi_work/cache64/masks_shard1.npy    [2203,64] uint8
roi_work/cache64/ids_shard1.npy
roi_work/cache64/meta_shard1.json
```

Both shards completed with no failed studies. The IDs were checked against
the raw alternating `train.csv` order; the minimum/median valid global slices
are 42/64 in both shards.

### Accepted full ROI cache

The existing builder was run with the accepted `final-r1` priors and the
exact global-cache IDs. It produced these files in `roi_work/cache/`:

```text
roi_<target>_shard0.npy              [2204,4,3,336,336] uint8
roi_<target>_masks_shard0.npy        [2204,4] uint8
roi_ids_shard0.npy
roi_provenance_shard0.csv.gz
roi_meta_shard0.json
roi_<target>_shard1.npy              [2203,4,3,336,336] uint8
roi_<target>_masks_shard1.npy        [2203,4] uint8
roi_ids_shard1.npy
roi_provenance_shard1.csv.gz
roi_meta_shard1.json
```

Both shards have no study-level failures and passed deterministic rebuilds.
The full validator passed: every ROI ID exactly equals its global-cache ID,
all arrays/masks have the required shapes and dtypes, valid triplets have
strictly increasing physical positions, provenance is complete, p95 padding is
12.59%/12.66% (shards 0/1), and every target clears the 90% two-window gate.

### Synovitis redesign audit and approved production policy

`kaggle/roi_cache/audit_synovitis_policy.py` performs a geometry-only audit; it
does not read pixels or modify a production cache. The shard-0 report is
`roi_work/synovitis_policy_audit/shard0_geometry_report.json`. The current
fixed overlapping-triplet policy achieved 34.17% at-least-two-window coverage.

The radiologist-approved policy preserves axial and sagittal fluid-sensitive
planes and the full `100 x 100 mm` crop. It makes two documented changes:

1. For each series, enumerate every valid consecutive three-slice group and
   select the two whose middle slices are physically nearest the predicted
   landmark. The groups are distinct and physically ordered; no slice is
   clamped or duplicated.
2. In the sagittal plane only, shift the crop centre 25 mm inferiorly. This
   retains a 25 mm superior margin for an expanded suprapatellar pouch; the
   axial crop stays centred on the landmark. A source acquisition that still
   exceeds 20% padding is deliberately masked rather than padded into a
   misleading complete-looking crop.

The chosen 25 mm policy achieved 97.60% two-window synovitis coverage in
shard 0 and 97.00% in shard 1. The radiologist explicitly rejected the 30 mm
option because it left insufficient superior margin for a potentially expanded
pouch. The historical 30 mm audit is retained as evidence, not as production
configuration.

`kaggle/roi_cache/render_synovitis_policy_review.py` created three review
sheets in `roi_work/synovitis_policy_audit/review_sheets/`: routine cases,
cases rescued by the asymmetric crop, and remaining invalid cases. Each row
shows the source sagittal image plus 0, 20, 25, and 30 mm inferior shifts; the
crosshair is the predicted suprapatellar landmark. The implementation is in
`TARGET_SPECS['synovitis']`: `nearest_distinct_triplets` and sagittal LPS
offset `(0, 0, -25)`. Valid provenance now records both landmark and actual
crop-centre XYZ values, so the shift remains auditable.

### Merged Kaggle ROI dataset

The original sharded cache remains in `roi_work/cache/` as the validated build
record. A complete Kaggle-ready merged copy is in `roi_work/cache_merged/`. It
contains one array and mask per target, all with shape respectively
`[4407,4,3,336,336]` and `[4407,4]`, plus `roi_ids.npy`,
`roi_provenance.csv.gz`, `roi_meta.json`, and `roi_priors.json`.

The merge order is **shard 0 followed by shard 1**, deliberately matching Arm
C's existing global-cache row order. It is not raw alternating `train.csv`
order. `kaggle/roi_cache/merge_roi_cache.py` uses chunked memmaps, verifies
every copied array chunk against its source, merges provenance, and writes the
merged metadata. It was run successfully for all 4,407 studies (88,140
provenance rows).

`kaggle/armc_roi/train_armc_roi_kaggle.py` now supports either format. When a merged
`roi_ids.npy` is attached, it requires exact equality to global cache64 IDs
concatenated as shard 0 then shard 1 before it reads any ROI sample.
`kaggle/armc_roi/check_roi_cache.py` recognizes and validates the same format.
For Kaggle, attach the `cache_merged` directory as the ROI dataset and the
existing cache64/labels dataset as the global input; run `check_roi_cache.py`
before training.

The primary Arm C ROI training recipe is now a strict Arm 13 extension:
CoAtNet-RMLP-2 at 384 px, required Arm 13 checkpoint warm start, two ROI-only
warm-up epochs, and conservative joint fine-tuning. Training samples two of
the four cached windows per target for memory, while validation and inference
use all four. See `../armc_roi_training/architecture.md`; do not infer the
current training policy from older ConvNeXt/Arm 12 planning notes.

An all-in-one archive of the available cache and Arm C scripts is at
`roi_work/armc_kaggle_upload.zip` (57 GiB, integrity-checked). It includes a
manifest explicitly noting that `train_pool.csv` and `gold_test.csv` were not
present locally and must be added before training.

## Remaining human gates

- Landmark correctness requires the 20 manual annotations and visual review;
  accepted priors do not yet exist for phases that have not run the fit gate.
- Fixed crop sizes remain hypotheses until LOO and held-out coverage pass.
- Canon/Toshiba and unknown-side studies need deliberate post-fit visual
  review.
- Run a multi-vendor mini-cache before committing roughly 27.8 GiB of outputs.
- Never use pathology labels or reports to select landmark studies or fit
  geometry priors.

The exact run order is in `kaggle/roi_cache/README.md`.
