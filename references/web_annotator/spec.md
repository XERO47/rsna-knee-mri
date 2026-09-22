# Web annotator: outcomes, clinical tasks, and export schema

Source: `REMOTE_ROI_ANNOTATION_APP_SPEC.md` (sections 1–3, 12) and
`WEB_ANNOTATOR_REFERENCE.md` decision summary. This is the contract an
implementation agent must follow; the engineering reality of how the code
implements it is in `backend.md` and `frontend.md`.

Status: implementation specification only
Project: Arm C DICOM-to-ROI cache
Host: one macOS computer holding the local RSNA data
Users: two invited radiologists and one administrator
Last reviewed: 2026-09-19

This document specifies a deliberately small web application that lets two
radiologists annotate the four ROI landmarks required by the Arm C cache. The
application runs on the project owner's Mac, reads DICOMs from `data-root`, and
is exposed through an HTTPS tunnel. Radiologists need only a modern browser and
their private link. No DICOM data, database, or annotation CSV is stored on
their computers by the application.

Before implementation, also read:

- `references/dataset/dataset_structure.md`
- `references/roi_cache/laterality_rules.md`
- `kaggle/roi_cache/README.md`
- `kaggle/roi_cache/roi_geometry.py`
- `kaggle/roi_cache/build_roi_cache.py`
- `kaggle/roi_cache/annotate_roi_pilot.py`

## Current default operating mode (as implemented)

Both radiologists now work together at the same screen and produce one agreed
point per landmark (see `WEB_ANNOTATOR_REFERENCE.md` "Decision summary").
`run_web_annotator.py` is the supported default entry point: it creates or
resumes a dedicated database under `roi_work/web_annotator_single`, enforces
exactly one account, gives that account every cohort study internally, runs
preflight, keeps macOS awake, starts Uvicorn, and opens the private link. The
multi-reader CLI, agreement, and adjudication code below remain available for
backward compatibility and tests but are not part of normal operation.

## 1. Outcome and scope

The application must provide:

1. A private browser link for each radiologist.
2. A patient-oriented progress display, not a global `1/400` task counter.
3. One MRI reference series at a time, with a slice control and accurate click
   capture.
4. Four saved landmarks for every assigned study.
5. Immediate durable saving after every accepted click.
6. Safe resume after browser closure, network loss, server restart, tunnel
   restart, Mac restart, or an abrupt process exit.
7. Strict separation of the two radiologists' records.
8. Administrator-only progress, export, backup, and adjudication preparation.
9. A final canonical CSV with the exact schema consumed by the existing
   `fit-priors` command.

The first release must not:

- train a model or build ROI caches;
- change cohort selection, DICOM geometry, series selection, or crop logic;
- expose raw DICOM downloads, local filesystem paths, reports, pathology
  labels, or competition target labels;
- depend on React, a JavaScript build tool, notebooks, cloud storage, or a
  hosted database;
- write into `data-root`;
- automatically average two radiologists' landmark positions;
- infer laterality from image appearance;
- expose the server by router port forwarding.

## 2. Existing scientific contract that must remain unchanged

### 2.1 Cohorts

The pilot contains 20 studies and 80 landmark tasks. The final cohort contains
100 studies and 400 landmark tasks. The 100-study manifest retains the original
20 pilot studies and freezes 80 studies for prior fitting and 20 studies for
the held-out audit.

The application must consume an existing manifest and annotation template. It
must not regenerate or reshuffle either one. The expected files are:

| Phase | Manifest | Empty annotation template |
|---|---|---|
| Pilot | `roi_work/pilot/pilot_manifest.csv` | `roi_work/pilot/pilot_annotations.csv` |
| Final | `roi_work/annotations/annotation_manifest.csv` | `roi_work/annotations/roi_annotations.csv` |

The manifest order is the patient order shown in the UI. Study UIDs remain the
internal identifiers, but the UI should normally display `Patient 7/20` or
`Patient 42/100`, not the full UID.

### 2.2 Four landmark tasks

The order is fixed by `roi_geometry.LANDMARKS`:

1. `medial_joint_line`
2. `lateral_joint_line`
3. `patellofemoral_center`
4. `suprapatellar_recess`

The reference-series policy must remain exactly aligned with the current local
annotator:

| Landmark | Primary reference | Permitted fallback | What is being marked |
|---|---|---|---|
| Medial joint-line center | Coronal non-fluid-sensitive | Coronal fluid-sensitive | Center of the medial tibiofemoral joint line |
| Lateral joint-line center | Coronal non-fluid-sensitive | Coronal fluid-sensitive | Center of the lateral tibiofemoral joint line |
| Patellofemoral center | Axial fluid-sensitive | Axial non-fluid-sensitive | Center of the patellofemoral articulation |
| Suprapatellar recess | Sagittal fluid-sensitive | None | Center of the suprapatellar recess |

This table reflects `REFERENCE_SLOT` in `annotate_roi_pilot.py`; the
implementation must import or reuse that single source rather than maintain a
silently divergent copy.

Before clinical use, the project owner and both radiologists must agree on a
one-page visual landmark guide using examples from the pilot. The server may
show that guide beside each task, but the guide must not change the four
machine-readable landmark names.

### 2.3 Required export schema

Every accepted annotation must eventually export these columns in this order:

```text
StudyInstanceUID
landmark
SeriesInstanceUID
SOPInstanceUID
row_px
column_px
patient_x_mm
patient_y_mm
patient_z_mm
laterality
annotator
qc_status
```

The canonical CSV passed to `fit-priors` must contain exactly one row per
`StudyInstanceUID` and landmark, exactly four unique landmarks per study, valid
finite patient coordinates, and `qc_status=pass`. Independent readings from
the two radiologists must never be placed together in that canonical CSV
because the current fitter correctly rejects duplicates.

### 2.4 Geometry and pixels

- Reuse `read_series_geometry`, `_read_pixels`, `pick_series`, and the
  `SliceGeometry` affine rather than writing new DICOM geometry.
- Sort slices by physical DICOM position through the existing geometry code,
  never by filename or `InstanceNumber`.
- Use the same modality-LUT and `MONOCHROME1` handling as the cache builder.
- Display the same deterministic 2nd-to-98th percentile grayscale transform as
  the desktop annotator unless a later reviewed display contract supersedes it.
- Send a lossless 8-bit grayscale PNG, not JPEG. A displayed slice is a view;
  the original DICOM remains authoritative.
- The browser sends its clicked source-image row and column. The server, not
  browser JavaScript, computes patient XYZ and validates the pixel-to-patient
  round trip before saving.

## 3. Recommended operating workflow for two radiologists

The application must support arbitrary assignments, but the default project
workflow should be:

### Gate A: 20-study pilot

- Both radiologists independently annotate all 20 pilot studies.
- Each radiologist therefore completes 20 patients and 80 clicks.
- Their records remain blind and separate while work is in progress.
- Generate an agreement report in patient-space millimetres after both finish.
- A project owner or radiologist adjudicates every duplicate into one canonical
  80-row pilot CSV. Never automatically average coordinates.
- Run the existing leave-one-study-out pilot gate and visually inspect the
  smoke render before creating the final cohort.

### Gate B: expand to 100 studies

- Run the existing deterministic `select-cohort` command only after Gate A
  passes. This retains the original 20 and adds 80 studies.
- The 20 retained studies provide the double-read overlap.
- Divide the 80 new studies deterministically: 40 to radiologist A and 40 to
  radiologist B, balanced across laterality, fit/audit partition, acquisition
  completeness, and manufacturer as far as practical.
- Each radiologist handles 60 patients in total: the same 20 overlap studies
  plus 40 unique studies.
- Export/adjudicate the overlap and combine it with the 80 single-reader studies
  to form exactly one canonical 400-row final CSV.

The app may also support a configuration in which both readers annotate all
100 studies, but that is not the default because it doubles workload without
being required by the current prior-fitting plan.

## 4. User roles and access

### 4.1 Radiologist

A radiologist can:

- see only their own assigned patients and annotations;
- open the current incomplete patient;
- move through physical slices in the prescribed series;
- place, replace, and save one point for the current landmark;
- navigate to previous assigned patients/tasks;
- mark a task as unable to annotate with a required reason;
- see their own completion count.

A radiologist cannot:

- see the other reader's clicks;
- export the database;
- change assignments or manifests;
- browse unassigned studies;
- request an arbitrary local file or DICOM UID;
- approve/adjudicate a final canonical reading.

### 4.2 Administrator

The administrator runs locally on the Mac and can:

- create/deactivate radiologist access tokens;
- load a frozen cohort and assignments;
- see progress and unresolved failures;
- create timestamped database backups;
- export each reader separately;
- generate a blinded agreement report;
- record adjudication decisions and export the canonical CSV;
- close a cohort to make it read-only.

The first release does not require a complex remote admin dashboard. These
operations may be local command-line commands, provided they are explicit,
validated, logged, and covered by tests.

### 4.3 Authentication

- Generate one independent, cryptographically random token per radiologist.
- Store only a strong hash of each token in the database or secrets file.
- Give each radiologist a private bootstrap link. On first request the server
  exchanges the token for a `Secure`, `HttpOnly`, `SameSite=Strict` session
  cookie and redirects to a clean URL.
- Do not print full tokens in routine request logs, save them to Git, put them
  in CSV exports, or reuse one token for both readers.
- The server must be unusable without a valid session, including image and API
  endpoints.
- Token revocation must immediately prevent new requests.

The random tunnel hostname is not authentication. The application token is
still required.

## 12. Export and adjudication

### 12.1 Per-reader export

Create one atomic CSV per reader with the existing 12-column schema. Add a
separate metadata JSON beside it containing cohort hash, reader ID, export
timestamp, row counts, incomplete/flagged counts, and application version.

An export marked complete must satisfy:

- all expected assigned study/landmark rows exist;
- every `pass` row has finite source and patient coordinates;
- all UIDs resolve back to the selected physical slice;
- server recomputation of patient XYZ matches stored values within tolerance;
- no duplicate keys;
- no unknown landmark names.

Use a temporary file, flush it, and atomically replace the target export. Never
edit the original empty template in place.

### 12.2 Agreement report

For double-read studies, report per-landmark:

- reader A and B patient XYZ;
- Euclidean 3D distance in millimetres;
- whether the source series/SOP differs;
- per-landmark median, interquartile range, 95th percentile, and maximum
  distance;
- sortable cases from largest to smallest disagreement.

Do not invent a clinical pass threshold in code. The radiologists and project
owner must review the pilot distribution and record a threshold or review rule
before final adjudication.

### 12.3 Canonical export

Each overlap landmark must have an explicit adjudication decision:

- choose reader A;
- choose reader B; or
- save a separately reviewed replacement point.

The canonical export then combines adjudicated overlap rows with the
single-reader rows. It must contain exactly 80 rows for the pilot or 400 rows
for the final cohort, exactly one row per study/landmark, and only approved
finite coordinates. Set `annotator` to the actual chosen reader or an explicit
adjudicator identifier; preserve provenance in the database and export
metadata rather than losing it.

Before returning success, run the same structural checks used by
`command_fit_priors`. The app must not itself mark the prior gate as passed.

## 15. Acceptance criteria

The MVP is complete only when:

- two distinct private links open two isolated reader accounts;
- every UI screen uses patient and per-patient landmark progress;
- all four landmarks can be saved on their prescribed physical series;
- browser/server coordinate agreement is demonstrated within tolerance;
- each save is transactional, versioned, idempotent, and immediately resumable;
- simultaneous readers cannot overwrite one another;
- raw DICOM paths and arbitrary file access are impossible through HTTP;
- the application never writes to `data-root`;
- abrupt browser, server, tunnel, and Mac restarts preserve committed work;
- per-reader exports match the existing schema;
- the canonical pilot export has exactly 80 unique approved rows;
- the canonical final export has exactly 400 unique approved rows;
- the existing `fit-priors` command accepts the canonical export structurally;
- automated, browser, and remote rehearsal tests pass;
- an administrator can back up, export, revoke access, and close the cohort;
- documentation explains one-command startup, tunnel startup, link sharing,
  progress checking, backup, export, shutdown, and recovery.

## 17. Decisions intentionally left for human sign-off

The implementation agent must not guess these:

- final radiologist display names;
- exact clinical wording and example image for each landmark;
- whether the competition terms permit use of the chosen tunnel;
- whether the final 80 new studies use the recommended 40/40 split or are all
  double-read;
- the inter-reader disagreement threshold or rule that triggers adjudication;
- named/stable tunnel versus short-lived Quick Tunnel;
- access-token lifetime and annotation-session dates.

These decisions change operations or scientific interpretation, not the core
application architecture.
