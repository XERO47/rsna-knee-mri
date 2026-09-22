# Remote ROI landmark annotation application specification

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

This is not application code. It is the contract an implementation agent must
follow. Before implementation, also read:

- `references/DATASET_REFERENCE.md`
- `references/ROI_CACHE_REFERENCE.md`
- `kaggle/roi_cache/README.md`
- `kaggle/roi_cache/roi_geometry.py`
- `kaggle/roi_cache/build_roi_cache.py`
- `kaggle/roi_cache/annotate_roi_pilot.py`

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

## 5. Browser UI specification

Use one static HTML page, one CSS file, and one small JavaScript file. No
frontend build system is needed. The visual design can resemble the current
Matplotlib window, but a Matplotlib GUI cannot itself be delivered to a remote
browser. Use an HTML canvas or image element with an accurately positioned
overlay marker.

### 5.1 Header

Display:

- cohort name: `Pilot` or `Final`;
- reader display name;
- `Patient X/Y` where Y is that reader's assigned patient count;
- `N patients remaining`;
- `Landmark 1/4`, `2/4`, `3/4`, or `4/4`;
- a human-readable landmark name;
- four task-state indicators for the current patient: incomplete, saved, or
  unable-to-annotate;
- save state: `Unsaved point`, `Saving…`, `Saved`, or `Offline—not saved`.

Do not show `1/80` or `1/400` as the primary progress measure. Do not call a
study a task. Full UIDs may be available in a collapsed technical-details panel
for troubleshooting, but not as the main title.

### 5.2 Image area

- Show one grayscale slice centered on a dark neutral background.
- Preserve the source aspect ratio.
- Do not mirror, rotate, or horizontally flip the image.
- Show the selected point as a high-contrast crosshair that does not obscure
  the anatomy.
- Show slice position as `Slice X/Y`.
- Provide a range slider and previous/next slice buttons.
- Keyboard left/right changes the slice. Changing slice clears any unsaved
  point and asks for confirmation if necessary.
- Provide simple zoom-in, zoom-out, reset, and pan controls. Zooming must not
  alter saved source pixel coordinates.
- Do not show an old marker after changing slices.

### 5.3 Click mapping

The client must map a pointer location through the displayed image transform
back to fractional source pixel coordinates. Mapping must account for CSS
scaling, aspect-ratio letterboxing, zoom, pan, and high-DPI screens. A click
outside the actual image is rejected.

The save request includes the task identity, selected SOP UID or opaque slice
identifier, and fractional row/column. The server confirms:

1. the task is assigned to the authenticated reader;
2. the study, landmark, series, and slice are the server's current resolved
   entities, not arbitrary client values;
3. row and column are finite and inside source bounds;
4. pixel-to-patient-to-pixel error is at most `1e-5` pixels;
5. the task revision has not changed since it was loaded.

Only then may the record be committed.

### 5.4 Controls

Required controls:

- Previous patient
- Previous landmark
- Save and next landmark
- Next landmark
- Next patient
- Replace saved point
- Unable to annotate
- Reset view

`Save and next landmark` must be the primary action. The button is disabled
until a valid point exists. Replacing a saved point requires an explicit second
save and creates an audit event; it must not silently overwrite history.

`Unable to annotate` requires one of a short fixed list plus optional notes:

- anatomy not confidently visible;
- reference series unsuitable;
- corrupt/unreadable image;
- laterality concern;
- other.

These records remain incomplete for scientific export until reviewed. They
must not be exported as `qc_status=pass` and must not silently become zero
coordinates.

### 5.5 Instructions

Each landmark panel should show a short, agreed radiologist instruction and an
optional reviewed example image. The UI must not improvise clinical guidance.
The examples and wording require sign-off from both radiologists before the
pilot begins.

### 5.6 Completion screen

When all assigned work is saved or explicitly flagged, show:

- saved patient count;
- unresolved/flagged task count;
- a list of patients needing review;
- confirmation that data is already stored on the Mac;
- a safe sign-out button.

Do not rely on a final `Submit all` operation. Every individual save is already
durable.

## 6. Server architecture

### 6.1 Technology

Recommended stack:

- Python version supported by the existing project;
- FastAPI for the local HTTP API and static-file serving;
- Uvicorn with one worker, bound to `127.0.0.1`;
- SQLite in WAL mode for concurrent readers and short transactional writes;
- the current pydicom/OpenCV/NumPy geometry and pixel functions;
- plain HTML, CSS, and JavaScript for the browser.

One worker is intentional: the workload is two human users, and a single
process avoids duplicate in-memory caches and cross-process coordination. DICOM
decode should run outside the async event loop or the endpoints should be
ordinary synchronous handlers.

### 6.2 Proposed file layout

The implementation agent should create a self-contained directory without
modifying the scientific primitives unnecessarily:

```text
kaggle/roi_cache/web_annotator/
    README.md
    app.py
    config.py
    database.py
    services.py
    schemas.py
    cli.py
    static/
        index.html
        app.css
        app.js
    tests/
        test_api.py
        test_auth.py
        test_coordinates.py
        test_database.py
        test_exports.py
        test_resume.py
```

The names may change slightly, but separation of HTTP, persistence, scientific
services, static UI, CLI, and tests must remain clear.

### 6.3 Module responsibilities

| Module | Input | Output/responsibility |
|---|---|---|
| `config.py` | CLI arguments and environment secrets | Validated immutable paths, cohort, bind address, port, cache limits |
| `database.py` | SQLite path and domain records | Schema creation/migration, WAL settings, transactions, revisions, audit events, backup |
| `services.py` | Authenticated task identifiers | Existing series selection/geometry, slice decode, PNG render, patient-coordinate conversion |
| `schemas.py` | API payloads | Strict finite/bounded request and response validation |
| `app.py` | Configured services | Authenticated API, static UI, security headers, health endpoint |
| `cli.py` | Admin subcommands | Initialize, create users, import cohort, assign, status, backup, export, close cohort |
| `static/*` | API responses and user actions | Responsive browser annotator with exact source-coordinate mapping |

No module may independently reimplement the DICOM affine or cohort selector.

## 7. Persistence model

SQLite is the source of truth during annotation. Flat CSV is an export format,
not the concurrent write store.

### 7.1 Required logical tables

#### `cohorts`

- internal ID;
- name and phase (`pilot` or `final`);
- absolute manifest path recorded for audit;
- manifest SHA-256;
- annotation-template SHA-256;
- state: `open` or `closed`;
- created/closed timestamps.

#### `studies`

- cohort ID;
- StudyInstanceUID;
- fixed display order;
- laterality and source;
- frozen `prior_partition` where present;
- manifest metadata needed for audits.

#### `annotators`

- internal stable ID;
- display name;
- token hash;
- active flag;
- created/revoked timestamps.

#### `assignments`

- cohort ID;
- annotator ID;
- StudyInstanceUID;
- reader-specific display order;
- assignment group such as `overlap` or `single_reader`.

#### `annotations`

- cohort, annotator, study, and landmark composite identity;
- SeriesInstanceUID and SOPInstanceUID;
- row/column source pixels;
- patient XYZ millimetres;
- laterality;
- state: `incomplete`, `pass`, or `unable`;
- unable reason and notes;
- monotonically increasing revision;
- created and updated timestamps.

The unique key is `(cohort_id, annotator_id, StudyInstanceUID, landmark)`.

#### `annotation_events`

Append-only history containing create, replace, unable, reopen, adjudication,
and export events. It stores the before/after values or a complete versioned
snapshot, actor, timestamp, and request ID.

#### `adjudications`

- cohort, study, and landmark identity;
- chosen source reading or separately entered adjudicated reading;
- reviewer;
- reason/notes;
- timestamp and source revisions.

### 7.2 Transaction rules

- A save and its audit event occur in one transaction.
- Enable foreign keys, WAL journal mode, and a reasonable busy timeout.
- Never hold a write transaction open during DICOM reading or PNG encoding.
- Use optimistic revision checking so a stale browser tab cannot overwrite a
  newer point.
- Return a conflict response and reload the saved record when revisions differ.
- All timestamps are UTC ISO 8601 with timezone information.

### 7.3 Resume behavior

At login, resume at the first incomplete or flagged task according to the
reader's fixed assignment order. Previously saved tasks remain editable but do
not count as incomplete. The browser may remember only harmless UI state such
as zoom; scientific state always comes from SQLite.

After any restart, the server must recover solely from the database, frozen
manifest, and source DICOM tree. No in-memory queue may be necessary for
correctness.

### 7.4 Backup

- Before every server start, make a consistent SQLite backup if the database
  already exists.
- Make another timestamped backup on clean shutdown and before each export.
- Keep a bounded rotation, for example the latest 20 backups, unless the owner
  requests longer retention.
- Store the active database and backups under `roi_work/web_annotator`, which
  must remain Git-ignored.
- Backups are small; never copy or back up the 532 GB DICOM tree as part of this
  application.

## 8. HTTP/API contract

Exact route names are flexible, but the capabilities and authorization rules
are not.

| Method/capability | Purpose | Important response/input |
|---|---|---|
| Bootstrap login | Exchange private token for session | Redirect to clean URL; no token in response body |
| Logout | Revoke browser session | Clear cookie |
| Current user | Header identity and cohort | Reader name, cohort, assigned count |
| Progress | Patient-oriented completion | completed/remaining patients, per-landmark states |
| Task | Load assigned study/landmark | Opaque task ID, revision, series metadata, slice count, saved point |
| Slice image | Render one resolved physical slice | `image/png`, source rows/columns, SOP opaque ID |
| Save annotation | Validate and commit point | Stored row/column, patient XYZ, new revision |
| Mark unable | Commit reason | New revision and task state |
| Health | Local/tunnel check | No dataset details or secrets |

Rules:

- Use opaque server task IDs at the browser boundary where practical.
- All reader-specific routes derive annotator identity from the session, never
  a client-supplied annotator name.
- A reader cannot change a StudyInstanceUID in a request to escape assignment.
- The server resolves filesystem paths from validated database/manifest data;
  no route accepts a path.
- Image responses use `Cache-Control: private, no-store`.
- API responses use `Cache-Control: no-store`.
- Set CSP, `X-Content-Type-Options: nosniff`, a restrictive referrer policy,
  frame denial, and same-origin CORS behavior.
- Suppress stack traces and local paths in remote error responses. Write
  detailed errors to the local log with a request ID.

## 9. Slice rendering and performance

- Decode only the requested slice.
- Maintain a small bounded in-memory LRU cache of rendered PNG bytes keyed by
  SOP UID and display-transform version.
- A disk cache is optional. If implemented, cap it explicitly (for example at
  1 GB), make it disposable, and never confuse it with annotation state.
- Preload at most the adjacent slice after the current slice has rendered.
- Do not pre-render all studies or duplicate the DICOM tree.
- Return exact image dimensions with the task/slice metadata.
- Limit concurrent decode work so two fast sliders cannot exhaust memory.
- Cancel or ignore stale client image requests when a newer slice is selected.

The database, logs, exports, and optional PNG cache should remain tiny compared
with the dataset. A reasonable acceptance target is under 100 MB without a
disk image cache and under the configured cap with one. Startup must report the
active database size and cache size.

## 10. Exposing the Mac safely

### 10.1 Recommended simple session

Bind Uvicorn to `127.0.0.1`, not `0.0.0.0`, and expose that local port using a
Cloudflare Quick Tunnel. Cloudflare documents the form:

```text
cloudflared tunnel --url http://localhost:8000
```

This makes an outbound connection from the Mac and avoids router port
forwarding. It produces a random `trycloudflare.com` URL. Quick Tunnels are
officially intended for testing/development, the URL changes when the tunnel
restarts, and the current documented limit is 200 concurrent requests. Two
radiologists are well within that request-concurrency limit, but the owner must
expect to resend links if the tunnel restarts.

For a stable multi-day hostname, use a named Cloudflare Tunnel and access
policy instead. That requires a Cloudflare account and a domain; it is an
optional operational upgrade, not an MVP dependency.

Official references:

- [Cloudflare Tunnel getting started](https://developers.cloudflare.com/tunnel/get-started/)
- [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
- [FastAPI static files](https://fastapi.tiangolo.com/tutorial/static-files/)
- [Uvicorn deployment](https://www.uvicorn.org/deployment/)

### 10.2 Host requirements during a session

- The Mac must remain powered, awake, connected to the internet, and connected
  to the data volume.
- Keep the web server and tunnel in supervised terminal sessions or a launcher
  that reports both process states.
- Prevent sleep for the annotation window using a reversible macOS mechanism
  such as `caffeinate`; do not permanently change power settings without the
  owner's choice.
- If the data volume disconnects, enter read-only degraded mode, show a clear
  retryable error, and preserve all prior annotations.
- The launcher prints the local URL, tunnel URL, database path, cohort, and
  progress, but never prints radiologist tokens after initial creation.

### 10.3 Privacy and competition-data caution

Treat images as sensitive even if the competition states that they are
de-identified. Confirm that the competition terms permit this limited sharing
with the participating radiologists before opening the tunnel. If any source
contains real patient identifiers or institutional restricted data, do not use
a public Quick Tunnel; use an institution-approved system and access controls.

The UI should hide UIDs by default, omit reports and labels, avoid analytics,
avoid third-party JavaScript/fonts, and serve all assets locally. The tunnel
provider will relay the requested PNG bytes, so that operational fact must be
acceptable to the owner and radiologists.

## 11. Initialization and startup preflight

The implementation must provide a dry-run/preflight operation that fails before
opening a tunnel if any invariant is broken.

Preflight checks:

1. Competition root exists and contains `train_series.csv` and `train_series`.
2. Manifest and annotation template exist and match their recorded hashes on a
   resumed cohort.
3. Manifest StudyInstanceUIDs are unique.
4. The template contains exactly the four expected landmarks per study with no
   duplicate study/landmark keys.
5. Pilot phase is exactly 20 studies/80 rows or final phase is exactly 100
   studies/400 rows.
6. Every manifest study is present in `train_series.csv` and on disk.
7. Every task resolves its permitted reference series through the existing
   deterministic selector.
8. At least one slice per resolved series passes header geometry and pixel
   decode.
9. The database schema is current and consistent with the cohort hashes.
10. Assignments cover every required study at least once and do not include
    studies outside the cohort.
11. Radiologist tokens are distinct and active.
12. The database, backup directory, and log directory are writable; `data-root`
    is treated as read-only.
13. Free disk space is above a conservative configured floor.

Do not silently replace a changed manifest, reset a database, or regenerate
assignments. Require an explicit new cohort initialization.

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

## 13. Error handling

| Failure | Required behavior |
|---|---|
| Browser loses network before save response | Keep point visibly unsaved; allow retry with same request ID |
| Save committed but response lost | Idempotent retry returns the committed record, not a duplicate |
| Browser has stale revision | Reject overwrite, reload latest point, inform user |
| DICOM decode fails | Show local error ID; permit `unable to annotate`; continue other tasks |
| Reference series missing | Record unresolved task; never substitute unrelated plane |
| Tunnel restarts | Existing database remains authoritative; issue new URL if Quick Tunnel changed |
| Server/Mac restarts | Resume first incomplete task after login |
| Disk becomes low | Disable new image caching, warn administrator, keep annotation transactions working if safe |
| Database write fails | Never show `Saved`; retain unsaved marker and log locally |
| Manifest hash changes | Refuse startup for that cohort |
| Data volume disconnects | Do not mutate scientific records; retry after volume returns |

All user-visible errors should state whether the click was saved. Ambiguous
messages such as `Something went wrong` are unacceptable after a save attempt.

## 14. Testing requirements

No radiologist should use the app until all automated tests and a two-browser
smoke test pass.

### 14.1 Unit tests

- auth token hashing, login, expiry, and revocation;
- exact assignment authorization;
- physical slice order inherited from existing geometry;
- source-display-source coordinate mapping at normal, Retina/high-DPI, zoomed,
  panned, and letterboxed layouts;
- click bounds and non-finite input rejection;
- server pixel/patient round trip;
- annotation unique key and revision conflicts;
- idempotent retry;
- audit event creation on save and replace;
- atomic per-reader and canonical exports;
- exact 12-column export schema;
- manifest/template hash mismatch refusal;
- failure states do not create passing coordinates.

### 14.2 Integration tests

- initialize the real 20-study pilot in a temporary database without modifying
  current CSVs;
- authenticate two readers and prove each cannot read the other's task;
- render the known example DICOM through the HTTP endpoint and inspect the PNG;
- save a point, kill/restart the server, and confirm exact resume;
- issue two simultaneous saves and verify no lost update;
- replace an annotation and verify old/new audit history;
- export both readers and verify no duplicate rows inside either export;
- create explicit adjudications and validate the canonical pilot CSV;
- disconnect or rename a temporary test data root and confirm graceful degraded
  behavior.

### 14.3 Visual/browser smoke test

Test current Safari and Chrome, including one phone/tablet-sized viewport even
if desktop is recommended. For five studies spanning multiple vendors and both
laterality values:

1. compare browser slice against the current Matplotlib annotator;
2. confirm no rotation/mirroring difference;
3. click obvious locations near all four image corners and center;
4. compare saved source pixels and patient XYZ with desktop calculations;
5. change slice and ensure the marker clears;
6. zoom/pan and confirm the saved coordinate remains correct;
7. reload and confirm saved marker/progress returns.

### 14.4 Remote rehearsal

Before sending links to radiologists:

- run the Mac server and tunnel;
- connect from a device not on the same Wi-Fi;
- complete and replace several dummy/test annotations;
- restart both server and tunnel;
- confirm resume and create a verified backup/export;
- monitor Mac memory, disk growth, and slice latency for at least 30 minutes.

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

## 16. Implementation sequence for the next agent

Implement in these gates; do not start with remote exposure.

1. Persistence and import: SQLite schema, migrations, cohort hash checks,
   assignments, tokens, revisions, and backups.
2. Scientific service: reuse current selection, geometry, pixel decode, display,
   and patient-coordinate conversion with no HTTP involved.
3. Local API: authentication, task/progress, PNG, save, unable, and health.
4. Browser UI: patient/landmark progress, slice navigation, accurate overlay,
   save states, resume, and responsive layout.
5. Export/adjudication: per-reader CSVs, agreement report, explicit canonical
   choices, and existing fitter compatibility.
6. Test suite: unit, integration, visual comparison, restart, concurrency, and
   storage limits.
7. Local-only pilot rehearsal at `127.0.0.1`.
8. Tunnel rehearsal from an external network.
9. Two-radiologist 20-study pilot.
10. Only after the existing scientific pilot gate passes, initialize and run
    the 100-study cohort.

At every gate, preserve the current desktop annotator as a local diagnostic
fallback. Do not remove it until the web app has completed the full pilot and
its exported coordinates have been independently verified.

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
