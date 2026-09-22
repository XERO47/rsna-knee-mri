# Web annotator: backend architecture, persistence, and API contract

Source: `REMOTE_ROI_ANNOTATION_APP_SPEC.md` (sections 6–11, 13–14) and
`WEB_ANNOTATOR_REFERENCE.md` (decision summary, end-to-end data flow, file
contracts, function/class inventories, test inventory, operational incidents).

## 6.1 Technology

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

## 6.2 Proposed file layout

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

As implemented, two more modules exist at the repository-root/package level:
`single_user.py` (inside the package) and `run_web_annotator.py` (repository
root launcher) — see below.

## 6.3 Module responsibilities

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

## Non-obvious implementation facts

**Uvicorn signal handling.** `Server.capture_signals()` restores the original
(default) signal disposition and re-raises the captured `SIGINT`/`SIGTERM`
immediately after `uvicorn.run()` returns, which kills the process before any
further Python code in the calling function can execute. Any "on shutdown, do
X" logic for this app must live in the FastAPI `lifespan` context manager's
code *after* `yield`, which uvicorn awaits to completion as part of its own
graceful ASGI shutdown sequence (visible in logs as `Application shutdown
complete`) *before* it re-raises the signal. This is why `cli.py`'s `serve`
backup happens inside `app.py`'s ASGI `lifespan`, not after `uvicorn.run()`
returns.

**Geometry inverse bug (corrected 2026-09-21).** `roi_geometry.SliceGeometry.
patient_to_pixel` was originally a direct dot-product projection, not a true
matrix inverse of `pixel_to_patient`. For a real `ImageOrientationPatient`
that is only approximately orthogonal (allowed by the existing 1° gate in
`SliceGeometry.from_header`), the pixel→patient→pixel round-trip error grew
roughly linearly with distance from the slice's `IPP` corner, occasionally
exceeding the spec's `1e-5` px tolerance at image center (empirically up to
~1.75e-4 px). This reproduces the desktop annotator's identical acceptance
gate exactly — the web app does not loosen or tighten it. The fix solves the
two in-plane DICOM basis vectors together with `numpy.linalg.lstsq`, the true
inverse of `pixel_to_patient` for both exact and permitted rounded
orientations; it leaves every existing saved coordinate unchanged and retains
the strict `1e-5 px` safety gate. Covered by
`test_near_orthogonal_header_uses_a_true_in_plane_inverse`.

## End-to-end data flow

```text
roi_work/pilot/pilot_manifest.csv + pilot_annotations.csv (frozen, from build_roi_cache.py select-pilot)
        |
        +--> cli.command_init_cohort --> cohorts + studies rows (SHA-256 hash pinned)
        |
        +--> cli.command_create_reader --> annotators row + one-time raw token
        +--> cli.command_assign --> assignments rows + pre-created 'incomplete' annotations rows
        |
        +--> cli.command_preflight --> re-resolves every study/landmark's reference series
        |                              and decodes one real pixel array per unique series
        |
        +--> cli.command_serve --> app.create_app(config, db, science, cohort_name) --> uvicorn.run
                    |
                    +--> browser: GET /login?token=... --> session cookie --> GET /
                    +--> browser: GET /api/task --> app._reader_state --> resolved task bundle
                    +--> browser: GET /api/slice --> services.ScientificService.resolve_task
                    |                                 + .render_slice_png (bounded LRU cache)
                    +--> browser: POST /api/annotations/save
                                    --> services.compute_and_validate_point (server-computed XYZ,
                                        round-trip validated, never trusts the browser's coordinate)
                                    --> database.Database.save_point (transactional, versioned,
                                        idempotent, appends annotation_events row)
        |
        +--> cli.command_export_reader --> one CSV per reader + .meta.json (atomic, 12-column schema)
        +--> cli.command_agreement_report --> per-landmark 3D distance CSV + summary JSON (no threshold)
        +--> cli.command_adjudicate --> one adjudications row per overlap study/landmark
        +--> cli.command_canonical_export --> combines adjudicated overlap + single-reader rows
        |         (re-verifies every stored point's round trip; runs the same structural checks
        |          build_roi_cache.command_fit_priors runs, but never runs fit-priors itself)
        |
        +--> build_roi_cache.py fit-priors (run manually, unchanged, authoritative gate)
```

## 7. Persistence model

SQLite is the source of truth during annotation. Flat CSV is an export format,
not the concurrent write store.

### 7.1 Required logical tables

#### `cohorts`

- internal ID; name and phase (`pilot` or `final`); absolute manifest path
  recorded for audit; manifest SHA-256; annotation-template SHA-256; state:
  `open` or `closed`; created/closed timestamps.

#### `studies`

- cohort ID; StudyInstanceUID; fixed display order; laterality and source;
  frozen `prior_partition` where present; manifest metadata needed for audits.

#### `annotators`

- internal stable ID; display name; token hash; active flag; created/revoked
  timestamps.

#### `assignments`

- cohort ID; annotator ID; StudyInstanceUID; reader-specific display order;
  assignment group such as `overlap` or `single_reader`.

#### `annotations`

- cohort, annotator, study, and landmark composite identity; SeriesInstanceUID
  and SOPInstanceUID; row/column source pixels; patient XYZ millimetres;
  laterality; state: `incomplete`, `pass`, or `unable`; unable reason and
  notes; monotonically increasing revision; created and updated timestamps.

The unique key is `(cohort_id, annotator_id, StudyInstanceUID, landmark)`.

#### `annotation_events`

Append-only history containing create, replace, unable, reopen, adjudication,
and export events. It stores the before/after values or a complete versioned
snapshot, actor, timestamp, and request ID.

#### `adjudications`

- cohort, study, and landmark identity; chosen source reading or separately
  entered adjudicated reading; reviewer; reason/notes; timestamp and source
  revisions.

### 7.2 Transaction rules

- A save and its audit event occur in one transaction.
- Enable foreign keys, WAL journal mode, and a reasonable busy timeout
  (`busy_timeout=5000` as implemented).
- Never hold a write transaction open during DICOM reading or PNG encoding.
- Use optimistic revision checking so a stale browser tab cannot overwrite a
  newer point.
- Return a conflict response and reload the saved record when revisions differ.
- All timestamps are UTC ISO 8601 with timezone information.

As implemented: one SQLite database file (WAL mode) is the only mutable state;
a single process-local `threading.RLock` serializes every write so two
radiologists hitting the single Uvicorn worker at once cannot lose an update,
and an explicit `(cohort, annotator, request_id)`-keyed idempotency table
makes retried saves safe.

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
- Backups are small; never copy or back up the 532 GB DICOM tree as part of
  this application.

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

- Use opaque server task IDs at the browser boundary where practical. As
  implemented: `encode_task_id`/`decode_task_id` produce a symmetric,
  stateless base64url JSON blob carrying only `{study_uid, landmark}` — it is
  *not* a capability token; authorization comes from `_authorize_task`'s DB
  lookup, not from the encoding being hard to guess.
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
Cloudflare Quick Tunnel:

```bash
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

As implemented (`WEB_ANNOTATOR_REFERENCE.md`): the tunnel is intentionally a
second explicit process. Only the origin in the printed local login link is
replaced with the HTTPS tunnel hostname; the `/login?token=...` suffix is
unchanged. The application never starts a public tunnel itself.

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

`cli.command_preflight` implements every check above and exits `SystemExit(1)`
on failure: hash match, counts, on-disk studies, real reference-series
resolution plus one real pixel decode per unique series, assignment coverage,
writable dirs, free disk space.

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
- disconnect or rename a temporary test data root and confirm graceful
  degraded behavior.

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

## 16. Implementation sequence for the next agent

Implement in these gates; do not start with remote exposure.

1. Persistence and import: SQLite schema, migrations, cohort hash checks,
   assignments, tokens, revisions, and backups.
2. Scientific service: reuse current selection, geometry, pixel decode,
   display, and patient-coordinate conversion with no HTTP involved.
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

## Exact normal operation (as implemented)

From the `rsna-knee-mri` repository root:

```bash
source .venv_web/bin/activate
python run_web_annotator.py
```

This uses `../data-root`, the frozen pilot files under `roi_work/pilot`, the
cohort name `pilot`, port 8000, and work directory
`roi_work/web_annotator_single`. It prints the recoverable private link and
opens it after the socket becomes reachable. Ctrl-C shuts down cleanly and
writes a backup; the same command resumes. `--phase final` switches to the
frozen files under `roi_work/annotations` and cohort name `final`.

The default launcher must never call `cli.command_init_cohort` on every
restart: that command reloads the studies table. `prepare_single_user`
creates studies only for a new cohort and performs exact ID/order
verification on resume, thereby protecting existing assignments and
annotations.

The recoverable bootstrap token is stored only at
`roi_work/web_annotator_single/private_login_token.txt`, with mode `0600` and
inside the existing `roi_work/` Git ignore rule. If it is missing or does not
match the database, the launcher rotates the single account's token,
invalidates old sessions, and writes a replacement.

## File contracts

### `config.py`

Immutable `AppConfig` dataclass plus every numeric/placeholder default the
spec's section 17 says must not be silently guessed (session lifetime, backup
rotation depth, slice-cache bounds, disk-floor, decode concurrency).
`AppConfig.from_work_dir` derives `db_path`/`backup_dir`/`log_dir` from one
`work_dir` so every other module only ever receives one `AppConfig`, never
raw paths it has to reconstruct.

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `UNABLE_REASONS` | none | fixed 5-tuple | The exact fixed reason list from spec section 5.4; shared with `schemas.UnableRequest` and the UI. |
| `AppConfig` | validated paths/ints | immutable record | Single config object threaded through `cli.py`/`app.py`/`services.py`. |
| `AppConfig.from_work_dir` | competition root, work dir, overrides | `AppConfig` | Derives `db_path`/`backup_dir`/`log_dir`. |
| `AppConfig.ensure_directories` | none | side effect | Creates work/backup/log dirs; called once per CLI invocation. |

### `auth.py`

Three functions, no state. High-entropy tokens (`secrets.token_urlsafe`)
hashed with plain SHA-256 (a slow password hash is unnecessary — and would
only slow down legitimate logins — for a value with ~256 bits of entropy);
only the hash is ever persisted, matching spec section 4.3.

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `generate_token` | none | 256-bit URL-safe string | Bootstrap and session tokens; called only by `database.py`. |
| `hash_token` | raw token | hex SHA-256 | The only form ever written to SQLite. |
| `tokens_equal` | two strings | bool | `hmac.compare_digest` wrapper; currently unused by the hash-based lookup flow but kept for any future direct-token-compare path. |

### `database.py`

Single `Database` class wrapping one `sqlite3.Connection` (WAL,
`foreign_keys=ON`, `busy_timeout=5000`, `isolation_level=None` so transactions
are opened/closed explicitly via the internal `_Transaction` context manager).
Every write method takes the full set of validated values it needs and
returns the committed row as a plain `dict`; callers never see raw SQL.
`RevisionConflict`, `ManifestHashMismatch`, and `CohortClosed` are the three
exceptions the HTTP and CLI layers translate into user-facing responses.

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `SCHEMA_SQL` | none | DDL string | All 9 tables + indexes; executed once via `executescript`. |
| `now_iso` | none | UTC ISO-8601 string | Every `created_at`/`updated_at`/`expires_at` column. |
| `RevisionConflict` | current row dict | exception carrying `.current` | Raised by `save_point`/`mark_unable` on a stale `expected_revision`; `app.py` returns HTTP 409 with `.current` embedded. |
| `ManifestHashMismatch` | none | exception | Raised by `create_or_resume_cohort` on a hash mismatch; `cli.command_init_cohort` turns it into `SystemExit`. |
| `CohortClosed` | none | exception | Raised by `save_point`/`mark_unable` when `cohort_state != 'open'`. |
| `Cohort` | sqlite row | frozen dataclass | Read-only view of one `cohorts` row. |
| `Database.__init__` | db path | opens/pragmas connection, calls `_init_schema` | One connection for the process's lifetime. |
| `Database._init_schema` | none | DDL applied, version checked | Refuses to start against a schema from a different `SCHEMA_VERSION`. |
| `Database.backup` | backup dir, reason, max keep | new `.sqlite3` file path | Uses `sqlite3.Connection.backup()`; rotates oldest files beyond `max_backups`. |
| `Database.create_or_resume_cohort` | name, phase, manifest/template path+hash | `Cohort` | Insert-or-verify; never silently accepts a changed hash. |
| `Database.get_cohort` / `close_cohort` | cohort name | `Cohort` / none | `close_cohort` is the only way `cohorts.state` becomes `'closed'`. |
| `Database.replace_studies` | cohort id, manifest records | none | Wipes and reloads `studies` in manifest order, preserving `laterality`/`selection_role`/`prior_partition` and the full row as `manifest_json`. |
| `Database.list_studies` / `get_study` | cohort id (+ study uid) | rows | `get_study` looks up `laterality` for the row being saved. |
| `Database.create_annotator` | display name | `(id, raw_token)` | The database stores only its hash. |
| `Database.list_annotators` | none | all account rows | Lets single-user mode enforce that no second account exists. |
| `Database.rotate_annotator_token` | annotator id | new raw token | Used only when the private token file is missing or mismatched. |
| `Database.revoke_annotator` | display name | none | Sets `active=0` and deletes that annotator's `sessions` rows; immediate revocation. |
| `Database.authenticate_bootstrap_token` | raw token | active annotator row or `None` | Hashes then looks up; only matches `active=1` rows. |
| `Database.create_session` / `resolve_session` / `delete_session` | annotator id / raw session token | session token / annotator+session row / none | `resolve_session` checks both `annotator_active` and `expires_at`; expired/revoked behaves like "no session". |
| `Database.assign_studies` | cohort id, annotator id, study uids, group, landmarks | count of newly created assignment rows | Pre-creates one `incomplete` `annotations` row per landmark so `save_point`/`mark_unable` are always `UPDATE`s, never `INSERT`s. |
| `Database.list_assignments` | cohort id, annotator id | ordered rows | The reader's fixed navigation order. |
| `Database.save_point` | full point + `expected_revision` + `cohort_state` | committed annotation dict | Idempotency check → `RevisionConflict`/`CohortClosed` guards → `UPDATE` → `annotation_events` insert (`create` if prior state was not `pass`, else `replace`) → idempotency-key insert, all in one `BEGIN IMMEDIATE` transaction. |
| `Database.mark_unable` | reason, notes, `expected_revision`, `cohort_state` | committed annotation dict | Same shape as `save_point` but always writes `event_type='unable'` and nulls every coordinate/series field. |
| `Database.reopen` | cohort/annotator/study/landmark, actor, reason | reset-to-incomplete annotation dict | Administrative-only (CLI `reopen` subcommand); appends `event_type='reopen'` with the full prior row in `before_json`. |
| `Database.upsert_adjudication` / `list_adjudications` | cohort id (+ record) | none / rows | `upsert_adjudication` uses `ON CONFLICT ... DO UPDATE`. |
| `Database._transaction` | none | `_Transaction` context manager | `BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK`; every multi-statement write uses `with self._lock, self._transaction():`. |

### `services.py`

The only module allowed to import `roi_geometry`/`build_roi_cache`/
`annotate_roi_pilot` (via a `sys.path.insert` of `kaggle/roi_cache`, mirroring
how those modules import each other). `ScientificService` owns the one
`train_series.csv` DataFrame, a small LRU of resolved `ResolvedTask` (series
geometry) objects, and a `_BoundedByteCache` of rendered PNG bytes bounded by
*both* item count and total byte size. `compute_and_validate_point` is the
single choke point through which every save (HTTP or CLI) must pass; nothing
else in the codebase calls `SliceGeometry.pixel_to_patient` directly.

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `LANDMARKS`, `REFERENCE_SLOT` | none | re-exports | `roi_geometry.LANDMARKS` and `annotate_roi_pilot.REFERENCE_SLOT` — every other module imports from *this* module, never the underlying files directly. |
| `ReferenceSeriesUnavailable` / `DataRootUnavailable` / `RoundTripValidationFailed` / `OutOfBounds` | none | exceptions | `app.py` maps these to 422/503/400/400 respectively; `cli.py` lets them propagate as uncaught errors. |
| `_BoundedByteCache` | max items, max bytes | LRU cache object | Generic; used once, for rendered PNG bytes keyed by `(sop_uid, transform_version)`. |
| `ResolvedTask` | study uid, landmark, `SeriesGeometry`, series row | frozen dataclass | The unit everything else operates on; holds no annotator/session state. |
| `ScientificService.check_data_root_available` | none | raises or returns | The one place that checks `train_series.csv`/`train_series` exist. |
| `ScientificService.resolve_task` | study uid, landmark | `ResolvedTask` | Walks `REFERENCE_SLOT[landmark]` in order, calling `pick_series` then `read_series_geometry` on the first match; cached by `(study_uid, landmark)`. |
| `ScientificService.render_slice_png` | `ResolvedTask`, slice index | PNG bytes | Calls `annotate_roi_pilot._display_pixels` then encodes losslessly via Pillow; cached by resolved `sop_uid`. |
| `ScientificService.compute_and_validate_point` | `ResolvedTask`, sop uid, row, column | `(patient_xyz, resolved_sop_uid, slice_index)` or raises | **The** server-side trust boundary: confirms the sop_uid belongs to the resolved series, checks finiteness and image bounds, then requires the round trip within `config.ROUND_TRIP_TOLERANCE_PX`. |

### `schemas.py`

Two Pydantic request models. Pydantic's own float parsing does not by itself
reject `Infinity`/`NaN` that a client could still send as JSON (stdlib `json`
accepts those as an extension), so both pixel fields carry an explicit
`math.isfinite` validator. `request_id` is validated as a UUID string purely
so the client is nudged toward generating a fresh one per save *attempt* — the
server itself treats it as an opaque idempotency key.

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `SaveAnnotationRequest` | JSON body | validated model | `task_id`, `sop_uid`, finite `row_px`/`column_px`, `expected_revision >= 0`, UUID-shaped `request_id`. |
| `UnableRequest` | JSON body | validated model | Same shape, plus `reason: Literal[config.UNABLE_REASONS]` and an optional bounded-length `notes`. |

### `app.py`

`create_app(config, db, science, cohort_name) -> FastAPI` is the only public
entry point; everything else is a closure over those four arguments so no
route can accidentally reach a different cohort's data.

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `AnnotatorContext` | annotator id, display name, `Cohort` | slotted value object | What every route's dependency injects; never constructed from client-supplied fields. |
| `encode_task_id` / `decode_task_id` | study uid + landmark / opaque string | opaque base64url string / `(study_uid, landmark)` | Symmetric, stateless, not a secret. |
| `create_app` | `AppConfig`, `Database`, `ScientificService`, cohort name | `FastAPI` | Called once by `cli.command_serve` and once per test. |
| `_current_cohort` | none | `Cohort` | Re-fetched per request so `close-cohort` takes effect on the *next* request without restarting the server. |
| `_require_session` | `Request` | `AnnotatorContext` or raises `_Unauthorized` | The `Depends(...)` used by every reader-facing route. |
| `security_and_logging` | request, `call_next` | response | The one HTTP middleware: request-id, structured log line, security headers, default `Cache-Control: no-store`. |
| `login` / `logout` / `health` | token query / cookie / none | redirect+cookie / cleared cookie / `{"status":"ok"}` | `login` never puts the token in the redirect. `health` is intentionally unauthenticated. |
| `me` / `progress` | session | reader identity+count / patient-oriented progress | Both call `_reader_state`. |
| `get_task` | `patient_index?`, `landmark?`, session | full task bundle | Defaults to `_reader_state`'s computed resume position; reports `reference_error`/`data_volume_unavailable` instead of a 500. |
| `get_slice` | `task_id`, `index`, session | PNG bytes + `X-Slice-*` headers | Calls `_authorize_task` first, then `science.resolve_task`/`render_slice_png`. |
| `save_annotation` / `mark_unable` | validated body, session | committed annotation / 409 / 503 / 400 | `save_annotation` is the only route that calls `services.compute_and_validate_point`. |
| `index` | session cookie | the static SPA page or a 401 page | Not behind `Depends` since it must return HTML, not JSON, on failure. |
| `_authorize_task` | db, context, task_id | `(study_uid, landmark)` or raises `_TaskAuthorizationError` | Decodes the task_id, then requires an existing `annotations` row for *this session's* `(cohort, annotator)` — the actual authorization check. A forged or replayed task_id from another reader's session always 403s. |
| `_reader_state` | db, cohort, annotator id | dict with `assignments`, `annotations_by_key`, counts, `resume_patient_index`/`resume_landmark_index`, `all_complete` | Turns raw assignment/annotation rows into every progress/resume decision the API makes. |

### `cli.py`

One `argparse` subcommand per admin operation in spec section 4.2/section 21;
`build_parser()` is the single source of the CLI surface. Every subcommand
function takes only `argparse.Namespace` and is independently callable from
Python. `command_serve` is the only function that imports `uvicorn`/`app.py`
(deferred import, so `python -m ...cli --help` stays fast).

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `command_init_cohort` | `--cohort --phase --manifest --template` | cohort registered | Hashes both files, backs up any pre-existing DB first, loads the manifest into `studies`. |
| `command_preflight` | `--cohort` | pass/fail report, `SystemExit(1)` on failure | Implements every numbered check in spec section 11. |
| `command_create_reader` / `command_revoke_reader` | name (+ base url) | printed one-time link / revocation | Thin wrappers over `Database.create_annotator`/`revoke_annotator`. |
| `command_assign` | `--cohort --reader --study-list --group` | assignment count | `--study-list all` or a CSV of `StudyInstanceUID`; never computes a 40/40 split itself. |
| `command_status` | `--cohort` | per-reader progress + DB/backup size | Read-only. |
| `command_backup` | `--reason` | new backup file path | Manual on-demand backup. |
| `_annotation_row_to_export` | one annotation row, reader name | one 12-column export dict | `pass` rows get their real fields; everything else gets blanks and `qc_status` = the row's own state. |
| `_atomic_write_csv` | `DataFrame`, output path | none | `.tmp` file + `os.replace`. |
| `_verify_pass_row_round_trip` | `ScientificService`, one row | raises on mismatch | Re-resolves the series and re-runs `compute_and_validate_point` on the *stored* pixel, comparing recomputed patient XYZ within `1e-3` mm. |
| `command_export_reader` | `--cohort --reader --output` | atomic CSV + `.meta.json` | Verifies every `pass` row first, backs up the DB, writes the CSV, records an `export` audit event per annotation. |
| `command_agreement_report` | `--cohort --output` | CSV of all pairwise reader distances per study/landmark + `.summary.json` | Never emits a pass/fail verdict — only median/IQR/p95/max per landmark. |
| `command_adjudicate` | `--cohort --study --landmark --decision ...` | one `adjudications` row | `replacement` computes patient XYZ itself via `services`, never trusting a client-supplied coordinate, exactly like the HTTP save path. |
| `_canonical_rows` | db, cohort | list of export-shaped dicts, or raises | For each study/landmark: ≥2 assigned readers ⇒ require an adjudication; exactly 1 ⇒ take that reader's `pass` row directly; 0 ⇒ hard failure. |
| `_validate_canonical_structure` | `DataFrame`, expected row count | raises on any violation | Duplicates only the *non-scientific* checks from the top of `build_roi_cache.command_fit_priors` (columns, duplicate keys, four-landmarks-per-study, exact row count, all-`pass`, finite coordinates). |
| `command_canonical_export` | `--cohort --output` | canonical CSV + `.meta.json` | Runs `_canonical_rows` → `_validate_canonical_structure` → re-verifies every row's round trip → atomic write. Never runs `fit-priors` itself. |
| `command_close_cohort` / `command_reopen` | `--cohort` (+ reader/study/landmark/reviewer/reason) | cohort closed / one task reset | `close_cohort` backs up first; `reopen` is the only way to move a `pass`/`unable` row back to `incomplete`. |
| `command_serve` | `--cohort --host --port` | runs Uvicorn until signaled | Pre-start backup, builds the app via `app.create_app`, calls `uvicorn.run`. The clean-shutdown backup is **not** here — see the `lifespan` note above. |
| `build_parser` / `main` | argv | `ArgumentParser` / none | `build_parser` is the single source of every subcommand's flags. |

### `single_user.py` and `run_web_annotator.py`

| Symbol | Input | Output/behavior | Purpose |
|---|---|---|---|
| `DEFAULT_READER_NAME` | none | `Joint Radiology Review` | Fixed provenance name shown by the shared account. Record the two human radiologist names separately in project notes. |
| `TOKEN_FILE_NAME` | none | `private_login_token.txt` | Recoverable local secret inside the ignored single-user work directory. |
| `SingleUserSetup` | cohort/reader/count/link/path/created | frozen result | Complete handoff from setup to the root launcher. |
| `_write_private_token` | path and raw token | atomic mode-`0600` file | Writes through a `.tmp` file and `os.replace`; never writes inside the repository's tracked source tree. |
| `_validated_manifest` | manifest path and phase | typed DataFrame | Requires exactly 20 pilot or 100 final unique study IDs before database mutation. |
| `prepare_single_user` | config, cohort, phase, frozen files, base URL | `SingleUserSetup` | Creates a new cohort or hash-verifies an existing one; preserves study rows on resume; rejects closed/mismatched cohorts and multiple accounts; creates/repairs the one token; idempotently gives all studies to the account. |
| `REPOSITORY_ROOT` | script location | absolute repo root | Makes data/work/manifest defaults independent of the caller's current directory. |
| `_open_when_ready` | URL, host, port | opens default browser after socket accepts | Avoids opening the login link before Uvicorn is ready. |
| `_arguments` | argv | parsed pilot/final/path/port options | Keeps the zero-argument pilot path simple while retaining diagnostic overrides. |
| `main` | parsed arguments | setup → preflight → `caffeinate` → serve | The supported operator entry point. |

## Test inventory (`tests/`)

| File | Scope | Needs real data-root? |
|---|---|---|
| `_fixtures.py` | `HAS_REAL_DATA` flag + `RealDataHarness` (initializes a temp DB against the real frozen 20-study pilot manifest) | Provides the fixture. |
| `test_auth.py` | token/hash properties | No |
| `test_coordinates.py` | `compute_and_validate_point` bounds/finiteness/round-trip, using synthetic geometry (no DICOM I/O) | No |
| `test_database.py` | schema/PRAGMAs, cohort hash mismatch, assignment creation, save/replace/unable/reopen event history, revision conflicts, idempotent retry, concurrent-save race, sessions, backup rotation | No |
| `test_api.py` | full FastAPI `TestClient` over real pilot data: login/cookie/hides-token, session required everywhere, reader isolation, PNG headers + magic bytes, save computes XYZ server-side, stale-revision 409, idempotent retry, unable reason enum, static path traversal, forged task_id | Yes (auto-skips otherwise) |
| `test_resume.py` | kill the `Database`/app and rebuild from the same sqlite file mid-test; confirms resume position and that a `pass` task stays reachable/editable | Yes |
| `test_exports.py` | per-reader export schema + round-trip re-verification, full real 80-row pilot canonical export, overlap-without-adjudication failure, agreement report, manifest-hash-change refusal, `_BoundedByteCache` limits | Yes (except cache-limit tests) |
| `test_single_user.py` | exactly-one-account enforcement, automatic all-study assignment, token file mode, safe second-run reuse, missing-token rotation | No |
| `test_static_ui.py` | anatomical chip labels rather than state words, large outlined target contract, static asset cache-version bump | No |

Run everything with:

```bash
python -m unittest discover -v -s kaggle/roi_cache -p 'test_*.py'
```

(`-s kaggle/roi_cache`, not `-s kaggle/roi_cache/web_annotator/tests` without
`-t` — the top-level dir matters for this package's relative imports.)

Last known-good result: **73/73 tests passed**, including all 14 pre-existing
ROI geometry/builder tests and all web-annotator tests.

## Operational incidents (chronological)

**2026-09-21, final-cohort save incident.** After four final-cohort patients
were completed, a normal click on patient 5 returned HTTP `400` (server still
live). Root cause and fix: see "Non-obvious implementation facts" above
(`patient_to_pixel` inverse). At the correction point, the final SQLite
database had 16 saved landmarks across exactly 4 completed patients, verified
present with patient-space coordinates and audit events before restart.

**Final-cohort completion checkpoint (2026-09-21).** The first full final
annotation pass produced 396 `pass` landmarks across 99 fully completed
patients, plus 4 `unable` landmarks for patient 39. `PRAGMA integrity_check`
returned `ok`; the clean-shutdown backup is retained under
`roi_work/web_annotator_single/backups/`. Do **not** run final `fit-priors`
from this state — the scientific gate requires exactly 100 studies with all
four landmarks approved and finite, and rejects `unable`/incomplete rows.
Patient 39's four reference views were all available; it is a review task, not
an eligible exclusion. See `../roi_cache/laterality_rules.md` for how the
patient-39 series-selection conflict was subsequently investigated and
resolved via the `final-r1` revision database.

## What is *not* implemented yet

- No HTTP admin dashboard (deliberately — spec section 4.2 allows CLI-only).
- No named/stable Cloudflare Tunnel automation (spec section 17 leaves this to
  the operator; only the Quick Tunnel command is documented).
- `ScientificService.cache_stats()` is not exposed through any route or CLI
  command yet; wire it into `cli.command_status` or a future `/api/health`
  detail if operational visibility into cache pressure is needed.
- No automatic creation of the frozen final 100-study manifest/template. Run
  the existing scientific `select-cohort` gate first; after those files exist,
  `python run_web_annotator.py --phase final` gives the single shared account
  all 100 studies automatically.
