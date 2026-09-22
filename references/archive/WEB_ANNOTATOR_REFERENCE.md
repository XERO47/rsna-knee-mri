# Remote ROI annotator engineering reference

Written after the first working implementation on branch
`codex/remote-roi-annotator`, validated against the real 20-study pilot at
`../data-root` (preflight, live server, real save, real PNG byte-compare
against the desktop annotator, process-kill + restart resume, full 80-row
canonical pilot export). This is the durable handoff for future agents
working on `kaggle/roi_cache/web_annotator`; read it before touching the code,
the same way `ROI_CACHE_REFERENCE.md` is read before touching
`kaggle/roi_cache/build_roi_cache.py`.

Read first, in this order:

1. `references/REMOTE_ROI_ANNOTATION_APP_SPEC.md` — the contract.
2. `kaggle/roi_cache/web_annotator/README.md` — the operator run-book.
3. This file — how the code actually implements the contract.

## Decision summary

### Current default: one shared radiology account

Both radiologists now work together at the same screen and produce one agreed
point per landmark. `run_web_annotator.py` is the supported default entry
point. It creates or resumes a dedicated database under
`roi_work/web_annotator_single`, enforces exactly one account, gives that
account every cohort study internally, runs preflight, keeps macOS awake,
starts Uvicorn, and opens the private link. The assignment table remains an
internal authorization invariant; the operator no longer manages it.

The recoverable bootstrap token is stored only at
`roi_work/web_annotator_single/private_login_token.txt`, with mode `0600` and
inside the existing `roi_work/` Git ignore rule. If it is missing or does not
match the database, the launcher rotates the single account's token,
invalidates old sessions, and writes a replacement. The multi-reader CLI,
agreement, and adjudication code remain available for backward compatibility
and tests but are not part of normal operation.

### Exact normal operation

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

The tunnel is intentionally a second explicit process:

```bash
cloudflared tunnel --url http://localhost:8000
```

Only the origin in the printed local login link is replaced with the HTTPS
tunnel hostname. The `/login?token=...` suffix is unchanged. The application
never starts a public tunnel itself.

The default launcher must never call `cli.command_init_cohort` on every
restart: that command reloads the studies table. `prepare_single_user` creates
studies only for a new cohort and performs exact ID/order verification on
resume, thereby protecting existing assignments and annotations.

The app is ten small modules plus a static 3-file browser UI and the
repository-root launcher. Nothing here
reimplements DICOM geometry, series selection, pixel decoding, or
patient-coordinate math — every one of those calls into `roi_geometry.py` /
`build_roi_cache.py` / `annotate_roi_pilot.py` (imported, never copied). One
SQLite database file (WAL mode) is the only mutable state; a single
process-local `threading.RLock` serializes every write so two radiologists
hitting the single Uvicorn worker at once cannot lose an update, and an
explicit `(cohort, annotator, request_id)`-independent-but-request_id-keyed
idempotency table makes retried saves safe.

One non-obvious, load-bearing fact discovered while smoke-testing against
real DICOMs, and the reason `cli.py`'s `serve` backup happens inside
`app.py`'s ASGI `lifespan`, not after `uvicorn.run()` returns: **Uvicorn's
`Server.capture_signals()` restores the original (default) signal
disposition and re-raises the captured `SIGINT`/`SIGTERM` immediately after
`uvicorn.run()` returns**, which kills the process before any further Python
code in the calling function can execute. Any "on shutdown, do X" logic for
this app must live in the FastAPI `lifespan` context manager's code *after*
`yield`, which uvicorn awaits to completion as part of its own graceful ASGI
shutdown sequence (visible in logs as `Application shutdown complete`)
*before* it re-raises the signal.

A second non-obvious fact: `roi_geometry.SliceGeometry.patient_to_pixel` is a
direct dot-product projection, not a true matrix inverse of
`pixel_to_patient`. For a real IOP that is only approximately orthogonal
(allowed by the existing 1° gate in `SliceGeometry.from_header`), the
pixel→patient→pixel round-trip error grows roughly linearly with distance
from the slice's `IPP` corner. A handful of real pilot studies exceed the
spec's `1e-5` px tolerance at their exact image center on at least one
landmark's reference series (empirically up to ~1.75e-4 px at center; see
`test_exports.py`'s `_save_center_point` candidate-fallback comment for the
measured values). This reproduces the desktop annotator's identical
acceptance gate exactly — the web app does not loosen or tighten it — so it
only means an occasional click far from a series' corner on those specific
studies is rejected with a retryable "click again" response rather than
silently accepted.

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

## File contracts

### `config.py`

Immutable `AppConfig` dataclass plus every numeric/placeholder default the
spec's section 17 says must not be silently guessed (session lifetime,
backup rotation depth, slice-cache bounds, disk-floor, decode concurrency).
`AppConfig.from_work_dir` derives `db_path`/`backup_dir`/`log_dir` from one
`work_dir` so every other module only ever receives one `AppConfig`, never
raw paths it has to reconstruct.

### `auth.py`

Three functions, no state. High-entropy tokens (`secrets.token_urlsafe`)
hashed with plain SHA-256 (a slow password hash is unnecessary — and would
only slow down legitimate logins — for a value with ~256 bits of entropy);
only the hash is ever persisted, matching spec section 4.3.

### `database.py`

Single `Database` class wrapping one `sqlite3.Connection` (WAL,
`foreign_keys=ON`, `busy_timeout=5000`, `isolation_level=None` so
transactions are opened/closed explicitly via the internal `_Transaction`
context manager). Every write method takes the full set of validated values
it needs and returns the committed row as a plain `dict`; callers (mostly
`app.py` and `cli.py`) never see raw SQL. `RevisionConflict`,
`ManifestHashMismatch`, and `CohortClosed` are the three exceptions the HTTP
and CLI layers translate into user-facing responses.

### `services.py`

The only module allowed to import `roi_geometry`/`build_roi_cache`/
`annotate_roi_pilot` (via a `sys.path.insert` of `kaggle/roi_cache`, mirroring
how those modules import each other). `ScientificService` owns the one
`train_series.csv` DataFrame, a small LRU of resolved `ResolvedTask`
(series geometry) objects, and a `_BoundedByteCache` of rendered PNG bytes
bounded by *both* item count and total byte size. `compute_and_validate_point`
is the single choke point through which every save (HTTP or CLI) must pass;
nothing else in the codebase calls `SliceGeometry.pixel_to_patient` directly.

### `schemas.py`

Two Pydantic request models. Pydantic's own float parsing does not by itself
reject `Infinity`/`NaN` that a client could still send as JSON (stdlib
`json` accepts those as an extension), so both pixel fields carry an explicit
`math.isfinite` validator. `request_id` is validated as a UUID string purely
so the client is nudged toward generating a fresh one per save *attempt*
(reused only for network retries of that exact attempt) — the server itself
treats it as an opaque idempotency key, not literally requiring UUID
semantics beyond that shape check.

### `app.py`

`create_app(config, db, science, cohort_name) -> FastAPI` is the only public
entry point; everything else is a closure over those four arguments so no
route can accidentally reach a different cohort's data. Task identity at the
HTTP boundary is an opaque base64url JSON blob (`encode_task_id`/
`decode_task_id`) carrying only `{study_uid, landmark}` — it is *not* a
capability token; every route that receives one still calls
`_authorize_task`, which re-derives `(study_uid, landmark)` and requires an
existing `annotations` row for `(cohort, session's annotator_id, study_uid,
landmark)`, so a forged or replayed task_id from another reader's session
always 403s (see `tests/test_api.py::test_task_id_cannot_be_forged_to_escape_assignment`).
The single `@app.middleware("http")` adds security headers, a per-request
`X-Request-Id`, structured logging, and translates the two internal
exceptions (`_Unauthorized`, `_TaskAuthorizationError`) into 401/403 without
leaking a stack trace.

### `cli.py`

One `argparse` subcommand per admin operation in spec section 4.2/section 21;
`build_parser()` is the single source of the CLI surface. Every subcommand
function takes only `argparse.Namespace` and is independently callable from
Python (the entire test suite calls these functions directly with a
`types.SimpleNamespace` instead of shelling out, which is also the pattern to
reuse for any new admin script). `command_serve` is the only function that
imports `uvicorn`/`app.py` (deferred import, so `python -m ...cli --help`
stays fast and does not require a running app for the other 13 subcommands).

### `single_user.py` and `run_web_annotator.py`

`single_user.prepare_single_user` safely creates or resumes the frozen cohort,
refuses any database containing a second account, preserves studies and
annotations on restart, repairs a missing private token by rotation, and
idempotently gives the shared account every case. The repository-root launcher
provides the standard pilot/final paths, runs preflight, opens the browser once
the socket is ready, manages `caffeinate`, and delegates serving to the existing
CLI/FastAPI path.

### `static/{index.html,app.css,app.js}`

One page, no build step, no framework. `app.js` owns exactly one piece of
scientific-adjacent logic — the CSS-pixel → source-pixel coordinate transform
for clicks — and nothing else; it never computes a patient coordinate. It
treats every server response as authoritative and re-fetches
`/api/progress` after every save to decide whether to show the completion
screen, rather than tracking completion state itself.

Current visual contract:

- The four per-patient chips always show anatomical names, never raw states
  such as `incomplete`: medial joint-line center, lateral joint-line center,
  patellofemoral center, and suprapatellar recess.
- A completed (`pass`) chip turns green. An incomplete chip remains neutral,
  an unable chip remains yellow, and the active chip has a blue outline.
- `drawCrosshair` renders a target reticle in CSS-pixel units so Retina scaling
  cannot make it tiny: 34 px arms, a 10 px radius circle, a center dot, a
  5 px black contrast under-stroke, and a 2.5 px colored stroke. Unsaved points
  are red and stored points are green.
- Static CSS/JS URLs carry `?v=2` so a normal page refresh fetches this UI
  revision instead of reusing an older browser cache.
- Scientific click coordinates are unchanged: the reticle is display-only and
  its center remains exactly the fractional source pixel sent to the server.

## Function and class inventory: `config.py`

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `UNABLE_REASONS` | none | fixed 5-tuple | The exact fixed reason list from spec section 5.4; `schemas.UnableRequest` and `static/index.html`'s `<select>` both read from this same tuple's values. |
| `AppConfig` | validated paths/ints | immutable record | Single config object threaded through `cli.py`/`app.py`/`services.py`. |
| `AppConfig.from_work_dir` | competition root, work dir, overrides | `AppConfig` | Derives `db_path`/`backup_dir`/`log_dir` so nothing else hardcodes them. |
| `AppConfig.ensure_directories` | none | side effect | Creates work/backup/log dirs; called once per CLI invocation. |

## Function and class inventory: `auth.py`

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `generate_token` | none | 256-bit URL-safe string | Bootstrap and session tokens; called only by `database.py`. |
| `hash_token` | raw token | hex SHA-256 | The only form ever written to SQLite. |
| `tokens_equal` | two strings | bool | `hmac.compare_digest` wrapper; currently unused by the hash-based lookup flow (hash comparison already goes through a `WHERE token_hash = ?` query) but kept for any future direct-token-compare path and exercised in `tests/test_auth.py`. |

## Function and class inventory: `database.py`

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `SCHEMA_SQL` | none | DDL string | All 9 tables + indexes; executed once via `executescript`. |
| `now_iso` | none | UTC ISO-8601 string | Every `created_at`/`updated_at`/`expires_at` column uses this. |
| `RevisionConflict` | current row dict | exception carrying `.current` | Raised by `save_point`/`mark_unable` on a stale `expected_revision`; `app.py` catches it and returns HTTP 409 with `.current` embedded. |
| `ManifestHashMismatch` | none | exception | Raised by `create_or_resume_cohort` when a resumed cohort's manifest/template hash no longer matches; `cli.command_init_cohort` turns it into `SystemExit`. |
| `CohortClosed` | none | exception | Raised by `save_point`/`mark_unable` when `cohort_state != 'open'`. |
| `Cohort` | sqlite row | frozen dataclass | Read-only view of one `cohorts` row; `_row_to_cohort` builds it. |
| `Database.__init__` | db path | opens/pragmas connection, calls `_init_schema` | One connection for the process's lifetime. |
| `Database._init_schema` | none | DDL applied, version checked | Refuses to start against a schema from a different `SCHEMA_VERSION`. |
| `Database.backup` | backup dir, reason, max keep | new `.sqlite3` file path | Uses `sqlite3.Connection.backup()` (safe to call against a live DB); rotates oldest files beyond `max_backups`. |
| `Database.create_or_resume_cohort` | name, phase, manifest/template path+hash | `Cohort` | Insert-or-verify; never silently accepts a changed hash. |
| `Database.get_cohort` / `close_cohort` | cohort name | `Cohort` / none | `close_cohort` is the only way `cohorts.state` becomes `'closed'`, which every write path checks. |
| `Database.replace_studies` | cohort id, manifest records | none | Wipes and reloads `studies` in manifest order (`display_order` = row index), preserving `laterality`/`selection_role`/`prior_partition` and the full row as `manifest_json` for audit. |
| `Database.list_studies` / `get_study` | cohort id (+ study uid) | rows | `get_study` is used by `app.save_annotation` to look up `laterality` for the row being saved. |
| `Database.create_annotator` | display name | `(id, raw_token)` | The database stores only its hash. The advanced CLI shows it once; single-user mode saves it in a mode-`0600`, Git-ignored local token file for recoverability. |
| `Database.list_annotators` | none | all account rows | Lets single-user mode enforce that no second account exists. |
| `Database.rotate_annotator_token` | annotator id | new raw token | Replaces the hash, reactivates the sole account, and deletes its sessions; used only when the private token file is missing or mismatched. |
| `Database.revoke_annotator` | display name | none | Sets `active=0` and deletes that annotator's `sessions` rows, so revocation is immediate. |
| `Database.get_annotator_by_name` / `get_annotator_by_id` | key | row | Used by `cli.py` (by name) and nothing currently needs the by-id form outside internal joins. |
| `Database.authenticate_bootstrap_token` | raw token | active annotator row or `None` | Hashes then looks up; only matches `active=1` rows. |
| `Database.create_session` / `resolve_session` / `delete_session` | annotator id / raw session token | session token / annotator+session row / none | `resolve_session` checks both `annotator_active` and `expires_at` in one query; an expired or revoked session behaves identically to "no session". |
| `Database.assign_studies` | cohort id, annotator id, study uids, group, landmarks | count of newly created assignment rows | Validates every uid is in the cohort, computes `reader_order` from the cohort's `display_order`, and — critically — pre-creates one `incomplete` `annotations` row per landmark so `save_point`/`mark_unable` are always `UPDATE`s, never `INSERT`s. |
| `Database.list_assignments` | cohort id, annotator id | ordered rows | The reader's fixed navigation order; `app._reader_state` iterates this directly. |
| `Database.get_annotation` / `list_annotations_for_study` / `_for_reader` / `_for_cohort` | various | row(s) | Read paths for the task endpoint, resume logic, exports, and the agreement report respectively. |
| `Database.get_idempotent_response` | request_id | cached response dict or `None` | Checked at the very start of `save_point`/`mark_unable`, outside the write transaction. |
| `Database.save_point` | full point + `expected_revision` + `cohort_state` | committed annotation dict | The core write: idempotency check → `RevisionConflict`/`CohortClosed` guards → `UPDATE` → `annotation_events` insert (`event_type` is `'create'` if the prior state was not `'pass'`, else `'replace'`) → idempotency-key insert, all in one `BEGIN IMMEDIATE` transaction. |
| `Database.mark_unable` | reason, notes, `expected_revision`, `cohort_state` | committed annotation dict | Same shape as `save_point` but always writes `event_type='unable'` and nulls every coordinate/series field. |
| `Database.reopen` | cohort/annotator/study/landmark, actor, reason | reset-to-incomplete annotation dict | Administrative-only (CLI `reopen` subcommand); appends an `event_type='reopen'` event with the full prior row in `before_json`. |
| `Database.list_events` | annotation id | ordered `annotation_events` rows | The append-only audit trail for one annotation; exercised in `test_database.py`. |
| `Database.record_export_event` | annotation ids, actor, request_id | none | Appends one `event_type='export'` row per exported annotation; called from `cli.command_export_reader`. |
| `Database.upsert_adjudication` / `list_adjudications` | cohort id (+ record) | none / rows | `upsert_adjudication` uses `ON CONFLICT (cohort_id, study_uid, landmark) DO UPDATE`, so re-running `adjudicate` for the same key replaces the prior decision rather than duplicating it. |
| `Database._transaction` | none | `_Transaction` context manager | `BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK`; every multi-statement write uses `with self._lock, self._transaction():`. |
| `copy_database_file` | source, destination path | none | Thin `shutil.copy2` wrapper; currently unused by any code path but kept as the obvious primitive if a future "clone this cohort's DB" operation is added. |

## Function and class inventory: `services.py`

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `LANDMARKS`, `REFERENCE_SLOT` | none | re-exports | `roi_geometry.LANDMARKS` and `annotate_roi_pilot.REFERENCE_SLOT` respectively — the single source of truth every other module imports from *this* module, never from the underlying files directly, so there is exactly one place that does the `sys.path` trick. |
| `ReferenceSeriesUnavailable` / `DataRootUnavailable` / `RoundTripValidationFailed` / `OutOfBounds` | none | exceptions | `app.py` maps these to 422/503/400/400 respectively; `cli.py` lets them propagate as uncaught errors (preflight/export/canonical-export are expected to fail loudly). |
| `_BoundedByteCache` | max items, max bytes | LRU cache object | Generic; used once, for rendered PNG bytes keyed by `(sop_uid, transform_version)`. Eviction checks both bounds after every `put`. |
| `ResolvedTask` | study uid, landmark, `SeriesGeometry`, series row | frozen dataclass | The unit everything else in this module and in `app.py`/`cli.py` operates on; holds no annotator/session state. |
| `ScientificService.__init__` | competition root, cache sizes, concurrency limit | instance | Lazily loads `train_series.csv` on first use, not at construction, so constructing one is cheap even if the data volume is briefly unavailable. |
| `ScientificService.check_data_root_available` | none | raises or returns | The one place that checks `train_series.csv`/`train_series` exist; called at the top of every DICOM-touching request handler. |
| `ScientificService.resolve_task` | study uid, landmark | `ResolvedTask` | Walks `REFERENCE_SLOT[landmark]` in order, calling `roi_geometry.pick_series` then `build_roi_cache.read_series_geometry` on the first match — identical logic to `annotate_roi_pilot.ROIAnnotator._reference_row`, deliberately re-derived here rather than imported because the desktop class also carries UI state; cached by `(study_uid, landmark)`. |
| `ScientificService.render_slice_png` | `ResolvedTask`, slice index | PNG bytes | Calls `annotate_roi_pilot._display_pixels` (same 2nd/98th-percentile-per-slice transform as the desktop tool) then encodes losslessly via Pillow; cached by resolved `sop_uid`. |
| `ScientificService.cache_stats` | none | dict of ints | Exposed for `cli.command_status`-style diagnostics; not currently wired into any HTTP route. |
| `ScientificService.compute_and_validate_point` | `ResolvedTask`, sop uid, row, column | `(patient_xyz, resolved_sop_uid, slice_index)` or raises | **The** server-side trust boundary: confirms the sop_uid belongs to the resolved series (never client-trusted), checks finiteness and image bounds, then requires the `pixel_to_patient`/`patient_to_pixel` round trip to be within `config.ROUND_TRIP_TOLERANCE_PX`. Called from `app.save_annotation`, `cli._verify_pass_row_round_trip`, and `cli.command_adjudicate`'s `replacement` path — i.e. every place a pixel ever becomes a patient coordinate. |

## Function and class inventory: `schemas.py`

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `SaveAnnotationRequest` | JSON body | validated model | `task_id`, `sop_uid`, finite `row_px`/`column_px`, `expected_revision >= 0`, UUID-shaped `request_id`. |
| `UnableRequest` | JSON body | validated model | Same shape, plus `reason: Literal[config.UNABLE_REASONS]` and an optional bounded-length `notes`. |

## Function and class inventory: `app.py`

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `AnnotatorContext` | annotator id, display name, `Cohort` | slotted value object | What every route's dependency injects; never constructed from client-supplied fields. |
| `encode_task_id` / `decode_task_id` | study uid + landmark / opaque string | opaque base64url string / `(study_uid, landmark)` | Symmetric, stateless, not a secret — authorization comes from `_authorize_task`'s DB lookup, not from the encoding being hard to guess. |
| `create_app` | `AppConfig`, `Database`, `ScientificService`, cohort name | `FastAPI` | Closes over all four so every route is implicitly scoped to one cohort; called once by `cli.command_serve` and once per test in `tests/_fixtures.RealDataHarness.make_app`. |
| `create_app.<locals>._current_cohort` | none | `Cohort` | Re-fetched per request (cheap indexed lookup) so a `close-cohort` CLI call takes effect on the *next* request without restarting the server. |
| `create_app.<locals>._require_session` | `Request` | `AnnotatorContext` or raises `_Unauthorized` | The `Depends(...)` used by every reader-facing route. |
| `create_app.<locals>.security_and_logging` | request, `call_next` | response | The one HTTP middleware: request-id, structured log line, security headers, and a default `Cache-Control: no-store` for any `/api/*` response that didn't set its own. |
| `login` / `logout` / `health` | token query / cookie / none | redirect+cookie / cleared cookie / `{"status":"ok"}` | `login` is the only route that ever sees a raw token; it never appears in the redirect. `health` is intentionally unauthenticated (see README "Known limitations"). |
| `me` / `progress` | session | reader identity+count / patient-oriented progress | Both call `_reader_state` and shape a subset of it for the client. |
| `get_task` | `patient_index?`, `landmark?`, session | full task bundle | Defaults to `_reader_state`'s computed resume position; resolves the reference series via `services` and reports `reference_error`/`data_volume_unavailable` instead of a 500 when that fails. |
| `get_slice` | `task_id`, `index`, session | PNG bytes + `X-Slice-*` headers | Calls `_authorize_task` first, then `science.resolve_task`/`render_slice_png`. |
| `save_annotation` / `mark_unable` | validated body, session | committed annotation / 409 / 503 / 400 | `save_annotation` is the only route that calls `services.compute_and_validate_point`; both call `_authorize_task` before touching the database. |
| `index` | session cookie | the static SPA page or a 401 page | Deliberately not behind `Depends` since it must return HTML, not JSON, on failure. |
| `_authorize_task` | db, context, task_id | `(study_uid, landmark)` or raises `_TaskAuthorizationError` | Decodes the task_id, then requires an existing `annotations` row for *this session's* `(cohort, annotator)` — the actual authorization check. |
| `_serialize_annotation` | sqlite row | plain dict | Shared shape for `save_annotation`'s success/conflict responses. |
| `_reader_state` | db, cohort, annotator id | dict with `assignments`, `annotations_by_key`, counts, `resume_patient_index`/`resume_landmark_index`, `all_complete` | The one function that turns raw assignment/annotation rows into every progress/resume decision the API makes; both `get_task` and `progress` are thin views over its output. |

## Function and class inventory: `cli.py`

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `_config` / `_open_db` / `_print` | argparse namespace | `AppConfig` / `Database` / stdout | Shared by every subcommand. |
| `command_init_cohort` | `--cohort --phase --manifest --template` | cohort registered | Hashes both files, backs up any pre-existing DB first, loads the manifest into `studies`. |
| `command_preflight` | `--cohort` | pass/fail report, `SystemExit(1)` on failure | Implements every numbered check in spec section 11: hash match, counts, on-disk studies, real reference-series resolution + one real pixel decode per unique series, assignment coverage, writable dirs, free disk space. |
| `command_create_reader` / `command_revoke_reader` | name (+ base url) | printed one-time link / revocation | Thin wrappers over `Database.create_annotator`/`revoke_annotator`. |
| `command_assign` | `--cohort --reader --study-list --group` | assignment count | `--study-list all` or a CSV of `StudyInstanceUID`; never computes a 40/40 split itself (spec section 17). |
| `command_status` | `--cohort` | per-reader progress + DB/backup size | Read-only; used for the "check progress" run-book step. |
| `command_backup` | `--reason` | new backup file path | Manual on-demand backup. |
| `_annotation_row_to_export` | one annotation row, reader name | one 12-column export dict | `pass` rows get their real fields; everything else gets blanks and `qc_status` = the row's own state (never silently `pass`). |
| `_atomic_write_csv` | `DataFrame`, output path | none | `.tmp` file + `os.replace`, matching `annotate_roi_pilot`'s own atomic-write pattern. |
| `_verify_pass_row_round_trip` | `ScientificService`, one row | raises on mismatch | Re-resolves the series and re-runs `compute_and_validate_point` on the *stored* pixel, then compares the recomputed patient XYZ to the stored value within `1e-3` mm — catches any bit-rot or bug between save time and export time. |
| `command_export_reader` | `--cohort --reader --output` | atomic CSV + `.meta.json` | Verifies every `pass` row first, backs up the DB, writes the CSV, records an `export` audit event per annotation. |
| `command_agreement_report` | `--cohort --output` | CSV of all pairwise reader distances per study/landmark + `.summary.json` | Never emits a pass/fail verdict (spec section 12.2) — only median/IQR/p95/max per landmark. |
| `command_adjudicate` | `--cohort --study --landmark --decision ...` | one `adjudications` row | `choose_reader` requires that reader's annotation to already be `pass`; `replacement` computes patient XYZ itself via `services`, never trusting a client-supplied coordinate, exactly like the HTTP save path. |
| `_canonical_rows` | db, cohort | list of export-shaped dicts, or raises | For each study/landmark: ≥2 assigned readers ⇒ require an adjudication; exactly 1 ⇒ take that reader's `pass` row directly; 0 ⇒ hard failure. Every failure mode is collected and reported together, not one-at-a-time. |
| `_validate_canonical_structure` | `DataFrame`, expected row count | raises on any violation | Deliberately duplicates only the *non-scientific* checks from the top of `build_roi_cache.command_fit_priors` (columns, duplicate keys, four-landmarks-per-study, exact row count, all-`pass`, finite coordinates) — see the module docstring on why this doesn't touch `build_roi_cache.py`. |
| `command_canonical_export` | `--cohort --output` | canonical CSV + `.meta.json`, printed next-command hint | Runs `_canonical_rows` → `_validate_canonical_structure` → re-verifies every row's round trip → atomic write. Never runs `fit-priors` itself. |
| `command_close_cohort` / `command_reopen` | `--cohort` (+ reader/study/landmark/reviewer/reason) | cohort closed / one task reset | `close_cohort` backs up first; `reopen` is the only way to move a `pass`/`unable` row back to `incomplete`. |
| `command_serve` | `--cohort --host --port` | runs Uvicorn until signaled | Pre-start backup, builds the app via `app.create_app`, calls `uvicorn.run`. The clean-shutdown backup is **not** here — see `app.py`'s `lifespan` and the "Decision summary" note above. |
| `build_parser` / `main` | argv | `ArgumentParser` / none | `build_parser` is the single source of every subcommand's flags; every subcommand's handler is independently unit-testable via `SimpleNamespace`, as the whole test suite does. |

## Function and class inventory: `single_user.py`

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `DEFAULT_READER_NAME` | none | `Joint Radiology Review` | Fixed provenance name shown by the shared account. Record the two human radiologist names separately in project notes. |
| `TOKEN_FILE_NAME` | none | `private_login_token.txt` | Recoverable local secret inside the ignored single-user work directory. |
| `SingleUserSetup` | cohort/reader/count/link/path/created | frozen result | Complete handoff from setup to the root launcher. |
| `_write_private_token` | path and raw token | atomic mode-`0600` file | Writes through a `.tmp` file and `os.replace`; never writes inside the repository's tracked source tree. |
| `_validated_manifest` | manifest path and phase | typed DataFrame | Requires exactly 20 pilot or 100 final unique study IDs before database mutation. |
| `prepare_single_user` | config, cohort, phase, frozen files, base URL | `SingleUserSetup` | Creates a new cohort or hash-verifies an existing one; preserves study rows on resume; rejects closed/mismatched cohorts and multiple accounts; creates/repairs the one token; idempotently gives all studies to the account. |

## Function inventory: `run_web_annotator.py`

| Symbol | Input | Output/behavior | Purpose |
|---|---|---|---|
| `REPOSITORY_ROOT` | script location | absolute repo root | Makes data/work/manifest defaults independent of the caller's current directory. |
| `_open_when_ready` | URL, host, port | opens default browser after socket accepts | Avoids opening the login link before Uvicorn is ready. |
| `_arguments` | argv | parsed pilot/final/path/port options | Keeps the zero-argument pilot path simple while retaining diagnostic overrides. |
| `main` | parsed arguments | setup → preflight → `caffeinate` → serve | The supported operator entry point. It prints the private link, runs the full existing preflight, starts macOS `caffeinate` tied to its PID, and delegates Uvicorn/backup behavior to `cli.command_serve`. |

## Important static UI functions

| Symbol | Input | Output/behavior | Purpose |
|---|---|---|---|
| `landmarkLabel` | machine landmark name | reviewed display title or fallback | Keeps machine identifiers out of the radiologist-facing chips. |
| `renderHeader` | authoritative task response | patient/landmark header and four chips | Uses `landmarkLabel(entry.landmark)`; state controls CSS color only and is never printed as the chip label. |
| `drawScene` | image plus pending/saved point | canvas frame | Red target for pending selection, green target for persisted point. |
| `drawCrosshair` | backing-store x/y and color | high-contrast target reticle | Multiplies all dimensions by device pixel ratio so the physical CSS size is stable. |
| `sourceToCanvas` / `canvasToSource` | source or canvas coordinates | inverse mapped coordinates | Accounts for letterboxing, zoom, pan, and backing-store scaling; marker styling does not participate in coordinate math. |
| `clientToCanvasCoordinates` | pointer event | backing-store x/y | Corrects CSS scaling and high-DPI canvas dimensions. |

## Test inventory (`tests/`)

| File | Scope | Needs real data-root? |
|---|---|---|
| `_fixtures.py` | `HAS_REAL_DATA` flag + `RealDataHarness` (initializes a temp DB against the real frozen 20-study pilot manifest) | Provides the fixture; itself needs it to be meaningful. |
| `test_auth.py` | token/hash properties | No |
| `test_coordinates.py` | `compute_and_validate_point` bounds/finiteness/round-trip, using synthetic `SliceGeometry`/`SeriesGeometry` (no DICOM I/O) | No |
| `test_database.py` | schema/PRAGMAs, cohort hash mismatch, assignment creation, save/replace/unable/reopen event history, revision conflicts, idempotent retry, concurrent-save race (real `threading.Thread`s against the real lock), sessions, backup rotation | No |
| `test_api.py` | full FastAPI `TestClient` over real pilot data: login/cookie/hides-token, session required everywhere, reader isolation, PNG headers + magic bytes, save computes XYZ server-side, stale-revision 409, idempotent retry, unable reason enum, static path traversal, forged task_id | Yes (auto-skips otherwise) |
| `test_resume.py` | kill the `Database`/app and rebuild from the same sqlite file mid-test; confirms resume position and that a `pass` task stays reachable/editable | Yes |
| `test_exports.py` | per-reader export schema + round-trip re-verification, full real 80-row pilot canonical export, overlap-without-adjudication failure, agreement report, manifest-hash-change refusal, `_BoundedByteCache` limits | Yes (except the cache-limit tests) |
| `test_single_user.py` | exactly-one-account enforcement, automatic all-study assignment, token file mode, safe second-run reuse, missing-token rotation | No |
| `test_static_ui.py` | anatomical chip labels rather than state words, large outlined target contract, static asset cache-version bump | No |

Run everything with:

```bash
python -m unittest discover -v -s kaggle/roi_cache -p 'test_*.py'
```

(`-s kaggle/roi_cache`, not `-s kaggle/roi_cache/web_annotator/tests` without
`-t` — see the README's testing section for why the top-level dir matters for
this package's relative imports.)

Current validation result after the single-user and UI changes: **73/73 tests
passed**, including all 14 pre-existing ROI geometry/builder tests and all
web-annotator tests. A real-data launcher smoke run created the one account,
automatically exposed all 20 pilot studies, and passed the full DICOM preflight.
The managed test environment blocked binding a second live localhost port, so
HTTP behavior remains covered by the FastAPI integration suite and the earlier
documented live-server smoke test.

## 2026-09-21: real final-cohort save incident and correction

After four final-cohort patients were completed, a normal click on patient 5
returned HTTP `400`. The server was still live; the browser UI incorrectly
labelled every non-2xx save response as **Offline**. The server's response was
instead an `invalid_point`: real `ImageOrientationPatient` row/column vectors
are accepted when near-orthogonal, but are not always mathematically exactly
orthogonal because of DICOM header rounding.

The former `SliceGeometry.patient_to_pixel` independently projected against
each unit vector. That shortcut is only an inverse for exactly orthogonal
vectors, so its error increased with distance from `ImagePositionPatient` and
could exceed the `1e-5 px` safety gate for an otherwise valid click. The stored
patient coordinate from `pixel_to_patient` was always correct; only the
validation inverse was faulty.

The correction in `roi_geometry.py` solves the two in-plane DICOM basis
vectors together with `numpy.linalg.lstsq`. This is the true inverse of
`pixel_to_patient` for both exact and permitted rounded IOPs. It leaves every
existing saved coordinate unchanged, retains the strict `1e-5 px` safety gate,
and is covered by `test_near_orthogonal_header_uses_a_true_in_plane_inverse`.
On the previously failing patient-5 reference slice, the image centre now
passes server validation.

`static/app.js` now treats a real network exception as **Offline** only. A
server-side rejected click remains selected, becomes **Point not saved**, and
shows a yellow explanation; it must be adjusted or marked unable rather than
silently retried as though connectivity failed. `index.html` now uses cache
version `v=3`, so refresh after a server restart loads this behavior.

At the correction point, the final SQLite database had **16 saved landmarks
across exactly 4 completed patients**. All were verified present with their
patient-space coordinates and audit events before restarting the server. The
restart used the same database, and a full final-100 DICOM preflight plus
`/health` check passed.

## 2026-09-21: web display-quality correction

The web client returns the same lossless 8-bit PNG generated from the desktop
annotator's own `_display_pixels` transform; no DICOM pixel decoding, modality
LUT, percentile windowing, source resolution, or click-coordinate math changed.
The visual issue was entirely canvas-side: `drawScene` had
`ctx.imageSmoothingEnabled = false`, which nearest-neighbour enlarged common
384×384 source MR images to the 900-pixel viewport and made them appear
pixelated. It now sets `imageSmoothingEnabled = true` and
`imageSmoothingQuality = "high"`. The update is cache-busted as static asset
version `v=4`; a normal browser refresh loads it without a server restart.
`test_enlarged_mri_slices_use_high_quality_canvas_scaling` protects this
radiologist-facing rendering contract.

## Final-cohort completion checkpoint (2026-09-21)

The first full final annotation pass produced **396 `pass` landmarks across 99
fully completed patients**, plus **four `unable` landmarks for patient 39**.
SQLite `PRAGMA integrity_check` returned `ok`, the local server was stopped
cleanly, and the resulting clean-shutdown backup is retained under
`roi_work/web_annotator_single/backups/`.

Do **not** run final `fit-priors` from this state. The scientific gate accepts
exactly 100 studies with all four landmarks approved and finite; it rejects
`unable`/incomplete rows. Patient 39 is manifest `cohort_complete` and has all
intended reference acquisitions available: coronal non-fluid for medial/lateral
joint lines, axial fluid for the patellofemoral centre, and sagittal fluid for
the suprapatellar recess. It is therefore a review/reannotation task, not an
eligible exclusion or an opportunity to reshuffle/replace a frozen cohort row.

To continue, reopen the four patient-39 tasks administratively (one command
per landmark; record the reviewer and reason), restart `run_web_annotator.py
--phase final`, mark all four points, then verify 400 `pass` rows. Only then:

1. Run `canonical-export` to write a verified 400-row annotation CSV.
2. Run `build_roi_cache.py fit-priors --phase final` with that canonical CSV
   and the frozen `annotation_manifest.csv`.
3. Require `accepted: true`, at least 95% held-out target coverage, zero mirror
   errors, and at least 95% audit laterality before cache generation.
4. Build and validate caches against the authoritative cache64 IDs; do not
   begin Arm C training before those build/validation gates pass.

## What is *not* implemented yet

- No HTTP admin dashboard (deliberately — spec section 4.2 allows CLI-only).
- No named/stable Cloudflare Tunnel automation (spec section 17 leaves this
  to the operator; only the Quick Tunnel command is documented).
- `ScientificService.cache_stats()` is not exposed through any route or CLI
  command yet; wire it into `cli.command_status` or a future `/api/health`
  detail if operational visibility into cache pressure is needed.
- No automatic creation of the frozen final 100-study manifest/template. Run
  the existing scientific `select-cohort` gate first; after those files exist,
  `python run_web_annotator.py --phase final` gives the single shared account
  all 100 studies automatically.
