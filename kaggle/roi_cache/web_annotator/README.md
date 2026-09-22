# Remote ROI landmark annotator (operator guide)

This is the Mac-hosted web application described in
`references/REMOTE_ROI_ANNOTATION_APP_SPEC.md`. It lets two radiologists
annotate the four Arm C ROI landmarks from any browser, while all DICOM data,
the SQLite database, and every export stay on this Mac. Read the spec before
operating this in production; this file is the practical run-book.

For a full class/function inventory and how the modules call into each other
(the engineering reference, not the run-book), see
`references/WEB_ANNOTATOR_REFERENCE.md`. Read that file first when resuming
work on this code — it exists specifically so a future session doesn't have
to re-derive the architecture from scratch.

**Do not use** `annotate_roi_pilot.py`/`run_local_pilot.py` for anything other
than local diagnostic comparison — they are unmodified and remain the
verified fallback. This application is entirely separate code under
`kaggle/roi_cache/web_annotator/`.

## Recommended: one shared account, one command

Both radiologists review cases together, so the normal workflow uses exactly
one shared account. From the repository root, run:

```bash
source .venv_web/bin/activate
python run_web_annotator.py
```

That command safely creates or resumes the 20-patient pilot, creates one
`Joint Radiology Review` account, gives it every patient, runs the real-DICOM
preflight, keeps the Mac awake, starts the server, prints the private link, and
opens it in the default browser. Stop with Ctrl-C and use the same command to
resume later. No initialization or assignment commands are needed.

The shared account's recoverable token is stored under the Git-ignored
`roi_work/web_annotator_single` directory with filesystem mode `0600`. This
single-account database is separate from any older multi-reader test database.
If port 8000 is occupied by an older server, stop that server first.

To expose the running application, use a second terminal:

```bash
cloudflared tunnel --url http://localhost:8000
```

Replace `http://127.0.0.1:8000` in the printed private link with the HTTPS
hostname printed by `cloudflared`; keep `/login?token=...` unchanged. Once the
frozen 100-patient manifest/template exist, start that cohort with:

```bash
python run_web_annotator.py --phase final
```

The detailed CLI below remains for diagnostics and unusual administration; it
is not needed for normal operation.

## 0. Decisions you must make before the pilot

Per spec section 17, the following are placeholders until you decide them.
Search `config.py` for `PLACEHOLDER` to find the exact defaults in code.

1. **Clinical provenance** — record both participating radiologists' names in
   the project notes; the UI account is `Joint Radiology Review`.
2. **Clinical wording + example image per landmark** — edit
   `static/guidance.json`. The UI prepends "DRAFT (pending radiologist
   sign-off)" to every instruction until both radiologists and you have
   reviewed and replaced the placeholder text.
3. **Whether your competition terms permit sharing images with the two
   radiologists over a Cloudflare Quick Tunnel** — confirm this yourself;
   the app does not check it.
4. **Quick Tunnel vs. a named/stable Cloudflare Tunnel** — see section 6.
5. **Session/token lifetime** — default is 90 days
   (`config.DEFAULT_SESSION_LIFETIME_DAYS`).

## 1. Install dependencies

```bash
cd rsna-knee-mri
python3 -m venv .venv_web
source .venv_web/bin/activate
pip install -r requirements.txt fastapi "uvicorn[standard]" httpx
```

(`fastapi`/`uvicorn`/`httpx` are the only additions beyond the existing ROI
pipeline's `requirements.txt`.)

## 2. Advanced/manual: initialize a cohort

The pilot manifest/template already exist at `roi_work/pilot/pilot_manifest.csv`
and `roi_work/pilot/pilot_annotations.csv` (created by
`build_roi_cache.py select-pilot`). Register them as a cohort:

```bash
python -m kaggle.roi_cache.web_annotator.cli init-cohort \
  --cohort pilot-2026 --phase pilot \
  --manifest roi_work/pilot/pilot_manifest.csv \
  --template roi_work/pilot/pilot_annotations.csv
```

This computes and stores SHA-256 hashes of both files. If either file
changes later, the server refuses to resume that cohort (spec 11.2) — you
must initialize a new cohort name deliberately.

In the advanced/manual workflow, create the reader and its assignments first,
then run preflight at the end of section 3. Preflight deliberately requires
every study to be assigned.

## 3. Advanced/manual: create reader links and assignments

```bash
python -m kaggle.roi_cache.web_annotator.cli create-reader --name "Dr. A" \
  --base-url http://127.0.0.1:8000
python -m kaggle.roi_cache.web_annotator.cli create-reader --name "Dr. B" \
  --base-url http://127.0.0.1:8000
```

Each prints a one-time bootstrap link, e.g.
`http://127.0.0.1:8000/login?token=<...>`. **Copy it now** — the raw token is
never shown again or stored in plaintext by the advanced CLI. Do not rerun
`create-reader` with the same name; it cannot recover an existing raw token.
For a tunnel, replace only the host portion of the link you saved.

Legacy multi-reader assignments can still be created explicitly:

```bash
python -m kaggle.roi_cache.web_annotator.cli assign \
  --cohort pilot-2026 --reader "Dr. A" --study-list all --group overlap
python -m kaggle.roi_cache.web_annotator.cli assign \
  --cohort pilot-2026 --reader "Dr. B" --study-list all --group overlap
```

Then run:

```bash
python -m kaggle.roi_cache.web_annotator.cli preflight --cohort pilot-2026
```

This checks hashes, exact study/landmark counts, reference-series resolution,
real-pixel decoding, assignment coverage, and free disk space.

## 4. Start the server (and, later, the tunnel)

```bash
caffeinate -dis python -m kaggle.roi_cache.web_annotator.cli serve \
  --cohort pilot-2026 --host 127.0.0.1 --port 8000
```

`caffeinate -dis` keeps the Mac awake and connected while a session is
active; it does not permanently change any power setting. `serve`:

- writes a pre-start backup if a database already exists;
- binds Uvicorn to `127.0.0.1` only, single worker;
- writes a clean-shutdown backup on exit (Ctrl-C).

Open `http://127.0.0.1:8000/` locally first and log in with each bootstrap
link to rehearse before inviting anyone else.

### Exposing it to the two radiologists

**Nothing here starts a tunnel automatically** — you start it deliberately:

```bash
cloudflared tunnel --url http://localhost:8000
```

This prints a random `https://<random>.trycloudflare.com` URL. Re-run
`create-reader --base-url https://<that-host>` (or just tell each
radiologist to replace the host in their existing link) so they get a
working `https://.../login?token=...` link. Quick Tunnels are for
testing/development and the hostname changes on every restart; for a stable
multi-day hostname, use a named Cloudflare Tunnel (requires a Cloudflare
account/domain) instead — this is an optional upgrade, not required for the
pilot.

## 5. Checking progress, backups, and safe shutdown

```bash
python -m kaggle.roi_cache.web_annotator.cli status --cohort pilot-2026
python -m kaggle.roi_cache.web_annotator.cli backup --reason manual-checkpoint
```

`status` never prints tokens or raw coordinates — only per-reader
patient/flagged counts and database size. To shut down safely, just Ctrl-C
the `serve` process (and `cloudflared` separately); a backup is written
automatically on the way out. Every individual save is already durable in
SQLite, so there is no "submit" step to remember.

To resume after any restart (browser, server, tunnel, or the whole Mac),
just restart `serve` with the same `--cohort` and reopen the same login
link — the reader resumes at their first incomplete or flagged task.

## 6. Export, agreement, adjudication, canonical CSV

Per-reader export (never both readers in one file):

```bash
python -m kaggle.roi_cache.web_annotator.cli export-reader \
  --cohort pilot-2026 --reader "Dr. A" --output roi_work/web_annotator/exports/dr_a.csv
```

Blinded agreement report once both readers finish all 20 studies:

```bash
python -m kaggle.roi_cache.web_annotator.cli agreement-report \
  --cohort pilot-2026 --output roi_work/web_annotator/exports/agreement.csv
```

This reports per-landmark 3D distance (median/IQR/p95/max) and never invents
a pass/fail rule — review the distribution with both radiologists and
record your own adjudication rule.

Record one explicit adjudication per overlapping study/landmark:

```bash
python -m kaggle.roi_cache.web_annotator.cli adjudicate \
  --cohort pilot-2026 --study <StudyInstanceUID> --landmark medial_joint_line \
  --decision choose_reader --chosen-reader "Dr. A" --reviewer "you" \
  --reason "closer to visible joint line on review"
```

or, to save an independently reviewed replacement point:

```bash
python -m kaggle.roi_cache.web_annotator.cli adjudicate \
  --cohort pilot-2026 --study <StudyInstanceUID> --landmark medial_joint_line \
  --decision replacement --reviewer "you" --reason "both readers off by ~4mm" \
  --replacement-sop-uid <SOPInstanceUID> --replacement-row 123.4 --replacement-column 88.1
```

The server (not you) computes and validates the replacement's patient XYZ
from the pixel you give it. After every overlap study/landmark has an
adjudication, and every single-reader study/landmark is `pass`:

```bash
python -m kaggle.roi_cache.web_annotator.cli canonical-export \
  --cohort pilot-2026 --output roi_work/canonical_pilot_annotations.csv
```

This produces exactly 80 rows for the pilot (400 for final), one row per
study/landmark, all `qc_status=pass`, and re-verifies every stored point's
pixel/patient round trip before writing. It prints the exact next command to
run — `build_roi_cache.py fit-priors` — which this app never runs on your
behalf; **this app does not itself mark the prior gate as passed.**

Close the cohort once it is finalized:

```bash
python -m kaggle.roi_cache.web_annotator.cli close-cohort --cohort pilot-2026
```

## 7. Storage layout

Everything mutable lives under `roi_work/web_annotator/` (already
`.gitignore`d by the repository's existing `roi_work/` rule):

```text
roi_work/web_annotator/
  annotator.sqlite3       # source of truth
  annotator.sqlite3-wal
  backups/                # timestamped, rotated to the newest 20
  logs/
```

Raw DICOMs are never copied or written to. The bounded PNG slice cache lives
in memory only (default 64 items / 200 MiB) and is never written to disk.

## 8. Running the tests

```bash
source .venv_web/bin/activate
# -t sets the top-level import root to kaggle/roi_cache so this package's
# relative imports (from . import ..., from .. import ...) resolve; -s scopes
# discovery to just this package's tests.
python -m unittest discover -v -t kaggle/roi_cache -s kaggle/roi_cache/web_annotator/tests -p 'test_*.py'

# Or run everything (existing ROI tests + this package) in one command:
python -m unittest discover -v -s kaggle/roi_cache -p 'test_*.py'
```

The integration tests (`test_api.py`, `test_resume.py`, `test_exports.py`)
run against the real frozen 20-study pilot manifest and the real
`../data-root` DICOMs; they auto-skip if either is not present, so the suite
still runs (unit-only) without the 530 GiB dataset mounted.

## 9. Known limitations

- A handful of studies in the pilot have `ImageOrientationPatient` row/column
  directions that are only approximately orthogonal (well within the
  existing 1° gate in `roi_geometry.SliceGeometry.from_header`, which this
  app does not change). Because `SliceGeometry.patient_to_pixel` is a direct
  dot-product projection rather than a full matrix inverse, the
  pixel→patient→pixel round-trip error grows with distance from the image
  corner and can exceed the spec's strict `1e-5` px tolerance for clicks far
  from the corner on those studies. This reproduces the existing desktop
  annotator's identical acceptance gate exactly — it is not new behavior
  introduced by this app — and simply means an occasional click near the
  edge of such a series is rejected with a clear "click again" message
  rather than silently accepted.
- The completion screen and per-patient progress are computed from the
  reader's own assignment order only; there is no cross-reader dashboard in
  the browser (by design — spec 4.1 forbids a reader seeing the other
  reader's clicks). Use the CLI `status`/`agreement-report` for the
  cross-reader view.
- `/static/*` (CSS/JS/the placeholder guidance text) is served without a
  session, since it contains no dataset content; every other route requires
  a session, including `/api/*` and `/api/slice`. `/health` is also
  unauthenticated by design so an external uptime check can hit it — it
  returns `{"status": "ok"}` only, no dataset or cohort details.
