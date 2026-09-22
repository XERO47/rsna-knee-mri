# Web annotator domain index

Files in this folder, split from `REMOTE_ROI_ANNOTATION_APP_SPEC.md` and
`WEB_ANNOTATOR_REFERENCE.md` (archived at `../archive/`).

| File | Read this for |
|---|---|
| `spec.md` | The contract: outcome/scope, cohorts (pilot 20/80, final 100/400), the 4 fixed landmark tasks and their reference-series table, the exact 12-column export schema, user roles/access, authentication rules, workflow gates A/B, adjudication/canonical-export rules, and the human sign-off list. Read this first for *what* the app must do and *why*. |
| `frontend.md` | HTML/CSS/JS constraints, header/image-area/controls UI spec, the exact click→source-pixel coordinate mapping contract, crosshair rendering details (`drawCrosshair`, `sourceToCanvas`/`canvasToSource`), static UI function inventory, and the two 2026-09-21 UI corrections (offline-vs-rejected-click labeling; canvas smoothing). |
| `backend.md` | FastAPI/Uvicorn/SQLite stack, file layout, module responsibilities, the two non-obvious implementation facts (Uvicorn signal handling in `lifespan`; the `patient_to_pixel` geometry inverse bug and fix), full persistence schema (`cohorts`/`studies`/`annotators`/`assignments`/`annotations`/`annotation_events`/`adjudications`), transaction/backup/resume rules, the HTTP/API contract, preflight checks, error-handling table, test inventory, per-module (`config.py`/`auth.py`/`database.py`/`services.py`/`schemas.py`/`app.py`/`cli.py`/`single_user.py`/`run_web_annotator.py`) symbol tables, and operational incident history. |

Use `spec.md` when deciding whether a behavior is required/forbidden. Use
`frontend.md` or `backend.md` when implementing or debugging the code that
realizes that behavior — they describe what the current implementation
actually does, including corrections made after real annotation runs.
