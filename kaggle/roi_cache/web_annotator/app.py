"""FastAPI application: authenticated API, static UI, security headers.

Bind Uvicorn to ``127.0.0.1`` with one worker (see ``cli.py serve``). Every
reader-facing route derives the annotator identity from the session cookie,
never from a client-supplied field, and every DICOM-touching route calls into
``services.ScientificService`` rather than doing its own geometry or pixel
work.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import database, services
from .config import AppConfig
from .database import CohortClosed, Database, RevisionConflict
from .schemas import SaveAnnotationRequest, UnableRequest

SESSION_COOKIE = "roi_session"
STATIC_DIR = Path(__file__).resolve().parent / "static"

logger = logging.getLogger("web_annotator")


class AnnotatorContext:
    __slots__ = ("annotator_id", "display_name", "cohort")

    def __init__(self, annotator_id: int, display_name: str, cohort: database.Cohort):
        self.annotator_id = annotator_id
        self.display_name = display_name
        self.cohort = cohort


def encode_task_id(study_uid: str, landmark: str) -> str:
    payload = json.dumps({"su": study_uid, "lm": landmark}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def decode_task_id(task_id: str) -> tuple[str, str]:
    padding = "=" * (-len(task_id) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(task_id + padding))
        return str(payload["su"]), str(payload["lm"])
    except Exception as error:  # noqa: BLE001 - any malformed task id is a 400
        raise ValueError("invalid task_id") from error


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def create_app(config: AppConfig, db: Database, science: services.ScientificService, cohort_name: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        # Uvicorn re-raises the captured OS signal with the default handler
        # restored right after `uvicorn.run()` returns, which kills the
        # process before any code after that call can run. The ASGI shutdown
        # sequence below is awaited to completion first, so this is the only
        # place a "clean shutdown" backup is guaranteed to actually happen.
        target = db.backup(config.backup_dir, "clean-shutdown", config.max_backups)
        logger.info("clean-shutdown backup written to %s", target)

    app = FastAPI(
        title="Remote ROI Annotator", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )

    def _current_cohort() -> database.Cohort:
        cohort = db.get_cohort(cohort_name)
        if cohort is None:
            raise RuntimeError(f"configured cohort {cohort_name!r} does not exist in the database")
        return cohort

    def _require_session(request: Request) -> AnnotatorContext:
        raw_token = request.cookies.get(SESSION_COOKIE)
        if not raw_token:
            raise _Unauthorized()
        session_row = db.resolve_session(raw_token)
        if session_row is None:
            raise _Unauthorized()
        return AnnotatorContext(
            annotator_id=int(session_row["annotator_id"]),
            display_name=str(session_row["display_name"]),
            cohort=_current_cohort(),
        )

    class _Unauthorized(Exception):
        pass

    @app.middleware("http")
    async def security_and_logging(request: Request, call_next):
        request_id = uuid.uuid4().hex
        start = time.monotonic()
        try:
            response = await call_next(request)
        except _Unauthorized:
            response = JSONResponse({"error": "unauthorized", "request_id": request_id}, status_code=401)
        except _TaskAuthorizationError as error:
            response = JSONResponse(
                {"error": "task_not_assigned", "message": str(error), "request_id": request_id},
                status_code=403,
            )
        except Exception as error:  # noqa: BLE001 - convert to a safe response
            logger.exception("unhandled error request_id=%s path=%s", request_id, request.url.path)
            response = JSONResponse(
                {"error": "internal_error", "request_id": request_id}, status_code=500
            )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; script-src 'self'; "
            "style-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        response.headers["X-Request-Id"] = request_id
        if request.url.path.startswith("/api") and "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id, request.method, request.url.path, response.status_code, duration_ms,
        )
        return response

    # -- bootstrap / session -------------------------------------------------
    @app.get("/login")
    def login(token: str = Query(..., min_length=1, max_length=256)):
        annotator = db.authenticate_bootstrap_token(token)
        if annotator is None:
            return HTMLResponse(
                "<html><body><h1>Invalid or revoked link</h1>"
                "<p>Ask the project owner for a new private link.</p></body></html>",
                status_code=401,
            )
        session_token = db.create_session(int(annotator["id"]), config.session_lifetime_days)
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            SESSION_COOKIE,
            session_token,
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=config.session_lifetime_days * 86400,
            path="/",
        )
        return response

    @app.post("/logout")
    def logout(request: Request):
        raw_token = request.cookies.get(SESSION_COOKIE)
        if raw_token:
            db.delete_session(raw_token)
        response = JSONResponse({"status": "logged_out"})
        response.delete_cookie(SESSION_COOKIE, path="/")
        _no_store(response)
        return response

    @app.get("/health")
    def health():
        return JSONResponse({"status": "ok"})

    # -- reader-facing API ----------------------------------------------
    @app.get("/api/me")
    def me(context: AnnotatorContext = Depends(_require_session)):
        state = _reader_state(db, context.cohort, context.annotator_id)
        response = JSONResponse({
            "reader": context.display_name,
            "cohort_name": context.cohort.name,
            "cohort_phase": context.cohort.phase,
            "patient_count": state["patient_count"],
        })
        _no_store(response)
        return response

    @app.get("/api/progress")
    def progress(context: AnnotatorContext = Depends(_require_session)):
        state = _reader_state(db, context.cohort, context.annotator_id)
        response = JSONResponse({
            "patient_count": state["patient_count"],
            "completed_patient_count": len(state["completed_patient_indices"]),
            "remaining_patient_count": state["patient_count"] - len(state["completed_patient_indices"]),
            "flagged_task_count": state["flagged_task_count"],
            "patients_needing_review": state["patients_needing_review"],
            "all_complete": state["all_complete"],
        })
        _no_store(response)
        return response

    @app.get("/api/task")
    def get_task(
        patient_index: int | None = Query(default=None, ge=0),
        landmark: str | None = Query(default=None, max_length=64),
        context: AnnotatorContext = Depends(_require_session),
    ):
        state = _reader_state(db, context.cohort, context.annotator_id)
        if state["patient_count"] == 0:
            response = JSONResponse({"error": "no_assignments"}, status_code=404)
            _no_store(response)
            return response
        resolved_patient_index = (
            patient_index if patient_index is not None else state["resume_patient_index"]
        )
        resolved_patient_index = max(0, min(state["patient_count"] - 1, resolved_patient_index))
        if landmark is not None and landmark in services.LANDMARKS:
            landmark_index = services.LANDMARKS.index(landmark)
        else:
            landmark_index = (
                state["resume_landmark_index"] if resolved_patient_index == state["resume_patient_index"] else 0
            )
        assignment = state["assignments"][resolved_patient_index]
        study_uid = assignment["study_uid"]
        current_landmark = services.LANDMARKS[landmark_index]
        annotation_row = state["annotations_by_key"].get((study_uid, current_landmark))
        landmark_states = [
            {
                "landmark": name,
                "state": state["annotations_by_key"].get((study_uid, name), {}).get("state", "incomplete"),
            }
            for name in services.LANDMARKS
        ]
        try:
            science.check_data_root_available()
            task = science.resolve_task(study_uid, current_landmark)
            series_info = {
                "plane": task.series.plane,
                "fluid_sensitive": task.series.fluid_sensitive,
                "slice_count": len(task.series.slices),
            }
            default_slice_index = len(task.series.slices) // 2
            if annotation_row and annotation_row.get("sop_uid"):
                default_slice_index = next(
                    (i for i, item in enumerate(task.series.slices) if item.sop_uid == annotation_row["sop_uid"]),
                    default_slice_index,
                )
            study_row = db.get_study(context.cohort.id, study_uid)
            laterality = str(study_row["laterality"]) if study_row and study_row["laterality"] else None
            image_medial_side = services.medial_image_side(task, laterality)
            reference_error = None
        except services.DataRootUnavailable:
            series_info, default_slice_index, laterality, image_medial_side, reference_error = (
                None, None, None, None, "data_volume_unavailable"
            )
        except services.ReferenceSeriesUnavailable:
            series_info, default_slice_index, laterality, image_medial_side, reference_error = (
                None, None, None, None, "reference_series_unavailable"
            )
        payload = {
            "task_id": encode_task_id(study_uid, current_landmark),
            "revision": annotation_row["revision"] if annotation_row else 0,
            "cohort_name": context.cohort.name,
            "cohort_phase": context.cohort.phase,
            "reader": context.display_name,
            "patient_index": resolved_patient_index,
            "patient_count": state["patient_count"],
            "remaining_patient_count": state["patient_count"] - len(state["completed_patient_indices"]),
            "landmark_index": landmark_index,
            "landmark_count": len(services.LANDMARKS),
            "landmark": current_landmark,
            "landmark_states": landmark_states,
            "state": annotation_row["state"] if annotation_row else "incomplete",
            "saved_point": (
                {
                    "sop_uid": annotation_row["sop_uid"],
                    "row_px": annotation_row["row_px"],
                    "column_px": annotation_row["column_px"],
                }
                if annotation_row and annotation_row["state"] == "pass"
                else None
            ),
            "previous_point": (
                db.latest_historical_pass_point(int(annotation_row["id"]))
                if annotation_row and annotation_row["state"] == "incomplete"
                else None
            ),
            "unable": (
                {"reason": annotation_row["unable_reason"], "notes": annotation_row["unable_notes"]}
                if annotation_row and annotation_row["state"] == "unable"
                else None
            ),
            "series": series_info,
            "default_slice_index": default_slice_index,
            "reference_error": reference_error,
            "study_uid": study_uid,
            "laterality": laterality,
            "image_medial_side": image_medial_side,
            "cohort_state": context.cohort.state,
        }
        response = JSONResponse(payload)
        _no_store(response)
        return response

    @app.get("/api/slice")
    def get_slice(
        task_id: str = Query(..., max_length=512),
        index: int = Query(..., ge=0),
        context: AnnotatorContext = Depends(_require_session),
    ):
        study_uid, landmark = _authorize_task(db, context, task_id)
        try:
            science.check_data_root_available()
            task = science.resolve_task(study_uid, landmark)
        except services.DataRootUnavailable:
            response = JSONResponse({"error": "data_volume_unavailable"}, status_code=503)
            _no_store(response)
            return response
        except services.ReferenceSeriesUnavailable:
            response = JSONResponse({"error": "reference_series_unavailable"}, status_code=422)
            _no_store(response)
            return response
        if not (0 <= index < len(task.series.slices)):
            response = JSONResponse({"error": "slice_index_out_of_bounds"}, status_code=400)
            _no_store(response)
            return response
        item = task.series.slices[index]
        png_bytes = science.render_slice_png(task, index)
        response = Response(content=png_bytes, media_type="image/png")
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Slice-Rows"] = str(item.rows)
        response.headers["X-Slice-Columns"] = str(item.columns)
        response.headers["X-Slice-Sop-Uid"] = item.sop_uid
        response.headers["X-Slice-Index"] = str(index)
        response.headers["X-Slice-Count"] = str(len(task.series.slices))
        return response

    @app.post("/api/annotations/save")
    def save_annotation(payload: SaveAnnotationRequest, context: AnnotatorContext = Depends(_require_session)):
        study_uid, landmark = _authorize_task(db, context, payload.task_id)
        try:
            science.check_data_root_available()
            task = science.resolve_task(study_uid, landmark)
            point, resolved_sop_uid, _slice_index = science.compute_and_validate_point(
                task, payload.sop_uid, payload.row_px, payload.column_px
            )
        except services.DataRootUnavailable:
            response = JSONResponse({"error": "data_volume_unavailable", "saved": False}, status_code=503)
            _no_store(response)
            return response
        except (services.OutOfBounds, services.RoundTripValidationFailed) as error:
            response = JSONResponse({"error": "invalid_point", "message": str(error), "saved": False}, status_code=400)
            _no_store(response)
            return response
        study_row = db.get_study(context.cohort.id, study_uid)
        laterality = study_row["laterality"] if study_row is not None else None
        try:
            record = db.save_point(
                cohort_id=context.cohort.id,
                annotator_id=context.annotator_id,
                study_uid=study_uid,
                landmark=landmark,
                actor=context.display_name,
                request_id=payload.request_id,
                expected_revision=payload.expected_revision,
                series_uid=task.series.series_uid,
                sop_uid=resolved_sop_uid,
                row_px=payload.row_px,
                column_px=payload.column_px,
                patient_x_mm=float(point[0]),
                patient_y_mm=float(point[1]),
                patient_z_mm=float(point[2]),
                laterality=laterality,
                cohort_state=context.cohort.state,
            )
        except RevisionConflict as conflict:
            response = JSONResponse(
                {"error": "stale_revision", "saved": False, "current": _serialize_annotation(conflict.current)},
                status_code=409,
            )
            _no_store(response)
            return response
        except CohortClosed as error:
            response = JSONResponse({"error": "cohort_closed", "saved": False, "message": str(error)}, status_code=409)
            _no_store(response)
            return response
        response = JSONResponse({"saved": True, "annotation": _serialize_annotation(record)})
        _no_store(response)
        return response

    @app.post("/api/annotations/unable")
    def mark_unable(payload: UnableRequest, context: AnnotatorContext = Depends(_require_session)):
        study_uid, landmark = _authorize_task(db, context, payload.task_id)
        try:
            record = db.mark_unable(
                cohort_id=context.cohort.id,
                annotator_id=context.annotator_id,
                study_uid=study_uid,
                landmark=landmark,
                actor=context.display_name,
                request_id=payload.request_id,
                expected_revision=payload.expected_revision,
                reason=payload.reason,
                notes=payload.notes,
                cohort_state=context.cohort.state,
            )
        except RevisionConflict as conflict:
            response = JSONResponse(
                {"error": "stale_revision", "saved": False, "current": _serialize_annotation(conflict.current)},
                status_code=409,
            )
            _no_store(response)
            return response
        except CohortClosed as error:
            response = JSONResponse({"error": "cohort_closed", "saved": False, "message": str(error)}, status_code=409)
            _no_store(response)
            return response
        response = JSONResponse({"saved": True, "annotation": _serialize_annotation(record)})
        _no_store(response)
        return response

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        raw_token = request.cookies.get(SESSION_COOKIE)
        if not raw_token or db.resolve_session(raw_token) is None:
            return HTMLResponse(
                "<html><body><h1>Private link required</h1>"
                "<p>Ask the project owner for your annotation link.</p></body></html>",
                status_code=401,
            )
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def _authorize_task(db: Database, context: AnnotatorContext, task_id: str) -> tuple[str, str]:
    try:
        study_uid, landmark = decode_task_id(task_id)
    except ValueError:
        raise _TaskAuthorizationError("invalid task_id")
    row = db.get_annotation(context.cohort.id, context.annotator_id, study_uid, landmark)
    if row is None:
        raise _TaskAuthorizationError("task is not assigned to this reader")
    return study_uid, landmark


class _TaskAuthorizationError(Exception):
    pass


def _serialize_annotation(row: Any) -> dict[str, Any]:
    mapping = dict(row)
    return {
        "study_uid": mapping.get("study_uid"),
        "landmark": mapping.get("landmark"),
        "state": mapping.get("state"),
        "revision": mapping.get("revision"),
        "sop_uid": mapping.get("sop_uid"),
        "row_px": mapping.get("row_px"),
        "column_px": mapping.get("column_px"),
        "patient_x_mm": mapping.get("patient_x_mm"),
        "patient_y_mm": mapping.get("patient_y_mm"),
        "patient_z_mm": mapping.get("patient_z_mm"),
        "unable_reason": mapping.get("unable_reason"),
        "unable_notes": mapping.get("unable_notes"),
    }


def _reader_state(db: Database, cohort: database.Cohort, annotator_id: int) -> dict[str, Any]:
    assignments = db.list_assignments(cohort.id, annotator_id)
    annotations = db.list_annotations_for_reader(cohort.id, annotator_id)
    annotations_by_key = {(row["study_uid"], row["landmark"]): dict(row) for row in annotations}
    patient_count = len(assignments)
    completed_patient_indices: list[int] = []
    patients_needing_review: list[dict[str, Any]] = []
    flagged_task_count = 0
    resume_patient_index = patient_count - 1 if patient_count else 0
    resume_landmark_index = len(services.LANDMARKS) - 1
    resume_found = False
    for patient_index, assignment in enumerate(assignments):
        study_uid = assignment["study_uid"]
        states = [
            annotations_by_key.get((study_uid, landmark), {}).get("state", "incomplete")
            for landmark in services.LANDMARKS
        ]
        flagged_task_count += states.count("unable")
        if all(state == "pass" for state in states):
            completed_patient_indices.append(patient_index)
        elif any(state == "unable" for state in states):
            patients_needing_review.append({
                "patient_index": patient_index,
                "flagged_landmarks": [
                    services.LANDMARKS[i] for i, state in enumerate(states) if state == "unable"
                ],
            })
        if not resume_found:
            for landmark_index, state in enumerate(states):
                if state in ("incomplete", "unable"):
                    resume_patient_index = patient_index
                    resume_landmark_index = landmark_index
                    resume_found = True
                    break
    return {
        "assignments": assignments,
        "annotations_by_key": annotations_by_key,
        "patient_count": patient_count,
        "completed_patient_indices": completed_patient_indices,
        "patients_needing_review": patients_needing_review,
        "flagged_task_count": flagged_task_count,
        "resume_patient_index": resume_patient_index,
        "resume_landmark_index": resume_landmark_index,
        "all_complete": len(completed_patient_indices) == patient_count and patient_count > 0,
    }
