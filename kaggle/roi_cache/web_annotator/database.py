"""SQLite persistence layer: schema, transactions, revisions, audit history.

SQLite is the source of truth. All writes go through :class:`Database`, which
serializes writes with a process-local lock (the app runs as one Uvicorn
worker for two human users, so this is sufficient and keeps the concurrency
model easy to reason about) and always commits the mutated row plus its
append-only audit event in one transaction.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from . import auth

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cohorts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    phase TEXT NOT NULL CHECK (phase IN ('pilot', 'final')),
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    template_path TEXT NOT NULL,
    template_sha256 TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open', 'closed')),
    created_at TEXT NOT NULL,
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS studies (
    id INTEGER PRIMARY KEY,
    cohort_id INTEGER NOT NULL REFERENCES cohorts(id),
    study_uid TEXT NOT NULL,
    display_order INTEGER NOT NULL,
    laterality TEXT,
    selection_role TEXT,
    prior_partition TEXT,
    manifest_json TEXT NOT NULL,
    UNIQUE (cohort_id, study_uid)
);

CREATE TABLE IF NOT EXISTS annotators (
    id INTEGER PRIMARY KEY,
    display_name TEXT NOT NULL UNIQUE,
    token_hash TEXT NOT NULL UNIQUE,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    session_token_hash TEXT NOT NULL UNIQUE,
    annotator_id INTEGER NOT NULL REFERENCES annotators(id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY,
    cohort_id INTEGER NOT NULL REFERENCES cohorts(id),
    annotator_id INTEGER NOT NULL REFERENCES annotators(id),
    study_uid TEXT NOT NULL,
    reader_order INTEGER NOT NULL,
    assignment_group TEXT NOT NULL,
    UNIQUE (cohort_id, annotator_id, study_uid)
);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY,
    cohort_id INTEGER NOT NULL REFERENCES cohorts(id),
    annotator_id INTEGER NOT NULL REFERENCES annotators(id),
    study_uid TEXT NOT NULL,
    landmark TEXT NOT NULL,
    series_uid TEXT,
    sop_uid TEXT,
    row_px REAL,
    column_px REAL,
    patient_x_mm REAL,
    patient_y_mm REAL,
    patient_z_mm REAL,
    laterality TEXT,
    state TEXT NOT NULL DEFAULT 'incomplete' CHECK (state IN ('incomplete', 'pass', 'unable')),
    unable_reason TEXT,
    unable_notes TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (cohort_id, annotator_id, study_uid, landmark)
);

CREATE TABLE IF NOT EXISTS annotation_events (
    id INTEGER PRIMARY KEY,
    annotation_id INTEGER NOT NULL REFERENCES annotations(id),
    event_type TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT NOT NULL,
    actor TEXT NOT NULL,
    request_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    request_id TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adjudications (
    id INTEGER PRIMARY KEY,
    cohort_id INTEGER NOT NULL REFERENCES cohorts(id),
    study_uid TEXT NOT NULL,
    landmark TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('choose_reader', 'replacement')),
    chosen_annotator_id INTEGER REFERENCES annotators(id),
    replacement_series_uid TEXT,
    replacement_sop_uid TEXT,
    replacement_row_px REAL,
    replacement_column_px REAL,
    replacement_patient_x_mm REAL,
    replacement_patient_y_mm REAL,
    replacement_patient_z_mm REAL,
    reviewer TEXT NOT NULL,
    reason TEXT,
    source_revision_a INTEGER,
    source_revision_b INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (cohort_id, study_uid, landmark)
);

CREATE INDEX IF NOT EXISTS idx_annotations_lookup
    ON annotations (cohort_id, annotator_id, study_uid, landmark);
CREATE INDEX IF NOT EXISTS idx_assignments_reader
    ON assignments (cohort_id, annotator_id, reader_order);
CREATE INDEX IF NOT EXISTS idx_events_annotation
    ON annotation_events (annotation_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RevisionConflict(Exception):
    """Raised when a save/unable request targets a stale revision."""

    def __init__(self, current: dict[str, Any]):
        super().__init__("annotation revision has changed since it was loaded")
        self.current = current


class ManifestHashMismatch(Exception):
    """Raised when a resumed cohort's manifest/template hash has changed."""


class CohortClosed(Exception):
    """Raised when a write is attempted against a closed cohort."""


@dataclass(frozen=True)
class Cohort:
    id: int
    name: str
    phase: str
    manifest_path: str
    manifest_sha256: str
    template_path: str
    template_sha256: str
    state: str


class Database:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._init_schema()

    # -- schema -----------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock:
            self._connection.executescript(SCHEMA_SQL)
            row = self._connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema version {row['value']} does not match "
                    f"application schema version {SCHEMA_VERSION}"
                )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # -- backup -------------------------------------------------------------
    def backup(self, backup_dir: Path, reason: str, max_backups: int) -> Path:
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = backup_dir / f"annotator-{timestamp}-{reason}.sqlite3"
        with self._lock:
            destination = sqlite3.connect(str(target))
            try:
                self._connection.backup(destination)
            finally:
                destination.close()
        existing = sorted(backup_dir.glob("annotator-*.sqlite3"))
        for stale in existing[:-max_backups] if max_backups > 0 else []:
            stale.unlink(missing_ok=True)
        return target

    # -- cohorts --------------------------------------------------------
    def create_or_resume_cohort(
        self,
        name: str,
        phase: str,
        manifest_path: str,
        manifest_sha256: str,
        template_path: str,
        template_sha256: str,
    ) -> Cohort:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM cohorts WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO cohorts (name, phase, manifest_path, manifest_sha256, "
                    "template_path, template_sha256, state, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'open', ?)",
                    (name, phase, manifest_path, manifest_sha256, template_path,
                     template_sha256, now_iso()),
                )
                row = self._connection.execute(
                    "SELECT * FROM cohorts WHERE name = ?", (name,)
                ).fetchone()
            else:
                if row["manifest_sha256"] != manifest_sha256 or row["template_sha256"] != template_sha256:
                    raise ManifestHashMismatch(
                        f"cohort {name!r} manifest/template hash changed on disk; "
                        "refusing to resume. Initialize a new cohort explicitly if this "
                        "is intentional."
                    )
            return _row_to_cohort(row)

    def get_cohort(self, name: str) -> Cohort | None:
        row = self._connection.execute(
            "SELECT * FROM cohorts WHERE name = ?", (name,)
        ).fetchone()
        return _row_to_cohort(row) if row is not None else None

    def close_cohort(self, name: str) -> None:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE cohorts SET state = 'closed', closed_at = ? WHERE name = ? AND state = 'open'",
                (now_iso(), name),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"cohort {name!r} does not exist or is already closed")

    # -- studies ----------------------------------------------------------
    def replace_studies(self, cohort_id: int, records: list[dict[str, Any]]) -> None:
        with self._lock, self._transaction():
            self._connection.execute("DELETE FROM studies WHERE cohort_id = ?", (cohort_id,))
            for order, record in enumerate(records):
                self._connection.execute(
                    "INSERT INTO studies (cohort_id, study_uid, display_order, laterality, "
                    "selection_role, prior_partition, manifest_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        cohort_id,
                        str(record["StudyInstanceUID"]),
                        order,
                        record.get("laterality"),
                        record.get("selection_role"),
                        record.get("prior_partition"),
                        json.dumps(record, default=str),
                    ),
                )

    def list_studies(self, cohort_id: int) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM studies WHERE cohort_id = ? ORDER BY display_order", (cohort_id,)
        ).fetchall()

    def get_study(self, cohort_id: int, study_uid: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM studies WHERE cohort_id = ? AND study_uid = ?",
            (cohort_id, study_uid),
        ).fetchone()

    # -- annotators / auth --------------------------------------------------
    def create_annotator(self, display_name: str) -> tuple[int, str]:
        token = auth.generate_token()
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "INSERT INTO annotators (display_name, token_hash, active, created_at) "
                    "VALUES (?, ?, 1, ?)",
                    (display_name, auth.hash_token(token), now_iso()),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"annotator {display_name!r} already exists") from error
            return int(cursor.lastrowid), token

    def list_annotators(self) -> list[sqlite3.Row]:
        """Return every account so single-user mode can enforce exclusivity."""
        return self._connection.execute(
            "SELECT * FROM annotators ORDER BY id"
        ).fetchall()

    def rotate_annotator_token(self, annotator_id: int) -> str:
        """Issue a replacement bootstrap token and invalidate old sessions."""
        token = auth.generate_token()
        with self._lock, self._transaction():
            cursor = self._connection.execute(
                "UPDATE annotators SET token_hash = ?, active = 1, revoked_at = NULL "
                "WHERE id = ?",
                (auth.hash_token(token), annotator_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"annotator id {annotator_id} does not exist")
            self._connection.execute(
                "DELETE FROM sessions WHERE annotator_id = ?",
                (annotator_id,),
            )
        return token

    def revoke_annotator(self, display_name: str) -> None:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE annotators SET active = 0, revoked_at = ? WHERE display_name = ? AND active = 1",
                (now_iso(), display_name),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"annotator {display_name!r} does not exist or is already revoked")
            annotator = self._connection.execute(
                "SELECT id FROM annotators WHERE display_name = ?", (display_name,)
            ).fetchone()
            self._connection.execute(
                "DELETE FROM sessions WHERE annotator_id = ?", (annotator["id"],)
            )

    def get_annotator_by_name(self, display_name: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM annotators WHERE display_name = ?", (display_name,)
        ).fetchone()

    def get_annotator_by_id(self, annotator_id: int) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM annotators WHERE id = ?", (annotator_id,)
        ).fetchone()

    def authenticate_bootstrap_token(self, raw_token: str) -> sqlite3.Row | None:
        token_hash = auth.hash_token(raw_token)
        row = self._connection.execute(
            "SELECT * FROM annotators WHERE token_hash = ? AND active = 1", (token_hash,)
        ).fetchone()
        return row

    def create_session(self, annotator_id: int, lifetime_days: int) -> str:
        raw_session_token = auth.generate_token()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=lifetime_days)).isoformat()
        with self._lock:
            self._connection.execute(
                "INSERT INTO sessions (session_token_hash, annotator_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (auth.hash_token(raw_session_token), annotator_id, now_iso(), expires_at),
            )
        return raw_session_token

    def resolve_session(self, raw_session_token: str) -> sqlite3.Row | None:
        token_hash = auth.hash_token(raw_session_token)
        row = self._connection.execute(
            "SELECT sessions.*, annotators.display_name, annotators.active AS annotator_active "
            "FROM sessions JOIN annotators ON annotators.id = sessions.annotator_id "
            "WHERE sessions.session_token_hash = ?",
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        if not row["annotator_active"]:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            return None
        return row

    def delete_session(self, raw_session_token: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM sessions WHERE session_token_hash = ?",
                (auth.hash_token(raw_session_token),),
            )

    # -- assignments --------------------------------------------------------
    def assign_studies(
        self,
        cohort_id: int,
        annotator_id: int,
        study_uids: list[str],
        group: str,
        landmarks: Iterable[str],
    ) -> int:
        studies = {row["study_uid"]: row for row in self.list_studies(cohort_id)}
        missing = [uid for uid in study_uids if uid not in studies]
        if missing:
            raise ValueError(f"studies not in cohort: {missing[:5]}")
        ordered = sorted(study_uids, key=lambda uid: studies[uid]["display_order"])
        created = 0
        with self._lock, self._transaction():
            for reader_order, study_uid in enumerate(ordered):
                cursor = self._connection.execute(
                    "INSERT OR IGNORE INTO assignments "
                    "(cohort_id, annotator_id, study_uid, reader_order, assignment_group) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (cohort_id, annotator_id, study_uid, reader_order, group),
                )
                created += cursor.rowcount
                laterality = studies[study_uid]["laterality"]
                for landmark in landmarks:
                    self._connection.execute(
                        "INSERT OR IGNORE INTO annotations "
                        "(cohort_id, annotator_id, study_uid, landmark, laterality, state, "
                        "revision, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, 'incomplete', 0, ?, ?)",
                        (cohort_id, annotator_id, study_uid, landmark, laterality,
                         now_iso(), now_iso()),
                    )
        return created

    def list_assignments(self, cohort_id: int, annotator_id: int) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM assignments WHERE cohort_id = ? AND annotator_id = ? "
            "ORDER BY reader_order",
            (cohort_id, annotator_id),
        ).fetchall()

    # -- annotations ----------------------------------------------------
    def get_annotation(
        self, cohort_id: int, annotator_id: int, study_uid: str, landmark: str
    ) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM annotations WHERE cohort_id = ? AND annotator_id = ? "
            "AND study_uid = ? AND landmark = ?",
            (cohort_id, annotator_id, study_uid, landmark),
        ).fetchone()

    def list_annotations_for_study(
        self, cohort_id: int, annotator_id: int, study_uid: str
    ) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM annotations WHERE cohort_id = ? AND annotator_id = ? AND study_uid = ?",
            (cohort_id, annotator_id, study_uid),
        ).fetchall()

    def list_annotations_for_reader(self, cohort_id: int, annotator_id: int) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM annotations WHERE cohort_id = ? AND annotator_id = ?",
            (cohort_id, annotator_id),
        ).fetchall()

    def list_annotations_for_cohort(self, cohort_id: int) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM annotations WHERE cohort_id = ?", (cohort_id,)
        ).fetchall()

    def get_idempotent_response(self, request_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT response_json FROM idempotency_keys WHERE request_id = ?", (request_id,)
        ).fetchone()
        return json.loads(row["response_json"]) if row is not None else None

    def save_point(
        self,
        cohort_id: int,
        annotator_id: int,
        study_uid: str,
        landmark: str,
        actor: str,
        request_id: str,
        expected_revision: int,
        series_uid: str,
        sop_uid: str,
        row_px: float,
        column_px: float,
        patient_x_mm: float,
        patient_y_mm: float,
        patient_z_mm: float,
        laterality: str | None,
        cohort_state: str,
    ) -> dict[str, Any]:
        cached = self.get_idempotent_response(request_id)
        if cached is not None:
            return cached
        if cohort_state != "open":
            raise CohortClosed(f"cohort is closed; {study_uid}/{landmark} cannot be modified")
        with self._lock, self._transaction():
            current = self.get_annotation(cohort_id, annotator_id, study_uid, landmark)
            if current is None:
                raise ValueError(f"no assignment for {study_uid}/{landmark}")
            if current["revision"] != expected_revision:
                raise RevisionConflict(dict(current))
            before = dict(current)
            event_type = "replace" if current["state"] == "pass" else "create"
            new_revision = current["revision"] + 1
            timestamp = now_iso()
            self._connection.execute(
                "UPDATE annotations SET series_uid = ?, sop_uid = ?, row_px = ?, column_px = ?, "
                "patient_x_mm = ?, patient_y_mm = ?, patient_z_mm = ?, laterality = ?, "
                "state = 'pass', unable_reason = NULL, unable_notes = NULL, revision = ?, "
                "updated_at = ? WHERE id = ?",
                (series_uid, sop_uid, row_px, column_px, patient_x_mm, patient_y_mm,
                 patient_z_mm, laterality, new_revision, timestamp, current["id"]),
            )
            after = dict(self.get_annotation(cohort_id, annotator_id, study_uid, landmark))
            self._connection.execute(
                "INSERT INTO annotation_events (annotation_id, event_type, before_json, "
                "after_json, actor, request_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (current["id"], event_type, json.dumps(before, default=str),
                 json.dumps(after, default=str), actor, request_id, timestamp),
            )
            self._connection.execute(
                "INSERT INTO idempotency_keys (request_id, response_json, created_at) "
                "VALUES (?, ?, ?)",
                (request_id, json.dumps(after, default=str), timestamp),
            )
        return after

    def mark_unable(
        self,
        cohort_id: int,
        annotator_id: int,
        study_uid: str,
        landmark: str,
        actor: str,
        request_id: str,
        expected_revision: int,
        reason: str,
        notes: str | None,
        cohort_state: str,
    ) -> dict[str, Any]:
        cached = self.get_idempotent_response(request_id)
        if cached is not None:
            return cached
        if cohort_state != "open":
            raise CohortClosed(f"cohort is closed; {study_uid}/{landmark} cannot be modified")
        with self._lock, self._transaction():
            current = self.get_annotation(cohort_id, annotator_id, study_uid, landmark)
            if current is None:
                raise ValueError(f"no assignment for {study_uid}/{landmark}")
            if current["revision"] != expected_revision:
                raise RevisionConflict(dict(current))
            before = dict(current)
            new_revision = current["revision"] + 1
            timestamp = now_iso()
            self._connection.execute(
                "UPDATE annotations SET state = 'unable', unable_reason = ?, unable_notes = ?, "
                "series_uid = NULL, sop_uid = NULL, row_px = NULL, column_px = NULL, "
                "patient_x_mm = NULL, patient_y_mm = NULL, patient_z_mm = NULL, "
                "revision = ?, updated_at = ? WHERE id = ?",
                (reason, notes, new_revision, timestamp, current["id"]),
            )
            after = dict(self.get_annotation(cohort_id, annotator_id, study_uid, landmark))
            self._connection.execute(
                "INSERT INTO annotation_events (annotation_id, event_type, before_json, "
                "after_json, actor, request_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (current["id"], "unable", json.dumps(before, default=str),
                 json.dumps(after, default=str), actor, request_id, timestamp),
            )
            self._connection.execute(
                "INSERT INTO idempotency_keys (request_id, response_json, created_at) "
                "VALUES (?, ?, ?)",
                (request_id, json.dumps(after, default=str), timestamp),
            )
        return after

    def reopen(
        self, cohort_id: int, annotator_id: int, study_uid: str, landmark: str, actor: str, reason: str,
    ) -> dict[str, Any]:
        """Administrative reset back to ``incomplete``. Preserves history."""
        with self._lock, self._transaction():
            current = self.get_annotation(cohort_id, annotator_id, study_uid, landmark)
            if current is None:
                raise ValueError(f"no assignment for {study_uid}/{landmark}")
            before = dict(current)
            timestamp = now_iso()
            self._connection.execute(
                "UPDATE annotations SET state = 'incomplete', unable_reason = NULL, "
                "unable_notes = NULL, series_uid = NULL, sop_uid = NULL, row_px = NULL, "
                "column_px = NULL, patient_x_mm = NULL, patient_y_mm = NULL, "
                "patient_z_mm = NULL, revision = revision + 1, updated_at = ? WHERE id = ?",
                (timestamp, current["id"]),
            )
            after = dict(self.get_annotation(cohort_id, annotator_id, study_uid, landmark))
            self._connection.execute(
                "INSERT INTO annotation_events (annotation_id, event_type, before_json, "
                "after_json, actor, request_id, created_at) VALUES (?, 'reopen', ?, ?, ?, ?, ?)",
                (current["id"], json.dumps(before, default=str), json.dumps(after, default=str),
                 actor, f"admin-reopen:{reason}", timestamp),
            )
        return after

    def list_events(self, annotation_id: int) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM annotation_events WHERE annotation_id = ? ORDER BY id",
            (annotation_id,),
        ).fetchall()

    def latest_historical_pass_point(self, annotation_id: int) -> dict[str, Any] | None:
        """Return the last saved point from an annotation's append-only history.

        Administrative reopening clears the live record so an unreviewed point
        cannot enter a canonical export. The preserved prior point can still be
        rendered as an amber reference during correction.
        """
        events = self._connection.execute(
            "SELECT before_json, after_json FROM annotation_events "
            "WHERE annotation_id = ? ORDER BY id DESC",
            (annotation_id,),
        ).fetchall()
        for event in events:
            for column in ("after_json", "before_json"):
                payload = json.loads(event[column])
                if payload.get("state") == "pass" and payload.get("sop_uid"):
                    return {
                        "sop_uid": payload["sop_uid"],
                        "row_px": payload["row_px"],
                        "column_px": payload["column_px"],
                    }
        return None

    def record_export_event(self, annotation_ids: list[int], actor: str, request_id: str) -> None:
        timestamp = now_iso()
        with self._lock, self._transaction():
            for annotation_id in annotation_ids:
                self._connection.execute(
                    "INSERT INTO annotation_events (annotation_id, event_type, before_json, "
                    "after_json, actor, request_id, created_at) "
                    "VALUES (?, 'export', NULL, '{}', ?, ?, ?)",
                    (annotation_id, actor, request_id, timestamp),
                )

    # -- adjudication -----------------------------------------------------
    def upsert_adjudication(self, cohort_id: int, record: dict[str, Any]) -> None:
        with self._lock, self._transaction():
            self._connection.execute(
                "INSERT INTO adjudications (cohort_id, study_uid, landmark, decision, "
                "chosen_annotator_id, replacement_series_uid, replacement_sop_uid, "
                "replacement_row_px, replacement_column_px, replacement_patient_x_mm, "
                "replacement_patient_y_mm, replacement_patient_z_mm, reviewer, reason, "
                "source_revision_a, source_revision_b, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (cohort_id, study_uid, landmark) DO UPDATE SET "
                "decision=excluded.decision, chosen_annotator_id=excluded.chosen_annotator_id, "
                "replacement_series_uid=excluded.replacement_series_uid, "
                "replacement_sop_uid=excluded.replacement_sop_uid, "
                "replacement_row_px=excluded.replacement_row_px, "
                "replacement_column_px=excluded.replacement_column_px, "
                "replacement_patient_x_mm=excluded.replacement_patient_x_mm, "
                "replacement_patient_y_mm=excluded.replacement_patient_y_mm, "
                "replacement_patient_z_mm=excluded.replacement_patient_z_mm, "
                "reviewer=excluded.reviewer, reason=excluded.reason, "
                "source_revision_a=excluded.source_revision_a, "
                "source_revision_b=excluded.source_revision_b, created_at=excluded.created_at",
                (
                    cohort_id, record["study_uid"], record["landmark"], record["decision"],
                    record.get("chosen_annotator_id"), record.get("replacement_series_uid"),
                    record.get("replacement_sop_uid"), record.get("replacement_row_px"),
                    record.get("replacement_column_px"), record.get("replacement_patient_x_mm"),
                    record.get("replacement_patient_y_mm"), record.get("replacement_patient_z_mm"),
                    record["reviewer"], record.get("reason"), record.get("source_revision_a"),
                    record.get("source_revision_b"), now_iso(),
                ),
            )

    def list_adjudications(self, cohort_id: int) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM adjudications WHERE cohort_id = ?", (cohort_id,)
        ).fetchall()

    # -- helpers ------------------------------------------------------------
    def _transaction(self):
        return _Transaction(self._connection)


class _Transaction:
    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def __enter__(self):
        self._connection.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._connection.execute("COMMIT")
        else:
            self._connection.execute("ROLLBACK")
        return False


def _row_to_cohort(row: sqlite3.Row) -> Cohort:
    return Cohort(
        id=row["id"],
        name=row["name"],
        phase=row["phase"],
        manifest_path=row["manifest_path"],
        manifest_sha256=row["manifest_sha256"],
        template_path=row["template_path"],
        template_sha256=row["template_sha256"],
        state=row["state"],
    )


def copy_database_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
