"""One-account setup used by the simple web annotator launcher.

The database still uses assignments internally to preserve the existing API
and authorization invariants. In single-account mode they are an implementation
detail: the only account automatically receives every study in the cohort.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import auth, services
from .config import AppConfig, FINAL_STUDY_COUNT, PILOT_STUDY_COUNT
from .database import Database, ManifestHashMismatch

_ROI_CACHE_DIR = Path(__file__).resolve().parents[1]
import sys

if str(_ROI_CACHE_DIR) not in sys.path:
    sys.path.insert(0, str(_ROI_CACHE_DIR))

import build_roi_cache  # noqa: E402


DEFAULT_READER_NAME = "Joint Radiology Review"
TOKEN_FILE_NAME = "private_login_token.txt"


@dataclass(frozen=True)
class SingleUserSetup:
    cohort_name: str
    reader_name: str
    study_count: int
    login_url: str
    token_path: Path
    created: bool


def _write_private_token(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(token + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _validated_manifest(path: Path, phase: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    expected = PILOT_STUDY_COUNT if phase == "pilot" else FINAL_STUDY_COUNT
    if len(frame) != expected:
        raise ValueError(f"{phase} manifest must contain exactly {expected} studies")
    if frame.StudyInstanceUID.duplicated().any():
        raise ValueError("manifest contains duplicate StudyInstanceUID values")
    return frame


def prepare_single_user(
    config: AppConfig,
    cohort_name: str,
    phase: str,
    manifest_path: Path,
    template_path: Path,
    base_url: str,
) -> SingleUserSetup:
    """Create or safely resume one cohort with exactly one shared account."""
    config.ensure_directories()
    manifest_path = manifest_path.resolve()
    template_path = template_path.resolve()
    manifest = _validated_manifest(manifest_path, phase)
    if not template_path.is_file():
        raise FileNotFoundError(f"annotation template not found: {template_path}")

    manifest_hash = build_roi_cache.sha256_file(manifest_path)
    template_hash = build_roi_cache.sha256_file(template_path)
    db = Database(config.db_path)
    try:
        existing = db.get_cohort(cohort_name)
        try:
            cohort = db.create_or_resume_cohort(
                cohort_name,
                phase,
                str(manifest_path),
                manifest_hash,
                str(template_path),
                template_hash,
            )
        except ManifestHashMismatch as error:
            raise RuntimeError(str(error)) from error
        if cohort.phase != phase:
            raise RuntimeError(
                f"cohort {cohort_name!r} is phase {cohort.phase!r}, not {phase!r}"
            )
        if cohort.state != "open":
            raise RuntimeError(f"cohort {cohort_name!r} is closed and cannot be resumed")

        if existing is None:
            db.replace_studies(cohort.id, manifest.to_dict("records"))
        else:
            stored_ids = [str(row["study_uid"]) for row in db.list_studies(cohort.id)]
            manifest_ids = manifest.StudyInstanceUID.astype(str).tolist()
            if stored_ids != manifest_ids:
                raise RuntimeError(
                    "stored study order does not exactly match the frozen manifest; "
                    "refusing to alter an existing cohort"
                )

        accounts = db.list_annotators()
        if len(accounts) > 1:
            raise RuntimeError(
                "single-account mode found more than one account. Use a fresh --work-dir "
                "or archive the older multi-reader database before continuing."
            )

        created = not accounts
        token_path = config.work_dir / TOKEN_FILE_NAME
        if created:
            annotator_id, token = db.create_annotator(DEFAULT_READER_NAME)
            reader_name = DEFAULT_READER_NAME
            _write_private_token(token_path, token)
        else:
            account = accounts[0]
            annotator_id = int(account["id"])
            reader_name = str(account["display_name"])
            token = token_path.read_text().strip() if token_path.is_file() else ""
            authenticated = db.authenticate_bootstrap_token(token) if token else None
            if authenticated is None or int(authenticated["id"]) != annotator_id:
                token = db.rotate_annotator_token(annotator_id)
                _write_private_token(token_path, token)

        study_ids = [str(row["study_uid"]) for row in db.list_studies(cohort.id)]
        db.assign_studies(
            cohort.id,
            annotator_id,
            study_ids,
            "single_reader",
            services.LANDMARKS,
        )
        login_url = f"{base_url.rstrip('/')}/login?token={token}"
        return SingleUserSetup(
            cohort_name=cohort_name,
            reader_name=reader_name,
            study_count=len(study_ids),
            login_url=login_url,
            token_path=token_path,
            created=created,
        )
    finally:
        db.close()
