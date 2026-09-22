"""Create a new ROI annotation revision without altering the source cohort.

This is intentionally a one-way copy of reviewed *pass* annotations.  It
creates a fresh single-user database from a revised 100-study manifest, copies
only the 99 studies shared by source and target, and leaves the replacement's
four landmarks incomplete for the radiologist.  The source SQLite database is
opened read-only in practice: this script never calls a source mutator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from web_annotator import services
from web_annotator.config import AppConfig
from web_annotator.database import Database
from web_annotator.single_user import DEFAULT_READER_NAME, prepare_single_user


def _source_rows(
    source_db: Database,
    source_cohort: str,
    source_reader: str,
) -> tuple[object, int, dict[tuple[str, str], dict[str, object]]]:
    cohort = source_db.get_cohort(source_cohort)
    if cohort is None:
        raise ValueError(f"source cohort {source_cohort!r} does not exist")
    reader = source_db.get_annotator_by_name(source_reader)
    if reader is None:
        raise ValueError(f"source reader {source_reader!r} does not exist")
    rows = [dict(row) for row in source_db.list_annotations_for_reader(cohort.id, int(reader["id"]))]
    return cohort, int(reader["id"]), {(str(row["study_uid"]), str(row["landmark"])): row for row in rows}


def migrate(arguments: argparse.Namespace) -> None:
    source_config = AppConfig.from_work_dir(arguments.competition_root, arguments.source_work_dir)
    target_config = AppConfig.from_work_dir(arguments.competition_root, arguments.target_work_dir)
    manifest_path = Path(arguments.manifest).resolve()
    template_path = Path(arguments.template).resolve()
    if target_config.db_path.exists():
        raise ValueError(
            f"target database already exists: {target_config.db_path}; refusing to merge into it"
        )
    manifest = pd.read_csv(manifest_path, dtype={"StudyInstanceUID": str})
    if len(manifest) != 100 or manifest.StudyInstanceUID.nunique() != 100:
        raise ValueError("revision manifest must contain exactly 100 unique studies")

    source_db = Database(source_config.db_path)
    target_db: Database | None = None
    try:
        source, _, source_by_key = _source_rows(source_db, arguments.source_cohort, arguments.source_reader)
        source_uids = {str(row["study_uid"]) for row in source_db.list_studies(source.id)}
        target_uids = set(manifest.StudyInstanceUID.astype(str))
        shared_uids = source_uids & target_uids
        removed_uids = source_uids - target_uids
        added_uids = target_uids - source_uids
        if len(source_uids) != 100 or len(shared_uids) != 99 or len(removed_uids) != 1 or len(added_uids) != 1:
            raise ValueError(
                "revision must replace exactly one source study: "
                f"source={len(source_uids)}, shared={len(shared_uids)}, "
                f"removed={len(removed_uids)}, added={len(added_uids)}"
            )
        for study_uid in sorted(shared_uids):
            for landmark in services.LANDMARKS:
                row = source_by_key.get((study_uid, landmark))
                if row is None or row["state"] != "pass":
                    raise ValueError(f"reviewed source row is not pass: {study_uid}/{landmark}")

        setup = prepare_single_user(
            config=target_config,
            cohort_name=arguments.target_cohort,
            phase="final",
            manifest_path=manifest_path,
            template_path=template_path,
            base_url=arguments.base_url,
        )
        target_db = Database(target_config.db_path)
        target = target_db.get_cohort(arguments.target_cohort)
        reader = target_db.get_annotator_by_name(DEFAULT_READER_NAME)
        if target is None or reader is None:
            raise RuntimeError("fresh revision setup did not create its cohort and shared account")

        copied = 0
        for study_uid in sorted(shared_uids):
            for landmark in services.LANDMARKS:
                source_row = source_by_key[(study_uid, landmark)]
                target_db.save_point(
                    cohort_id=target.id,
                    annotator_id=int(reader["id"]),
                    study_uid=study_uid,
                    landmark=landmark,
                    actor=f"migration:{arguments.source_cohort}->{arguments.target_cohort}",
                    request_id=f"migration-v1:{study_uid}:{landmark}",
                    expected_revision=0,
                    series_uid=str(source_row["series_uid"]),
                    sop_uid=str(source_row["sop_uid"]),
                    row_px=float(source_row["row_px"]),
                    column_px=float(source_row["column_px"]),
                    patient_x_mm=float(source_row["patient_x_mm"]),
                    patient_y_mm=float(source_row["patient_y_mm"]),
                    patient_z_mm=float(source_row["patient_z_mm"]),
                    laterality=source_row["laterality"],
                    cohort_state=target.state,
                )
                copied += 1

        annotations = [dict(row) for row in target_db.list_annotations_for_reader(target.id, int(reader["id"]))]
        copied_rows = [row for row in annotations if row["study_uid"] in shared_uids and row["state"] == "pass"]
        pending_rows = [row for row in annotations if row["study_uid"] in added_uids and row["state"] == "incomplete"]
        if len(copied_rows) != 99 * len(services.LANDMARKS) or len(pending_rows) != len(services.LANDMARKS):
            raise RuntimeError("post-migration state is not exactly 396 pass rows plus 4 replacement tasks")
        report = {
            "status": "ready_for_replacement_annotation",
            "source_cohort": arguments.source_cohort,
            "target_cohort": arguments.target_cohort,
            "removed_study_uid": next(iter(removed_uids)),
            "replacement_study_uid": next(iter(added_uids)),
            "copied_pass_annotations": copied,
            "replacement_incomplete_tasks": len(pending_rows),
            "source_database": str(source_config.db_path.resolve()),
            "target_database": str(target_config.db_path.resolve()),
            "login_url": setup.login_url,
            "note": "Original source database was not modified by this migration.",
        }
        report_path = target_config.work_dir / "revision_migration.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        source_db.close()
        if target_db is not None:
            target_db.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", required=True)
    parser.add_argument("--source-work-dir", required=True)
    parser.add_argument("--source-cohort", default="final")
    parser.add_argument("--source-reader", default=DEFAULT_READER_NAME)
    parser.add_argument("--target-work-dir", required=True)
    parser.add_argument("--target-cohort", default="final-r1")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    return parser.parse_args()


if __name__ == "__main__":
    migrate(parse_arguments())
