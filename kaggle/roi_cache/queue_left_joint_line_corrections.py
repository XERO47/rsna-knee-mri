"""Queue only left-knee medial/lateral landmarks for audited re-annotation.

The live rows become ``incomplete`` so they cannot be exported, but each old
point is retained in SQLite's append-only event history and displayed amber by
the web app. The two centreline landmarks remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_annotator.config import AppConfig
from web_annotator.database import Database
from web_annotator.single_user import DEFAULT_READER_NAME


LANDMARKS_TO_CORRECT = ("medial_joint_line", "lateral_joint_line")


def main(arguments: argparse.Namespace) -> None:
    config = AppConfig.from_work_dir(arguments.competition_root, arguments.work_dir)
    db = Database(config.db_path)
    try:
        cohort = db.get_cohort(arguments.cohort)
        reader = db.get_annotator_by_name(arguments.reader)
        if cohort is None or reader is None:
            raise ValueError("cohort or shared reader does not exist")
        if cohort.state != "open":
            raise ValueError("cohort is closed; it cannot be queued for correction")
        study_side = {row["study_uid"]: row["laterality"] for row in db.list_studies(cohort.id)}
        rows = [dict(row) for row in db.list_annotations_for_reader(cohort.id, int(reader["id"]))]
        selected = [
            row for row in rows
            if study_side.get(row["study_uid"]) == "L"
            and row["landmark"] in LANDMARKS_TO_CORRECT
        ]
        if len(selected) != 100:
            raise ValueError(f"expected 100 left joint-line rows, found {len(selected)}")
        if any(row["state"] != "pass" for row in selected):
            raise ValueError("all selected rows must currently be passing; refusing a partial/repeated queue")
        backup = db.backup(config.backup_dir, "pre-left-joint-line-correction", config.max_backups)
        for row in selected:
            db.reopen(
                cohort.id,
                int(reader["id"]),
                row["study_uid"],
                row["landmark"],
                actor="cli:left-joint-line-correction",
                reason="left_laterality_orientation_audit",
            )
        report = {
            "status": "queued_for_radiologist_correction",
            "cohort": arguments.cohort,
            "side": "L",
            "landmarks": list(LANDMARKS_TO_CORRECT),
            "reopened_tasks": len(selected),
            "unchanged_tasks": len(rows) - len(selected),
            "backup": str(backup),
            "old_points": "preserved in annotation_events and rendered amber by the web application",
        }
        output = config.work_dir / "left_joint_line_correction_queue.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        db.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--cohort", default="final-r1")
    parser.add_argument("--reader", default=DEFAULT_READER_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
