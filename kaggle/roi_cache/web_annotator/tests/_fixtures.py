"""Shared test fixtures: real pilot manifest/data-root, temp DB per test."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .. import cli, services
from ..app import create_app
from ..config import AppConfig
from ..database import Database

REPO_ROOT = Path(__file__).resolve().parents[4]
COMPETITION_ROOT = (REPO_ROOT.parent / "data-root").resolve()
PILOT_MANIFEST = REPO_ROOT / "roi_work" / "pilot" / "pilot_manifest.csv"
PILOT_TEMPLATE = REPO_ROOT / "roi_work" / "pilot" / "pilot_annotations.csv"

HAS_REAL_DATA = (
    (COMPETITION_ROOT / "train_series.csv").is_file()
    and PILOT_MANIFEST.is_file()
    and PILOT_TEMPLATE.is_file()
)


class RealDataHarness:
    """Initializes a temporary DB against the real frozen 20-study pilot."""

    def __init__(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="web_annotator_test_"))
        self.config = AppConfig.from_work_dir(COMPETITION_ROOT, self.tmp_dir / "work")
        self.config.ensure_directories()
        self.db = Database(self.config.db_path)
        self.cohort_name = "pilot-test"
        args = SimpleNamespace(
            work_dir=str(self.config.work_dir),
            competition_root=str(COMPETITION_ROOT),
            cohort=self.cohort_name,
            phase="pilot",
            manifest=str(PILOT_MANIFEST),
            template=str(PILOT_TEMPLATE),
        )
        cli.command_init_cohort(args)
        self.cohort = self.db.get_cohort(self.cohort_name)
        self.science = services.ScientificService(COMPETITION_ROOT)

    def create_reader(self, name: str) -> tuple[int, str]:
        return self.db.create_annotator(name)

    def assign(self, annotator_id: int, study_uids: list[str], group: str = "single_reader") -> None:
        self.db.assign_studies(self.cohort.id, annotator_id, study_uids, group, services.LANDMARKS)

    def all_study_uids(self) -> list[str]:
        return [row["study_uid"] for row in self.db.list_studies(self.cohort.id)]

    def make_app(self):
        return create_app(self.config, self.db, self.science, self.cohort_name)

    def close(self) -> None:
        self.db.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
