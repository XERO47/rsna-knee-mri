"""Single-account launcher setup and safe resume."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ..config import AppConfig
from ..database import Database
from ..services import LANDMARKS
from ..single_user import prepare_single_user


class SingleUserSetupTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="roi_single_user_"))
        self.work = self.root / "work"
        self.manifest = self.root / "manifest.csv"
        self.template = self.root / "template.csv"
        study_ids = [f"study-{index:02d}" for index in range(20)]
        pd.DataFrame({
            "StudyInstanceUID": study_ids,
            "laterality": ["L", "R"] * 10,
        }).to_csv(self.manifest, index=False)
        pd.DataFrame([
            {"StudyInstanceUID": study_uid, "landmark": landmark}
            for study_uid in study_ids
            for landmark in LANDMARKS
        ]).to_csv(self.template, index=False)
        self.config = AppConfig.from_work_dir(self.root / "data", self.work)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _prepare(self):
        return prepare_single_user(
            self.config,
            "pilot",
            "pilot",
            self.manifest,
            self.template,
            "http://127.0.0.1:8000",
        )

    def test_first_run_creates_one_account_and_assigns_every_study(self):
        setup = self._prepare()
        self.assertTrue(setup.created)
        self.assertEqual(setup.study_count, 20)
        self.assertIn("/login?token=", setup.login_url)
        self.assertEqual(os.stat(setup.token_path).st_mode & 0o777, 0o600)

        db = Database(self.config.db_path)
        try:
            cohort = db.get_cohort("pilot")
            accounts = db.list_annotators()
            self.assertEqual(len(accounts), 1)
            assignments = db.list_assignments(cohort.id, int(accounts[0]["id"]))
            annotations = db.list_annotations_for_reader(cohort.id, int(accounts[0]["id"]))
            self.assertEqual(len(assignments), 20)
            self.assertEqual(len(annotations), 20 * len(LANDMARKS))
        finally:
            db.close()

    def test_second_run_reuses_account_token_and_records(self):
        first = self._prepare()
        second = self._prepare()
        self.assertFalse(second.created)
        self.assertEqual(first.login_url, second.login_url)
        db = Database(self.config.db_path)
        try:
            self.assertEqual(len(db.list_annotators()), 1)
        finally:
            db.close()

    def test_missing_private_token_is_rotated_without_new_account(self):
        first = self._prepare()
        first.token_path.unlink()
        second = self._prepare()
        self.assertNotEqual(first.login_url, second.login_url)
        db = Database(self.config.db_path)
        try:
            self.assertEqual(len(db.list_annotators()), 1)
        finally:
            db.close()

    def test_refuses_a_database_with_multiple_accounts(self):
        self._prepare()
        db = Database(self.config.db_path)
        try:
            db.create_annotator("unexpected second account")
        finally:
            db.close()
        with self.assertRaisesRegex(RuntimeError, "more than one account"):
            self._prepare()


if __name__ == "__main__":
    unittest.main()
