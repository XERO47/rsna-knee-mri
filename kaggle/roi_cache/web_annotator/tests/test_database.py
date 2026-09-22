import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from .. import auth
from ..database import CohortClosed, Database, ManifestHashMismatch, RevisionConflict


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="web_annotator_db_test_"))
        self.db = Database(self.tmp_dir / "annotator.sqlite3")

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_cohort(self, name="pilot"):
        return self.db.create_or_resume_cohort(
            name, "pilot", "/manifest.csv", "hash-m", "/template.csv", "hash-t"
        )


class SchemaAndCohortTests(DatabaseTestCase):
    def test_journal_mode_is_wal(self):
        mode = self.db._connection.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode.lower(), "wal")

    def test_foreign_keys_enabled(self):
        value = self.db._connection.execute("PRAGMA foreign_keys").fetchone()[0]
        self.assertEqual(value, 1)

    def test_create_then_resume_same_hash_succeeds(self):
        first = self._make_cohort()
        second = self.db.create_or_resume_cohort(
            "pilot", "pilot", "/manifest.csv", "hash-m", "/template.csv", "hash-t"
        )
        self.assertEqual(first.id, second.id)

    def test_resume_with_changed_manifest_hash_refuses(self):
        self._make_cohort()
        with self.assertRaises(ManifestHashMismatch):
            self.db.create_or_resume_cohort(
                "pilot", "pilot", "/manifest.csv", "DIFFERENT-HASH", "/template.csv", "hash-t"
            )

    def test_close_cohort_is_idempotent_guarded(self):
        self._make_cohort()
        self.db.close_cohort("pilot")
        with self.assertRaises(ValueError):
            self.db.close_cohort("pilot")


class AssignmentAndAnnotationTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.cohort = self._make_cohort()
        self.db.replace_studies(self.cohort.id, [
            {"StudyInstanceUID": "study-1", "laterality": "L"},
            {"StudyInstanceUID": "study-2", "laterality": "R"},
        ])
        self.annotator_id, self.token = self.db.create_annotator("reader-a")

    def test_assign_creates_incomplete_annotation_rows(self):
        self.db.assign_studies(self.cohort.id, self.annotator_id, ["study-1"], "single_reader", ["lm1", "lm2"])
        rows = self.db.list_annotations_for_reader(self.cohort.id, self.annotator_id)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["state"] == "incomplete" for row in rows))
        self.assertTrue(all(row["revision"] == 0 for row in rows))

    def test_assign_rejects_study_outside_cohort(self):
        with self.assertRaises(ValueError):
            self.db.assign_studies(self.cohort.id, self.annotator_id, ["not-in-cohort"], "single_reader", ["lm1"])

    def test_save_point_bumps_revision_and_records_event(self):
        self.db.assign_studies(self.cohort.id, self.annotator_id, ["study-1"], "single_reader", ["lm1"])
        record = self.db.save_point(
            self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", "req-1", 0,
            "series-1", "sop-1", 10.0, 20.0, 1.0, 2.0, 3.0, "L", "open",
        )
        self.assertEqual(record["state"], "pass")
        self.assertEqual(record["revision"], 1)
        events = self.db.list_events(record["id"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "create")

    def test_replace_after_pass_logs_replace_event(self):
        self.db.assign_studies(self.cohort.id, self.annotator_id, ["study-1"], "single_reader", ["lm1"])
        first = self.db.save_point(
            self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", "req-1", 0,
            "series-1", "sop-1", 10.0, 20.0, 1.0, 2.0, 3.0, "L", "open",
        )
        second = self.db.save_point(
            self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", "req-2", first["revision"],
            "series-1", "sop-2", 11.0, 21.0, 4.0, 5.0, 6.0, "L", "open",
        )
        events = self.db.list_events(second["id"])
        self.assertEqual([event["event_type"] for event in events], ["create", "replace"])

    def test_stale_revision_is_rejected(self):
        self.db.assign_studies(self.cohort.id, self.annotator_id, ["study-1"], "single_reader", ["lm1"])
        self.db.save_point(
            self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", "req-1", 0,
            "series-1", "sop-1", 10.0, 20.0, 1.0, 2.0, 3.0, "L", "open",
        )
        with self.assertRaises(RevisionConflict):
            self.db.save_point(
                self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", "req-2", 0,
                "series-1", "sop-2", 11.0, 21.0, 4.0, 5.0, 6.0, "L", "open",
            )

    def test_idempotent_retry_returns_same_committed_record_without_new_event(self):
        self.db.assign_studies(self.cohort.id, self.annotator_id, ["study-1"], "single_reader", ["lm1"])
        first = self.db.save_point(
            self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", "req-1", 0,
            "series-1", "sop-1", 10.0, 20.0, 1.0, 2.0, 3.0, "L", "open",
        )
        retry = self.db.save_point(
            self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", "req-1", 0,
            "series-1", "sop-1", 10.0, 20.0, 1.0, 2.0, 3.0, "L", "open",
        )
        self.assertEqual(first, retry)
        events = self.db.list_events(first["id"])
        self.assertEqual(len(events), 1)

    def test_mark_unable_clears_coordinates_and_does_not_pass(self):
        self.db.assign_studies(self.cohort.id, self.annotator_id, ["study-1"], "single_reader", ["lm1"])
        record = self.db.mark_unable(
            self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", "req-u1", 0,
            "anatomy_not_visible", "cannot see it", "open",
        )
        self.assertEqual(record["state"], "unable")
        self.assertIsNone(record["patient_x_mm"])
        self.assertIsNone(record["sop_uid"])

    def test_closed_cohort_rejects_new_saves(self):
        self.db.assign_studies(self.cohort.id, self.annotator_id, ["study-1"], "single_reader", ["lm1"])
        with self.assertRaises(CohortClosed):
            self.db.save_point(
                self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", "req-1", 0,
                "series-1", "sop-1", 10.0, 20.0, 1.0, 2.0, 3.0, "L", "closed",
            )

    def test_reopen_resets_state_and_appends_history(self):
        self.db.assign_studies(self.cohort.id, self.annotator_id, ["study-1"], "single_reader", ["lm1"])
        saved = self.db.save_point(
            self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", "req-1", 0,
            "series-1", "sop-1", 10.0, 20.0, 1.0, 2.0, 3.0, "L", "open",
        )
        reopened = self.db.reopen(self.cohort.id, self.annotator_id, "study-1", "lm1", "admin", "correction")
        self.assertEqual(reopened["state"], "incomplete")
        events = self.db.list_events(saved["id"])
        self.assertEqual([event["event_type"] for event in events], ["create", "reopen"])

    def test_concurrent_saves_no_lost_update(self):
        self.db.assign_studies(self.cohort.id, self.annotator_id, ["study-1"], "single_reader", ["lm1"])
        results = {}

        def attempt(name, row_value):
            try:
                results[name] = self.db.save_point(
                    self.cohort.id, self.annotator_id, "study-1", "lm1", "reader-a", f"req-{name}", 0,
                    "series-1", "sop-1", row_value, 20.0, 1.0, 2.0, 3.0, "L", "open",
                )
            except RevisionConflict as conflict:
                results[name] = conflict

        threads = [threading.Thread(target=attempt, args=(name, float(index))) for index, name in enumerate(["a", "b", "c"])]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        successes = [value for value in results.values() if not isinstance(value, RevisionConflict)]
        conflicts = [value for value in results.values() if isinstance(value, RevisionConflict)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(conflicts), 2)
        final = self.db.get_annotation(self.cohort.id, self.annotator_id, "study-1", "lm1")
        self.assertEqual(final["revision"], 1)


class SessionTests(DatabaseTestCase):
    def test_authenticate_bootstrap_token_and_session_lifecycle(self):
        annotator_id, token = self.db.create_annotator("reader-a")
        self.assertIsNotNone(self.db.authenticate_bootstrap_token(token))
        self.assertIsNone(self.db.authenticate_bootstrap_token("wrong-token"))
        session_token = self.db.create_session(annotator_id, lifetime_days=1)
        resolved = self.db.resolve_session(session_token)
        self.assertEqual(resolved["annotator_id"], annotator_id)
        self.db.delete_session(session_token)
        self.assertIsNone(self.db.resolve_session(session_token))

    def test_revoke_deactivates_token_and_deletes_sessions(self):
        annotator_id, token = self.db.create_annotator("reader-a")
        session_token = self.db.create_session(annotator_id, lifetime_days=1)
        self.db.revoke_annotator("reader-a")
        self.assertIsNone(self.db.authenticate_bootstrap_token(token))
        self.assertIsNone(self.db.resolve_session(session_token))


class BackupTests(DatabaseTestCase):
    def test_backup_creates_file_and_rotates(self):
        self._make_cohort()
        backup_dir = self.tmp_dir / "backups"
        for index in range(3):
            self.db.backup(backup_dir, f"reason-{index}", max_backups=2)
        backups = sorted(backup_dir.glob("annotator-*.sqlite3"))
        self.assertEqual(len(backups), 2)


if __name__ == "__main__":
    unittest.main()
