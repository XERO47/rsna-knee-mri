"""Restart/resume behavior: the server must recover solely from SQLite."""

import unittest
import uuid

from fastapi.testclient import TestClient

from . import _fixtures
from .. import services


@unittest.skipUnless(_fixtures.HAS_REAL_DATA, "real data-root / pilot manifest not available")
class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.harness = _fixtures.RealDataHarness()
        self.study_uids = self.harness.all_study_uids()[:2]
        self.annotator_id, self.token = self.harness.create_reader("reader-a")
        self.harness.assign(self.annotator_id, self.study_uids, group="single_reader")

    def tearDown(self):
        self.harness.close()

    def _client(self):
        return TestClient(self.harness.make_app(), base_url="https://testserver")

    def _login(self, client, token):
        client.get(f"/login?token={token}", follow_redirects=False)

    def test_resume_lands_on_first_incomplete_landmark_after_a_partial_save(self):
        client = self._client()
        self._login(client, self.token)
        task = client.get("/api/task").json()
        self.assertEqual(task["landmark_index"], 0)
        slice_response = client.get(f"/api/slice?task_id={task['task_id']}&index=0")
        client.post("/api/annotations/save", json={
            "task_id": task["task_id"],
            "sop_uid": slice_response.headers["X-Slice-Sop-Uid"],
            "row_px": int(slice_response.headers["X-Slice-Rows"]) / 2.0,
            "column_px": int(slice_response.headers["X-Slice-Columns"]) / 2.0,
            "expected_revision": task["revision"],
            "request_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        })

        # Simulate a server/process restart: a brand new Database connection
        # and app instance against the exact same sqlite file on disk, with
        # no in-memory state carried over.
        self.harness.db.close()
        self.harness.db = self.harness.db.__class__(self.harness.config.db_path)
        restarted_client = TestClient(self.harness.make_app(), base_url="https://testserver")
        self._login(restarted_client, self.token)
        resumed_task = restarted_client.get("/api/task").json()

        self.assertEqual(resumed_task["patient_index"], 0)
        self.assertEqual(resumed_task["landmark_index"], 1)
        self.assertEqual(resumed_task["landmark_states"][0]["state"], "pass")

    def test_saved_task_is_editable_but_not_counted_incomplete(self):
        client = self._client()
        self._login(client, self.token)
        for landmark in services.LANDMARKS:
            task = client.get(f"/api/task?landmark={landmark}").json()
            slice_response = client.get(f"/api/slice?task_id={task['task_id']}&index=0")
            response = client.post("/api/annotations/save", json={
                "task_id": task["task_id"],
                "sop_uid": slice_response.headers["X-Slice-Sop-Uid"],
                "row_px": int(slice_response.headers["X-Slice-Rows"]) / 2.0,
                "column_px": int(slice_response.headers["X-Slice-Columns"]) / 2.0,
                "expected_revision": task["revision"],
                "request_id": str(uuid.uuid4()),
            })
            self.assertEqual(response.status_code, 200)

        progress = client.get("/api/progress").json()
        self.assertEqual(progress["completed_patient_count"], 1)
        self.assertEqual(progress["remaining_patient_count"], 1)

        # A completed patient's task must still be reachable and editable.
        task = client.get(f"/api/task?patient_index=0&landmark={services.LANDMARKS[0]}").json()
        self.assertEqual(task["state"], "pass")


if __name__ == "__main__":
    unittest.main()
