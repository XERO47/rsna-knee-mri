"""Integration tests against the real frozen 20-study pilot data-root.

Skipped automatically if ../data-root or the frozen pilot manifest/template
are not present, so the suite stays runnable without the 530 GiB dataset.
"""

import unittest

from fastapi.testclient import TestClient

from . import _fixtures
from .. import services


@unittest.skipUnless(_fixtures.HAS_REAL_DATA, "real data-root / pilot manifest not available")
class ApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.harness = _fixtures.RealDataHarness()
        study_uids = self.harness.all_study_uids()
        self.study_a, self.study_b = study_uids[0], study_uids[1]
        self.annotator_a_id, self.token_a = self.harness.create_reader("reader-a")
        self.annotator_b_id, self.token_b = self.harness.create_reader("reader-b")
        self.harness.assign(self.annotator_a_id, [self.study_a], group="single_reader")
        self.harness.assign(self.annotator_b_id, [self.study_b], group="single_reader")
        # Secure cookies are only ever sent back over https; simulate the TLS
        # termination that Cloudflare Tunnel (or the localhost secure-context
        # exception) provides in real deployments.
        self.client = TestClient(self.harness.make_app(), base_url="https://testserver")

    def tearDown(self):
        self.harness.close()

    def _login(self, token):
        response = self.client.get(f"/login?token={token}", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertNotIn(token, response.headers.get("location", ""))
        return response.cookies

    def test_health_is_unauthenticated_and_minimal(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_root_requires_session(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 401)

    def test_invalid_bootstrap_token_is_rejected(self):
        response = self.client.get("/login?token=not-a-real-token")
        self.assertEqual(response.status_code, 401)

    def test_login_exchanges_token_for_cookie_and_hides_it_from_url(self):
        cookies = self._login(self.token_a)
        self.assertIn("roi_session", cookies)
        response = self.client.get("/api/me")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reader"], "reader-a")

    def test_api_requires_session(self):
        response = self.client.get("/api/task")
        self.assertEqual(response.status_code, 401)

    def test_reader_cannot_access_other_readers_task(self):
        self._login(self.token_a)
        task = self.client.get("/api/task").json()
        self.assertEqual(task["study_uid"], self.study_a)

        # Reader A's opaque task_id must not authorize as reader B.
        self.client.cookies.clear()
        self._login(self.token_b)
        response = self.client.get(f"/api/slice?task_id={task['task_id']}&index=0")
        self.assertEqual(response.status_code, 403)

    def test_slice_endpoint_returns_lossless_png_with_geometry_headers(self):
        self._login(self.token_a)
        task = self.client.get("/api/task").json()
        self.assertIsNotNone(task["series"])
        response = self.client.get(f"/api/slice?task_id={task['task_id']}&index=0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertTrue(int(response.headers["X-Slice-Rows"]) > 0)
        self.assertTrue(int(response.headers["X-Slice-Columns"]) > 0)
        self.assertEqual(response.content[:8], b"\x89PNG\r\n\x1a\n")

    def test_save_computes_patient_xyz_server_side_and_round_trips(self):
        self._login(self.token_a)
        task = self.client.get("/api/task").json()
        slice_response = self.client.get(f"/api/slice?task_id={task['task_id']}&index=0")
        sop_uid = slice_response.headers["X-Slice-Sop-Uid"]
        rows = int(slice_response.headers["X-Slice-Rows"])
        columns = int(slice_response.headers["X-Slice-Columns"])
        response = self.client.post("/api/annotations/save", json={
            "task_id": task["task_id"],
            "sop_uid": sop_uid,
            "row_px": rows / 2.0,
            "column_px": columns / 2.0,
            "expected_revision": task["revision"],
            "request_id": "11111111-1111-4111-8111-111111111111",
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["saved"])
        self.assertEqual(body["annotation"]["state"], "pass")
        for key in ("patient_x_mm", "patient_y_mm", "patient_z_mm"):
            self.assertIsInstance(body["annotation"][key], float)
        # The client never sent a patient coordinate; the server must have
        # computed it, and re-requesting the task should reflect the save.
        refreshed = self.client.get(f"/api/task?landmark={task['landmark']}").json()
        self.assertEqual(refreshed["state"], "pass")
        self.assertEqual(refreshed["revision"], 1)

    def test_stale_revision_returns_conflict_and_does_not_overwrite(self):
        self._login(self.token_a)
        task = self.client.get("/api/task").json()
        slice_response = self.client.get(f"/api/slice?task_id={task['task_id']}&index=0")
        sop_uid = slice_response.headers["X-Slice-Sop-Uid"]
        rows = int(slice_response.headers["X-Slice-Rows"])
        columns = int(slice_response.headers["X-Slice-Columns"])
        payload = {
            "task_id": task["task_id"], "sop_uid": sop_uid,
            "row_px": rows / 2.0, "column_px": columns / 2.0,
            "expected_revision": task["revision"], "request_id": "22222222-2222-4222-8222-222222222222",
        }
        first = self.client.post("/api/annotations/save", json=payload)
        self.assertEqual(first.status_code, 200)
        stale_payload = dict(payload, row_px=rows / 4.0, request_id="33333333-3333-4333-8333-333333333333")
        second = self.client.post("/api/annotations/save", json=stale_payload)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(second.json()["error"], "stale_revision")

    def test_idempotent_retry_of_same_request_id_is_a_no_op(self):
        self._login(self.token_a)
        task = self.client.get("/api/task").json()
        slice_response = self.client.get(f"/api/slice?task_id={task['task_id']}&index=0")
        sop_uid = slice_response.headers["X-Slice-Sop-Uid"]
        rows = int(slice_response.headers["X-Slice-Rows"])
        columns = int(slice_response.headers["X-Slice-Columns"])
        payload = {
            "task_id": task["task_id"], "sop_uid": sop_uid,
            "row_px": rows / 2.0, "column_px": columns / 2.0,
            "expected_revision": task["revision"], "request_id": "44444444-4444-4444-8444-444444444444",
        }
        first = self.client.post("/api/annotations/save", json=payload)
        retry = self.client.post("/api/annotations/save", json=payload)
        self.assertEqual(first.json(), retry.json())

    def test_unable_reason_is_required_and_not_recorded_as_pass(self):
        self._login(self.token_a)
        task = self.client.get("/api/task").json()
        response = self.client.post("/api/annotations/unable", json={
            "task_id": task["task_id"], "reason": "anatomy_not_visible", "notes": None,
            "expected_revision": task["revision"], "request_id": "55555555-5555-4555-8555-555555555555",
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()["annotation"]
        self.assertEqual(body["state"], "unable")
        self.assertIsNone(body["patient_x_mm"])

    def test_unable_rejects_unknown_reason(self):
        self._login(self.token_a)
        task = self.client.get("/api/task").json()
        response = self.client.post("/api/annotations/unable", json={
            "task_id": task["task_id"], "reason": "not-a-real-reason",
            "expected_revision": task["revision"], "request_id": "66666666-6666-4666-8666-666666666666",
        })
        self.assertEqual(response.status_code, 422)

    def test_path_traversal_on_static_is_rejected(self):
        response = self.client.get("/static/../../../etc/passwd")
        self.assertIn(response.status_code, (400, 403, 404))

    def test_task_id_cannot_be_forged_to_escape_assignment(self):
        self._login(self.token_a)
        import base64
        import json as json_module
        forged = base64.urlsafe_b64encode(
            json_module.dumps({"su": self.study_b, "lm": services.LANDMARKS[0]}).encode()
        ).decode().rstrip("=")
        response = self.client.get(f"/api/slice?task_id={forged}&index=0")
        self.assertEqual(response.status_code, 403)

    def test_logout_clears_session(self):
        self._login(self.token_a)
        self.client.post("/logout")
        response = self.client.get("/api/me")
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
