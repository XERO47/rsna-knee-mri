"""Per-reader export, canonical export, agreement report, and cache limits."""

import json
import unittest
from types import SimpleNamespace

import pandas as pd

from . import _fixtures
from .. import cli, services
from ..database import ManifestHashMismatch


@unittest.skipUnless(_fixtures.HAS_REAL_DATA, "real data-root / pilot manifest not available")
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.harness = _fixtures.RealDataHarness()

    def tearDown(self):
        self.harness.close()

    def _save_center_point(self, annotator_id, study_uid, landmark):
        task = self.harness.science.resolve_task(study_uid, landmark)
        middle_index = len(task.series.slices) // 2
        item = task.series.slices[middle_index]
        # Real IOP vectors are only guaranteed orthogonal to within the 1e-3
        # degree gate in roi_geometry.SliceGeometry.from_header, so the exact
        # image center can occasionally miss the desktop annotator's 1e-5 px
        # round-trip tolerance too (the affine round trip error grows with
        # distance from IPP). Try a few candidate points, closest-to-center
        # first, exactly as a radiologist would just click again nearby.
        candidates = [
            (item.rows / 2.0, item.columns / 2.0),
            (item.rows / 4.0, item.columns / 4.0),
            (item.rows / 8.0, item.columns / 8.0),
            (item.rows / 16.0, item.columns / 16.0),
            (item.rows / 32.0, item.columns / 32.0),
            (item.rows / 64.0, item.columns / 64.0),
            (item.rows / 128.0, item.columns / 128.0),
            (2.0, 2.0),
        ]
        last_error = None
        for row_px, column_px in candidates:
            try:
                point, sop_uid, _ = self.harness.science.compute_and_validate_point(
                    task, item.sop_uid, row_px, column_px
                )
                break
            except services.RoundTripValidationFailed as error:
                last_error = error
        else:
            raise last_error
        annotation = self.harness.db.get_annotation(self.harness.cohort.id, annotator_id, study_uid, landmark)
        return self.harness.db.save_point(
            self.harness.cohort.id, annotator_id, study_uid, landmark, "test",
            f"seed-{annotator_id}-{study_uid}-{landmark}",
            annotation["revision"], task.series.series_uid, sop_uid, row_px, column_px,
            float(point[0]), float(point[1]), float(point[2]),
            self.harness.db.get_study(self.harness.cohort.id, study_uid)["laterality"], "open",
        )

    def test_export_reader_has_exact_twelve_column_schema_and_atomicity(self):
        annotator_id, _ = self.harness.create_reader("solo")
        study_uid = self.harness.all_study_uids()[0]
        self.harness.assign(annotator_id, [study_uid], group="single_reader")
        for landmark in services.LANDMARKS[:2]:
            self._save_center_point(annotator_id, study_uid, landmark)

        output = self.harness.tmp_dir / "reader_export.csv"
        cli.command_export_reader(SimpleNamespace(
            work_dir=str(self.harness.config.work_dir),
            competition_root=str(_fixtures.COMPETITION_ROOT),
            cohort=self.harness.cohort_name,
            reader="solo",
            output=str(output),
        ))
        self.assertTrue(output.exists())
        frame = pd.read_csv(output, dtype={"StudyInstanceUID": str})
        self.assertEqual(list(frame.columns), cli.EXPORT_COLUMNS)
        self.assertEqual(len(frame), len(services.LANDMARKS))
        pass_rows = frame[frame.qc_status == "pass"]
        incomplete_rows = frame[frame.qc_status == "incomplete"]
        self.assertEqual(len(pass_rows), 2)
        self.assertEqual(len(incomplete_rows), 2)
        self.assertTrue(incomplete_rows[["row_px", "column_px", "patient_x_mm"]].isna().all().all())

        metadata = json.loads(output.with_suffix(".csv.meta.json").read_text())
        self.assertEqual(metadata["pass_count"], 2)
        self.assertEqual(metadata["incomplete_count"], 2)

    def test_export_reader_rejects_stored_point_that_fails_recomputed_round_trip(self):
        annotator_id, _ = self.harness.create_reader("solo")
        study_uid = self.harness.all_study_uids()[0]
        self.harness.assign(annotator_id, [study_uid], group="single_reader")
        self._save_center_point(annotator_id, study_uid, services.LANDMARKS[0])
        # Corrupt the stored coordinate directly to simulate bit-rot / a bug
        # that produced a value inconsistent with the resolved series.
        self.harness.db._connection.execute(
            "UPDATE annotations SET patient_x_mm = patient_x_mm + 500 WHERE study_uid = ?",
            (study_uid,),
        )
        with self.assertRaises(ValueError):
            cli.command_export_reader(SimpleNamespace(
                work_dir=str(self.harness.config.work_dir),
                competition_root=str(_fixtures.COMPETITION_ROOT),
                cohort=self.harness.cohort_name,
                reader="solo",
                output=str(self.harness.tmp_dir / "bad_export.csv"),
            ))

    def test_full_pilot_canonical_export_has_exactly_eighty_rows(self):
        annotator_id, _ = self.harness.create_reader("solo")
        study_uids = self.harness.all_study_uids()
        self.harness.assign(annotator_id, study_uids, group="single_reader")
        for study_uid in study_uids:
            for landmark in services.LANDMARKS:
                self._save_center_point(annotator_id, study_uid, landmark)

        output = self.harness.tmp_dir / "canonical.csv"
        cli.command_canonical_export(SimpleNamespace(
            work_dir=str(self.harness.config.work_dir),
            competition_root=str(_fixtures.COMPETITION_ROOT),
            cohort=self.harness.cohort_name,
            output=str(output),
        ))
        frame = pd.read_csv(output, dtype={"StudyInstanceUID": str})
        self.assertEqual(len(frame), 80)
        self.assertEqual(len(frame), len(frame.drop_duplicates(["StudyInstanceUID", "landmark"])))
        self.assertTrue((frame.qc_status == "pass").all())
        self.assertTrue(frame[["patient_x_mm", "patient_y_mm", "patient_z_mm"]].notna().all().all())
        self.assertEqual(set(frame.landmark), set(services.LANDMARKS))

    def test_canonical_export_requires_adjudication_for_overlap_studies(self):
        annotator_a, _ = self.harness.create_reader("reader-a")
        annotator_b, _ = self.harness.create_reader("reader-b")
        study_uids = self.harness.all_study_uids()
        overlap_study = study_uids[0]
        self.harness.assign(annotator_a, [overlap_study], group="overlap")
        self.harness.assign(annotator_b, [overlap_study], group="overlap")
        for landmark in services.LANDMARKS:
            self._save_center_point(annotator_a, overlap_study, landmark)
            self._save_center_point(annotator_b, overlap_study, landmark)

        with self.assertRaises(ValueError):
            cli.command_canonical_export(SimpleNamespace(
                work_dir=str(self.harness.config.work_dir),
                competition_root=str(_fixtures.COMPETITION_ROOT),
                cohort=self.harness.cohort_name,
                output=str(self.harness.tmp_dir / "canonical_incomplete.csv"),
            ))

        for landmark in services.LANDMARKS:
            cli.command_adjudicate(SimpleNamespace(
                work_dir=str(self.harness.config.work_dir),
                competition_root=str(_fixtures.COMPETITION_ROOT),
                cohort=self.harness.cohort_name,
                study=overlap_study,
                landmark=landmark,
                decision="choose_reader",
                reviewer="owner",
                reason="identical synthetic points",
                chosen_reader="reader-a",
                replacement_sop_uid=None,
                replacement_row=None,
                replacement_column=None,
            ))
        remaining = study_uids[1:]
        self.harness.assign(annotator_a, remaining, group="single_reader")
        for study_uid in remaining:
            for landmark in services.LANDMARKS:
                self._save_center_point(annotator_a, study_uid, landmark)

        output = self.harness.tmp_dir / "canonical_adjudicated.csv"
        cli.command_canonical_export(SimpleNamespace(
            work_dir=str(self.harness.config.work_dir),
            competition_root=str(_fixtures.COMPETITION_ROOT),
            cohort=self.harness.cohort_name,
            output=str(output),
        ))
        frame = pd.read_csv(output, dtype={"StudyInstanceUID": str})
        self.assertEqual(len(frame), 80)
        overlap_rows = frame[frame.StudyInstanceUID == overlap_study]
        self.assertTrue((overlap_rows.annotator == "reader-a").all())

    def test_agreement_report_reports_distance_without_a_pass_fail_threshold(self):
        annotator_a, _ = self.harness.create_reader("reader-a")
        annotator_b, _ = self.harness.create_reader("reader-b")
        study_uid = self.harness.all_study_uids()[0]
        self.harness.assign(annotator_a, [study_uid], group="overlap")
        self.harness.assign(annotator_b, [study_uid], group="overlap")
        for landmark in services.LANDMARKS:
            self._save_center_point(annotator_a, study_uid, landmark)
            self._save_center_point(annotator_b, study_uid, landmark)

        output = self.harness.tmp_dir / "agreement.csv"
        cli.command_agreement_report(SimpleNamespace(
            work_dir=str(self.harness.config.work_dir),
            competition_root=str(_fixtures.COMPETITION_ROOT),
            cohort=self.harness.cohort_name,
            output=str(output),
        ))
        frame = pd.read_csv(output)
        self.assertEqual(len(frame), len(services.LANDMARKS))
        self.assertTrue((frame.distance_mm >= 0).all())
        summary = json.loads(output.with_suffix(".csv.summary.json").read_text())
        for landmark in services.LANDMARKS:
            self.assertIn("median_mm", summary[landmark])
        self.assertNotIn("pass_threshold_mm", json.dumps(summary))

    def test_manifest_hash_change_refuses_resume(self):
        modified_manifest = self.harness.tmp_dir / "modified_manifest.csv"
        frame = pd.read_csv(_fixtures.PILOT_MANIFEST, dtype={"StudyInstanceUID": str})
        frame.to_csv(modified_manifest, index=False)
        cli.command_init_cohort(SimpleNamespace(
            work_dir=str(self.harness.config.work_dir),
            competition_root=str(_fixtures.COMPETITION_ROOT),
            cohort="second-cohort",
            phase="pilot",
            manifest=str(modified_manifest),
            template=str(_fixtures.PILOT_TEMPLATE),
        ))
        modified_manifest.write_text(modified_manifest.read_text() + "\n# tampered\n")
        with self.assertRaises(SystemExit):
            cli.command_init_cohort(SimpleNamespace(
                work_dir=str(self.harness.config.work_dir),
                competition_root=str(_fixtures.COMPETITION_ROOT),
                cohort="second-cohort",
                phase="pilot",
                manifest=str(modified_manifest),
                template=str(_fixtures.PILOT_TEMPLATE),
            ))


class SliceCacheLimitTests(unittest.TestCase):
    def test_bounded_byte_cache_evicts_oldest_beyond_item_cap(self):
        cache = services._BoundedByteCache(max_items=2, max_bytes=10_000)
        cache.put("a", b"1")
        cache.put("b", b"2")
        cache.put("c", b"3")
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))
        self.assertIsNotNone(cache.get("c"))
        self.assertEqual(len(cache), 2)

    def test_bounded_byte_cache_evicts_beyond_byte_cap(self):
        cache = services._BoundedByteCache(max_items=100, max_bytes=10)
        cache.put("a", b"12345")
        cache.put("b", b"12345")
        cache.put("c", b"12345")
        self.assertLessEqual(cache.total_bytes, 10)


if __name__ == "__main__":
    unittest.main()
