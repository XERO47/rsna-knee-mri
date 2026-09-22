"""Server-side pixel/patient validation, without touching real DICOM files.

Uses synthetic geometry objects (the same pattern as the existing
``test_roi_geometry.make_slice`` helper) so these tests are fast and do not
depend on ``data-root`` being mounted.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_ROI_CACHE_DIR = Path(__file__).resolve().parents[2]
if str(_ROI_CACHE_DIR) not in sys.path:
    sys.path.insert(0, str(_ROI_CACHE_DIR))
import roi_geometry  # noqa: E402  (kaggle/roi_cache/roi_geometry.py, same import style as services.py)

from .. import services


def make_slice(sop_uid="sop-1", rows=64, columns=64, spacing=(0.5, 0.5)):
    return roi_geometry.SliceGeometry(
        path=Path(f"/does/not/matter/{sop_uid}.dcm"),
        sop_uid=sop_uid,
        ipp=np.array([0.0, 0.0, 0.0]),
        row_dir=np.array([1.0, 0.0, 0.0]),
        col_dir=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        pixel_spacing=spacing,
        rows=rows,
        columns=columns,
        position=0.0,
    )


def make_task(slices):
    series = roi_geometry.SeriesGeometry(
        study_uid="study-1",
        series_uid="series-1",
        plane="Coronal",
        fluid_sensitive=1,
        slices=tuple(slices),
    )
    return services.ResolvedTask(study_uid="study-1", landmark="medial_joint_line", series=series, series_row={})


class CoordinateValidationTests(unittest.TestCase):
    def setUp(self):
        self.slice_a = make_slice("sop-a")
        self.task = make_task([self.slice_a])

    def test_valid_click_round_trips_within_tolerance(self):
        science = services.ScientificService.__new__(services.ScientificService)
        point, sop_uid, index = science.compute_and_validate_point(self.task, "sop-a", 10.4, 20.7)
        self.assertEqual(sop_uid, "sop-a")
        self.assertEqual(index, 0)
        expected = self.slice_a.pixel_to_patient(10.4, 20.7)
        self.assertTrue(np.allclose(point, expected))

    def test_non_finite_pixel_is_rejected(self):
        science = services.ScientificService.__new__(services.ScientificService)
        with self.assertRaises(services.OutOfBounds):
            science.compute_and_validate_point(self.task, "sop-a", math.nan, 1.0)
        with self.assertRaises(services.OutOfBounds):
            science.compute_and_validate_point(self.task, "sop-a", math.inf, 1.0)

    def test_out_of_bounds_pixel_is_rejected(self):
        science = services.ScientificService.__new__(services.ScientificService)
        with self.assertRaises(services.OutOfBounds):
            science.compute_and_validate_point(self.task, "sop-a", -1.0, 1.0)
        with self.assertRaises(services.OutOfBounds):
            science.compute_and_validate_point(self.task, "sop-a", 64.0, 1.0)

    def test_unknown_sop_uid_is_rejected(self):
        science = services.ScientificService.__new__(services.ScientificService)
        with self.assertRaises(services.OutOfBounds):
            science.compute_and_validate_point(self.task, "not-a-real-sop", 1.0, 1.0)

    def test_broken_inverse_fails_round_trip_gate(self):
        class BrokenSlice:
            sop_uid = "sop-broken"
            rows = 10
            columns = 10

            def pixel_to_patient(self, row, column):
                return np.array([float(row), float(column), 0.0])

            def patient_to_pixel(self, point):
                # Deliberately wrong inverse to exercise the rejection path.
                return point[0] + 5.0, point[1] + 5.0

        task = make_task([BrokenSlice()])
        science = services.ScientificService.__new__(services.ScientificService)
        with self.assertRaises(services.RoundTripValidationFailed):
            science.compute_and_validate_point(task, "sop-broken", 3.0, 3.0)


if __name__ == "__main__":
    unittest.main()
