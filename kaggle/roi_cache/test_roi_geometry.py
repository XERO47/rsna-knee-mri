from __future__ import annotations

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from roi_geometry import (
    SliceGeometry,
    assert_exact_id_alignment,
    canonical_axes,
    centered_triplet_indices,
    crop_bounds,
    nearest_distinct_triplet_indices,
    padding_fraction,
    pick_series,
    resolve_laterality,
    sort_deduplicate_slices,
    triplet_indices,
)


def make_slice(
    name: str,
    position: float,
    iop=(1, 0, 0, 0, 1, 0),
    spacing=(1.0, 1.0),
    rows=200,
    columns=200,
) -> SliceGeometry:
    row = np.asarray(iop[:3], dtype=float)
    column = np.asarray(iop[3:], dtype=float)
    normal = np.cross(row, column)
    header = SimpleNamespace(
        ImageOrientationPatient=iop,
        ImagePositionPatient=(normal * position).tolist(),
        PixelSpacing=spacing,
        Rows=rows,
        Columns=columns,
        SOPInstanceUID=name,
    )
    return SliceGeometry.from_header(Path(name + ".dcm"), header)


class GeometryTests(unittest.TestCase):
    def test_pixel_patient_round_trip(self):
        item = make_slice("one", 17.0, spacing=(0.7, 0.8))
        point = item.pixel_to_patient(41.25, 73.5)
        row, column = item.patient_to_pixel(point)
        self.assertAlmostEqual(row, 41.25, places=10)
        self.assertAlmostEqual(column, 73.5, places=10)

    def test_left_and_right_medial_axes(self):
        left_medial, anterior, superior = canonical_axes("L")
        right_medial, _, _ = canonical_axes("R")
        np.testing.assert_array_equal(left_medial, [-1, 0, 0])
        np.testing.assert_array_equal(right_medial, [1, 0, 0])
        np.testing.assert_array_equal(anterior, [0, -1, 0])
        np.testing.assert_array_equal(superior, [0, 0, 1])
        self.assertEqual(resolve_laterality(["L", "LEFT", "R"])[0], "L")
        self.assertEqual(resolve_laterality(["L", "R"]), ("UNKNOWN", "conflict"))

    def test_reversed_filenames_sort_by_physical_position(self):
        values = [make_slice("z", 1.0), make_slice("a", 3.0), make_slice("m", 2.0)]
        result = sort_deduplicate_slices(values)
        self.assertEqual([item.position for item in result], [1.0, 2.0, 3.0])
        self.assertEqual([item.sop_uid for item in result], ["z", "m", "a"])

    def test_position_deduplication_uses_point_one_mm(self):
        values = [
            make_slice("a", 0.0),
            make_slice("b", 0.09),
            make_slice("c", 1.0),
            make_slice("d", 2.0),
        ]
        result = sort_deduplicate_slices(values)
        self.assertEqual([item.sop_uid for item in result], ["a", "c", "d"])

    def test_reversed_orientation_is_rejected(self):
        values = [
            make_slice("a", 0.0),
            make_slice("b", 1.0, iop=(-1, 0, 0, 0, -1, 0)),
            make_slice("c", 2.0),
        ]
        with self.assertRaisesRegex(ValueError, "orientation"):
            sort_deduplicate_slices(values)

    def test_oblique_projection_round_trip(self):
        root = np.sqrt(0.5)
        item = make_slice("oblique", 12.0, iop=(root, root, 0, -root, root, 0))
        point = item.pixel_to_patient(91.75, 13.125)
        row, column = item.patient_to_pixel(point)
        self.assertAlmostEqual(row, 91.75, places=9)
        self.assertAlmostEqual(column, 13.125, places=9)

    def test_near_orthogonal_header_uses_a_true_in_plane_inverse(self):
        # A real-world IOP may be accepted by the 1-degree geometry gate yet
        # have a tiny row/column dot product due to header rounding.  The
        # inverse must still round-trip a click far from the image origin.
        item = make_slice(
            "rounded-iop",
            12.0,
            iop=(1.0, 0.0, 0.0, 0.0002, 1.0, 0.0),
            spacing=(0.7, 0.8),
        )
        point = item.pixel_to_patient(300.25, 250.75)
        row, column = item.patient_to_pixel(point)
        self.assertAlmostEqual(row, 300.25, places=10)
        self.assertAlmostEqual(column, 250.75, places=10)

    def test_crop_extent_and_padding(self):
        item = make_slice("crop", 0.0)
        center = item.pixel_to_patient(100, 100)
        bounds = crop_bounds(item, center, "Axial", (20.0, 10.0), "R")
        self.assertEqual(bounds, (95, 105, 90, 110))
        self.assertEqual(padding_fraction(bounds, item.rows, item.columns), 0.0)
        self.assertAlmostEqual(padding_fraction((-5, 5, 90, 110), 200, 200), 0.5)

    def test_missing_series_and_boundary_windows_are_invalid(self):
        rows = [{"Anatomical_Plane": "Coronal", "Fluid_Sensitive": 0, "SeriesInstanceUID": "x"}]
        self.assertIsNone(pick_series(rows, "Axial", 1))
        self.assertEqual(triplet_indices(0, 8), (None, (0, 1, 2)))
        self.assertEqual(triplet_indices(7, 8), ((5, 6, 7), None))
        self.assertEqual(centered_triplet_indices(4, 8), (3, 4, 5))
        self.assertIsNone(centered_triplet_indices(0, 8))
        self.assertIsNone(centered_triplet_indices(7, 8))

    def test_nearest_distinct_triplets_stay_in_bounds_without_duplication(self):
        self.assertEqual(
            nearest_distinct_triplet_indices(0, 8),
            ((0, 1, 2), (1, 2, 3)),
        )
        self.assertEqual(
            nearest_distinct_triplet_indices(7, 8),
            ((5, 6, 7), (4, 5, 6)),
        )
        self.assertEqual(
            nearest_distinct_triplet_indices(3, 3),
            ((0, 1, 2), None),
        )

    def test_series_selection_is_deterministic_and_exact(self):
        rows = [
            {"Anatomical_Plane": "Coronal", "Fluid_Sensitive": 1, "SeriesInstanceUID": "first"},
            {"Anatomical_Plane": "Coronal", "Fluid_Sensitive": 1, "SeriesInstanceUID": "second"},
            {"Anatomical_Plane": "Coronal", "Fluid_Sensitive": 0, "SeriesInstanceUID": "structural"},
        ]
        self.assertEqual(pick_series(rows, "Coronal", 1)["SeriesInstanceUID"], "first")
        self.assertEqual(pick_series(rows, "Coronal", 0)["SeriesInstanceUID"], "structural")
        axial = [{"Anatomical_Plane": "Axial", "Fluid_Sensitive": 0, "SeriesInstanceUID": "fallback"}]
        self.assertEqual(pick_series(axial, "Axial", 1)["SeriesInstanceUID"], "fallback")
        self.assertIsNone(pick_series(axial, "Axial", 1, allow_fluid_fallback=False))

    def test_exact_id_alignment_rejects_reordering(self):
        assert_exact_id_alignment(["a", "b"], ["a", "b"])
        with self.assertRaises(ValueError):
            assert_exact_id_alignment(["a", "b"], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
