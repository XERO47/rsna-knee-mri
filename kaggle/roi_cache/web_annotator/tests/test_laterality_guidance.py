from __future__ import annotations

import unittest
from types import SimpleNamespace

from kaggle.roi_cache.web_annotator.services import medial_image_side


class _Slice:
    rows = 100
    columns = 100

    def __init__(self, x_per_column: float, x_per_row: float = 0.0):
        self.x_per_column = x_per_column
        self.x_per_row = x_per_row

    def pixel_to_patient(self, row: float, column: float):
        return (self.x_per_column * column + self.x_per_row * row, 0.0, 0.0)


def task(x_per_column: float, x_per_row: float = 0.0, plane: str = "Coronal"):
    return SimpleNamespace(series=SimpleNamespace(plane=plane, middle=_Slice(x_per_column, x_per_row)))


class LateralityGuidanceTests(unittest.TestCase):
    def test_medial_edge_uses_dicom_patient_x_and_laterality(self):
        self.assertEqual(medial_image_side(task(1.0), "R"), "right")
        self.assertEqual(medial_image_side(task(1.0), "L"), "left")
        self.assertEqual(medial_image_side(task(-1.0), "R"), "left")
        self.assertEqual(medial_image_side(task(-1.0), "L"), "right")

    def test_guidance_is_withheld_if_horizontal_side_would_be_misleading(self):
        self.assertIsNone(medial_image_side(task(0.1, 1.0), "L"))
        self.assertIsNone(medial_image_side(task(1.0, plane="Sagittal"), "L"))
        self.assertIsNone(medial_image_side(task(1.0), None))


if __name__ == "__main__":
    unittest.main()
