from __future__ import annotations

import tempfile
import unittest
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


if importlib.util.find_spec("cv2") is None:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.setNumThreads = lambda _: None
    cv2_stub.INTER_LINEAR = 1
    sys.modules["cv2"] = cv2_stub

if importlib.util.find_spec("pydicom") is None:
    pydicom_stub = types.ModuleType("pydicom")
    handlers_stub = types.ModuleType("pydicom.pixel_data_handlers")
    util_stub = types.ModuleType("pydicom.pixel_data_handlers.util")
    util_stub.apply_modality_lut = lambda pixels, dataset: pixels
    pydicom_stub.pixel_data_handlers = handlers_stub
    handlers_stub.util = util_stub
    sys.modules["pydicom"] = pydicom_stub
    sys.modules["pydicom.pixel_data_handlers"] = handlers_stub
    sys.modules["pydicom.pixel_data_handlers.util"] = util_stub

from build_roi_cache import PROVENANCE_FIELDS, build_study
from roi_geometry import TARGET_SPECS


class BuilderFailureTests(unittest.TestCase):
    def test_missing_series_produces_zero_windows_and_complete_provenance(self):
        priors = {
            "landmarks": {
                specification["landmark"]: {"median_offset_mm": [0.0, 0.0, 0.0]}
                for specification in TARGET_SPECS.values()
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            data, masks, provenance = build_study(
                "missing-study",
                [],
                Path(directory),
                priors,
            )
        for target in TARGET_SPECS:
            self.assertEqual(data[target].shape, (4, 3, 336, 336))
            self.assertEqual(data[target].dtype, np.uint8)
            self.assertFalse(data[target].any())
            self.assertEqual(masks[target].tolist(), [0, 0, 0, 0])
        self.assertEqual(len(provenance), 20)
        self.assertTrue(all(set(record) == set(PROVENANCE_FIELDS) for record in provenance))
        self.assertTrue(all(record["failure_reason"] == "no_valid_selected_series" for record in provenance))


if __name__ == "__main__":
    unittest.main()
