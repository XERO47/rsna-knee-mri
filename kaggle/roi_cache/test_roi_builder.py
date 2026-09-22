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

from build_roi_cache import (
    PROVENANCE_FIELDS,
    _audit_coverage,
    _crop_center,
    _side_independent_window_strategy,
    _window_candidates,
    build_study,
)
from roi_geometry import TARGET_SPECS


class BuilderFailureTests(unittest.TestCase):
    def test_laterality_dependent_targets_always_get_two_side_hypotheses(self):
        priors = {
            "landmarks": {
                specification["landmark"]: {"median_offset_mm": [12.0, 3.0, 4.0]}
                for specification in TARGET_SPECS.values()
            }
        }
        candidates = _window_candidates(
            TARGET_SPECS["mcl"],
            priors,
            np.zeros(3),
            "UNKNOWN",
        )
        self.assertEqual([side for side, _ in candidates], ["L", "R"])
        self.assertAlmostEqual(candidates[0][1][0], -12.0)
        self.assertAlmostEqual(candidates[1][1][0], 12.0)

    def test_side_independent_target_ignores_laterality(self):
        priors = {
            "landmarks": {
                "patellofemoral_center": {"median_offset_mm": [9.0, 3.0, 4.0]}
            }
        }
        left = _window_candidates(TARGET_SPECS["pf_oa"], priors, np.zeros(3), "L")
        right = _window_candidates(TARGET_SPECS["pf_oa"], priors, np.zeros(3), "R")
        self.assertEqual(left[0][0], "CENTER")
        np.testing.assert_array_equal(left[0][1], right[0][1])
        self.assertAlmostEqual(left[0][1][0], 0.0)

    def test_synovitis_declares_approved_asymmetric_sagittal_policy(self):
        specification = TARGET_SPECS["synovitis"]
        self.assertEqual(
            _side_independent_window_strategy(specification),
            "nearest_distinct_triplets",
        )
        landmark = np.asarray([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(
            _crop_center(specification, "Axial", landmark),
            landmark,
        )
        np.testing.assert_array_equal(
            _crop_center(specification, "Sagittal", landmark),
            [4.0, 5.0, -19.0],
        )

    def test_mirror_audit_compares_landmarks_not_origin_sign(self):
        offsets = {
            study: {
                "medial_joint_line": np.asarray([8.0, 0.0, 0.0]),
                "lateral_joint_line": np.asarray([2.0, 0.0, 0.0]),
                "patellofemoral_center": np.zeros(3),
                "suprapatellar_recess": np.zeros(3),
            }
            for study in ("a", "b", "c")
        }
        audit = _audit_coverage(
            ["a", "b", "c"],
            ["a", "b", "c"],
            offsets,
            {"a": "L", "b": "R", "c": "L"},
            leave_one_out=True,
        )
        self.assertEqual(audit["mirror_errors"], 0)

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
