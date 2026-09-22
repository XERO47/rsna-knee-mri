from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_roi_revision import description_plane, is_description_consistent, stratified_qc_rows


class RevisionPreparationTests(unittest.TestCase):
    def test_description_plane_requires_one_unambiguous_token(self):
        self.assertEqual(description_plane("pd_tse_fs_sag_320"), "Sagittal")
        self.assertEqual(description_plane("t2_pd_fs_TRA"), "Axial")
        self.assertEqual(description_plane("t1_tse_cor_320"), "Coronal")
        self.assertEqual(description_plane("cor_sag_localizer"), "Unknown")
        self.assertEqual(description_plane("routine knee"), "Unknown")

    def test_explicit_direction_conflict_is_rejected_but_unknown_is_deferred(self):
        self.assertTrue(is_description_consistent({
            "medial_joint_line": {"declared_plane": "Coronal", "description_plane": "Unknown"},
            "patellofemoral_center": {"declared_plane": "Axial", "description_plane": "Axial"},
        }))
        self.assertFalse(is_description_consistent({
            "medial_joint_line": {"declared_plane": "Coronal", "description_plane": "Sagittal"},
        }))

    def test_qc_includes_patient_one_excludes_replaced_case_and_adds_candidate(self):
        manifest = pd.DataFrame({
            "StudyInstanceUID": [f"study-{index}" for index in range(1, 101)],
            "manufacturer_bin": ["A", "B"] * 50,
            "laterality": ["L", "R"] * 50,
            "prior_partition": ["fit"] * 80 + ["audit"] * 20,
            "complete_preferred_planes": [True] * 80 + [False] * 20,
        })
        qc = stratified_qc_rows(manifest, "study-39", "candidate-1", 20)
        self.assertEqual(len(qc), 20)
        self.assertEqual(qc.iloc[0].StudyInstanceUID, "study-1")
        self.assertNotIn("study-39", qc.StudyInstanceUID.to_list())
        self.assertEqual(qc.iloc[-1].review_kind, "replacement_candidate")
        self.assertEqual(qc.iloc[-1].StudyInstanceUID, "candidate-1")


if __name__ == "__main__":
    unittest.main()
