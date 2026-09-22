"""Regression checks for the radiologist-facing static UI contract."""

from __future__ import annotations

import unittest
from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "static"


class StaticUiTests(unittest.TestCase):
    def test_landmark_chips_show_anatomy_names_not_state_words(self):
        script = (STATIC / "app.js").read_text()
        self.assertIn("landmarkLabel(entry.landmark)", script)
        self.assertNotIn("`${index + 1}. ${entry.state}`", script)

    def test_marker_is_large_target_with_contrast_outline(self):
        script = (STATIC / "app.js").read_text()
        self.assertIn("const armLength = 34 * dpr", script)
        self.assertIn("ctx.arc(x, y, outerRadius", script)
        self.assertIn('ctx.strokeStyle = "rgba(0, 0, 0, 0.9)"', script)

    def test_static_asset_version_is_bumped(self):
        markup = (STATIC / "index.html").read_text()
        self.assertIn("/static/app.css?v=5", markup)
        self.assertIn("/static/app.js?v=5", markup)

    def test_validation_error_is_not_mislabelled_as_offline(self):
        script = (STATIC / "app.js").read_text()
        self.assertIn('setSaveState("rejected")', script)
        self.assertIn("Point not saved:", script)
        markup = (STATIC / "index.html").read_text()
        self.assertIn('id="rejected-banner"', markup)

    def test_enlarged_mri_slices_use_high_quality_canvas_scaling(self):
        script = (STATIC / "app.js").read_text()
        self.assertIn("ctx.imageSmoothingEnabled = true", script)
        self.assertIn('ctx.imageSmoothingQuality = "high"', script)

    def test_laterality_guidance_and_old_marker_are_visible_during_correction(self):
        script = (STATIC / "app.js").read_text()
        self.assertIn('"LEFT KNEE"', script)
        self.assertIn("MEDIAL is on the", script)
        self.assertIn('drawCrosshair(x, y, "#3fbf7f")', script)
        self.assertNotIn("state.task.saved_point = null", script)


if __name__ == "__main__":
    unittest.main()
