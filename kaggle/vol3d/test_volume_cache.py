# -*- coding: utf-8 -*-
"""CPU unit tests for the DICOM-free core of build_volume_cache.py.

The DICOM reading / physical ordering can only be smoke-tested on Kaggle (needs the corpus), but the
resampling math and series-selection rule are pure and tested here on synthetic inputs."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "bvc", Path(__file__).resolve().parent / "build_volume_cache.py")
bvc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bvc)

FAIL = 0


def check(name, cond):
    global FAIL
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    FAIL += 0 if cond else 1


# --- pick_series_for_plane (pure) ---
rows = [{"SeriesInstanceUID": "b", "Fluid_Sensitive": 0},
        {"SeriesInstanceUID": "a", "Fluid_Sensitive": 1},
        {"SeriesInstanceUID": "c", "Fluid_Sensitive": 1}]
check("prefers a fluid-sensitive series", bvc.pick_series_for_plane(rows)["Fluid_Sensitive"] == 1)
check("fluid tie broken deterministically by UID",
      bvc.pick_series_for_plane(rows)["SeriesInstanceUID"] == "a")
check("no fluid-sensitive -> first by UID",
      bvc.pick_series_for_plane([{"SeriesInstanceUID": "z", "Fluid_Sensitive": 0},
                                 {"SeriesInstanceUID": "m", "Fluid_Sensitive": 0}])
      ["SeriesInstanceUID"] == "m")
check("empty -> None", bvc.pick_series_for_plane([]) is None)

# --- resample_volume (needs cv2) ---
try:
    import cv2  # noqa
    D, HW = bvc.DEPTH, bvc.HW
    rng = np.random.default_rng(0)

    many = [rng.uniform(0, 4000, (100, 120)).astype(np.float32) for _ in range(10)]
    v = bvc.resample_volume(many, D, HW)
    check("shape is [D,HW,HW]", v.shape == (D, HW, HW))
    check("dtype uint8", v.dtype == np.uint8)
    check("uses full 0..255 range on noise", v.min() == 0 and v.max() == 255)

    one = [np.full((64, 64), 1234.0, np.float32)]
    v1 = bvc.resample_volume(one, D, HW)
    check("single slice -> depth repeated, constant volume", v1.shape == (D, HW, HW))

    exactly = [rng.uniform(0, 1, (50, 50)).astype(np.float32) for _ in range(D)]
    vd = bvc.resample_volume(exactly, D, HW)
    check("exactly D slices -> depth preserved", vd.shape == (D, HW, HW))

    check("empty slices -> zeros", bvc.resample_volume([], D, HW).sum() == 0)
except ImportError:
    print("SKIP  resample_volume tests (cv2 not installed in this env; runs on Kaggle)")

print(f"\n{FAIL} failure(s)")
sys.exit(1 if FAIL else 0)
