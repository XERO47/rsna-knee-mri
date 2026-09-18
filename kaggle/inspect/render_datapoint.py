"""Render real slices from two example studies so we can see what a training
data point actually looks like: a pool study (soft pseudo-labels, from-report)
and a gold study (real 0/1 labels). CPU-only, no GPU needed -- just reads the
competition's mounted DICOM files and saves PNG grids to /kaggle/working.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

STUDIES = {
    "pool": "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260",
    "gold": "1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149",
}
N_SLICES = 5


def find_root(input_root: Path, marker: str, skip_dirs=("train_series", "test_series")) -> Path:
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{marker!r} not found under {input_root}")


COMP_ROOT = find_root(Path("/kaggle/input"), "train_series.csv")
SERIES_ROOT = COMP_ROOT / "train_series"
WORK = Path("/kaggle/working")

import pandas as pd
series_df = pd.read_csv(COMP_ROOT / "train_series.csv")


def sorted_series_files(series_dir: Path):
    files = list(series_dir.glob("*.dcm"))
    tagged = []
    for f in files:
        try:
            n = pydicom.dcmread(str(f), stop_before_pixels=True).InstanceNumber
        except Exception:
            n = 0
        tagged.append((int(n), f))
    tagged.sort(key=lambda t: t[0])
    return [f for _, f in tagged]


def load_slice(path: Path) -> np.ndarray:
    dcm = pydicom.dcmread(str(path))
    arr = dcm.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    arr -= arr.min()
    denom = arr.max()
    if denom > 0:
        arr /= denom
    return arr


for tag, uid in STUDIES.items():
    rows = series_df[series_df.StudyInstanceUID == uid].sort_values("Anatomical_Plane")
    n_series = len(rows)
    fig, axes = plt.subplots(n_series, N_SLICES, figsize=(N_SLICES * 2.4, n_series * 2.4))
    if n_series == 1:
        axes = axes[None, :]

    for r, (_, row) in enumerate(rows.iterrows()):
        series_uid = row["SeriesInstanceUID"]
        plane = row["Anatomical_Plane"]
        fs = "FS" if row["Fluid_Sensitive"] else "non-FS"
        fatsat = "fatsat" if row["Fat_Suppression"] else "no-fatsat"

        series_dir = SERIES_ROOT / uid / series_uid
        files = sorted_series_files(series_dir)
        n = len(files)
        print(f"{tag} {plane} {fs}: {n} slices")

        fracs = np.linspace(0.10, 0.90, N_SLICES)
        idxs = np.clip((fracs * (n - 1)).round().astype(int), 0, n - 1)

        for c, fi in enumerate(idxs):
            ax = axes[r, c]
            try:
                img = load_slice(files[fi])
                ax.imshow(img, cmap="gray")
            except Exception as e:
                ax.text(0.5, 0.5, f"err\n{e}", ha="center", va="center", fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{plane}\n{fs}, {fatsat}\n(n={n})", fontsize=9, rotation=0,
                              ha="right", va="center")

    fig.suptitle(f"{tag.upper()} study {uid[-12:]}", fontsize=11)
    fig.tight_layout()
    out = WORK / f"{tag}_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)
    plt.close(fig)

print("done")
