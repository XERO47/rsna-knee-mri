"""Dataset utilities for the RSNA Knee Abnormality Detection competition.

Data layout on disk (after scripts/download_data.py --unzip):
    data/raw/train.csv
    data/raw/train_series.csv
    data/raw/train_series/<study_uid>/<series_uid>/<instance_uid>.dcm
    data/raw/test.csv
    data/raw/test_series.csv
    data/raw/test_series/<study_uid>/<series_uid>/<instance_uid>.dcm

Exact column names are confirmed once the data is downloaded; this module
reads the CSVs dynamically rather than hardcoding a schema.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset


def read_dicom_as_array(path: Path) -> np.ndarray:
    dcm = pydicom.dcmread(str(path))
    arr = dcm.pixel_array.astype(np.float32)
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr /= max_val
    return arr


def list_series_files(series_dir: Path) -> list[Path]:
    return sorted(series_dir.glob("*.dcm"))


class KneeSeriesDataset(Dataset):
    """Loads one representative slice (or stack) per series.

    Args:
        series_csv: path to train_series.csv / test_series.csv
        series_root: path to the directory holding <study>/<series>/*.dcm
        labels_csv: optional path to train.csv for supervised training
        transform: optional callable applied to the image array
    """

    def __init__(self, series_csv, series_root, labels_csv=None, transform=None):
        self.series_df = pd.read_csv(series_csv)
        self.series_root = Path(series_root)
        self.labels_df = pd.read_csv(labels_csv) if labels_csv else None
        self.transform = transform

    def __len__(self):
        return len(self.series_df)

    def __getitem__(self, idx):
        row = self.series_df.iloc[idx]
        study_uid = row[[c for c in row.index if "study" in c.lower()][0]]
        series_uid = row[[c for c in row.index if "series" in c.lower()][0]]

        series_dir = self.series_root / str(study_uid) / str(series_uid)
        files = list_series_files(series_dir)
        if not files:
            raise FileNotFoundError(f"No DICOM files found in {series_dir}")

        mid_slice = files[len(files) // 2]
        image = read_dicom_as_array(mid_slice)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).unsqueeze(0)

        sample = {"image": image, "study_uid": study_uid, "series_uid": series_uid}

        if self.labels_df is not None:
            match = self.labels_df[
                self.labels_df.apply(lambda r: str(study_uid) in r.values, axis=1)
            ]
            if not match.empty:
                sample["label"] = match.iloc[0].to_dict()

        return sample
