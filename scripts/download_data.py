"""Download the RSNA Knee Abnormality Detection competition data.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME /
KAGGLE_KEY env vars), and that the competition rules have been accepted on
kaggle.com.

Usage:
    python scripts/download_data.py [--dest data/raw] [--unzip]
"""
import argparse
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "rsna-knee-abnormality-detection"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/raw", help="Download destination")
    parser.add_argument("--unzip", action="store_true", help="Unzip archives after download")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(COMPETITION, path=str(dest), unzip=args.unzip)


if __name__ == "__main__":
    main()
