"""Merge two validated ROI shards into one Arm C-compatible Kaggle dataset.

The merged row order is deliberately ``roi shard 0`` followed by ``roi shard
1``.  This is the same order used by Arm C after it concatenates the two
global-cache ID vectors; it is *not* raw alternating ``train.csv`` order.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from roi_geometry import TARGET_SPECS


WINDOWS = 4
IMAGE_TAIL = (4, 3, 336, 336)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_array(source0: Path, source1: Path, destination: Path, chunk: int) -> tuple[int, ...]:
    first = np.load(source0, mmap_mode="r")
    second = np.load(source1, mmap_mode="r")
    if first.dtype != second.dtype or first.shape[1:] != second.shape[1:]:
        raise ValueError(f"incompatible shard arrays: {source0.name}, {source1.name}")
    output = open_memmap(destination, mode="w+", dtype=first.dtype, shape=(len(first) + len(second), *first.shape[1:]))
    cursor = 0
    for source in (first, second):
        for start in range(0, len(source), chunk):
            stop = min(start + chunk, len(source))
            output[cursor + start:cursor + stop] = source[start:stop]
        cursor += len(source)
    output.flush()
    del output
    return (len(first) + len(second), *first.shape[1:])


def verify_array(source0: Path, source1: Path, merged: Path, chunk: int) -> None:
    output = np.load(merged, mmap_mode="r")
    cursor = 0
    for source_path in (source0, source1):
        source = np.load(source_path, mmap_mode="r")
        for start in range(0, len(source), chunk):
            stop = min(start + chunk, len(source))
            if not np.array_equal(output[cursor + start:cursor + stop], source[start:stop]):
                raise RuntimeError(f"merged content mismatch: {merged.name} at row {cursor + start}")
        cursor += len(source)


def merge_provenance(source0: Path, source1: Path, destination: Path) -> int:
    rows = 0
    with gzip.open(destination, "wt", newline="") as output:
        writer = None
        for source in (source0, source1):
            with gzip.open(source, "rt", newline="") as input_stream:
                reader = csv.DictReader(input_stream)
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                    writer.writeheader()
                elif reader.fieldnames != writer.fieldnames:
                    raise ValueError("ROI provenance schemas differ between shards")
                for row in reader:
                    writer.writerow(row)
                    rows += 1
    return rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-studies", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    source = arguments.cache_dir.resolve()
    output = arguments.output_dir.resolve()
    if source == output:
        raise ValueError("output-dir must be different from cache-dir")
    if arguments.chunk_studies < 1:
        raise ValueError("chunk-studies must be positive")
    output.mkdir(parents=True, exist_ok=True)
    ids0 = np.load(source / "roi_ids_shard0.npy", allow_pickle=True).astype(str)
    ids1 = np.load(source / "roi_ids_shard1.npy", allow_pickle=True).astype(str)
    ids = np.concatenate([ids0, ids1])
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("ROI IDs are duplicated across shards")
    np.save(output / "roi_ids.npy", ids)

    arrays: dict[str, dict[str, object]] = {}
    for target in TARGET_SPECS:
        for suffix in ("", "_masks"):
            name0 = source / f"roi_{target}{suffix}_shard0.npy"
            name1 = source / f"roi_{target}{suffix}_shard1.npy"
            merged = output / f"roi_{target}{suffix}.npy"
            shape = copy_array(name0, name1, merged, arguments.chunk_studies)
            verify_array(name0, name1, merged, arguments.chunk_studies)
            arrays.setdefault(target, {})["data" if not suffix else "masks"] = {
                "file": merged.name,
                "shape": shape,
                "sha256": sha256(merged),
            }
        data = np.load(output / f"roi_{target}.npy", mmap_mode="r")
        mask = np.load(output / f"roi_{target}_masks.npy", mmap_mode="r")
        if data.shape != (len(ids), *IMAGE_TAIL) or data.dtype != np.uint8:
            raise RuntimeError(f"invalid merged ROI tensor for {target}: {data.shape}/{data.dtype}")
        if mask.shape != (len(ids), WINDOWS) or mask.dtype != np.uint8 or not np.isin(np.unique(mask), [0, 1]).all():
            raise RuntimeError(f"invalid merged ROI mask for {target}")

    rows = merge_provenance(
        source / "roi_provenance_shard0.csv.gz",
        source / "roi_provenance_shard1.csv.gz",
        output / "roi_provenance.csv.gz",
    )
    shutil.copy2(source / "roi_priors.json", output / "roi_priors.json")
    source_meta = [json.loads((source / f"roi_meta_shard{shard}.json").read_text()) for shard in (0, 1)]
    metadata = {
        "version": 1,
        "row_order": "concatenated_roi_shard0_then_roi_shard1_matches_arm16_global_row_order",
        "study_count": len(ids),
        "ids_file": "roi_ids.npy",
        "ids_sha256": sha256(output / "roi_ids.npy"),
        "provenance_file": "roi_provenance.csv.gz",
        "provenance_rows": rows,
        "source_shards": [
            {"file": f"roi_meta_shard{shard}.json", "metadata": source_meta[shard]}
            for shard in (0, 1)
        ],
        "targets": arrays,
    }
    (output / "roi_meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps({
        "passed": True,
        "study_count": len(ids),
        "provenance_rows": rows,
        "output_dir": str(output),
        "row_order": metadata["row_order"],
    }, indent=2))


if __name__ == "__main__":
    main()
