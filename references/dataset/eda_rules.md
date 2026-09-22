# Dataset EDA: reproducibility rules and evidence map

Source: `DATASET_REFERENCE.md` (evidence map, limits of this EDA) and
`references/run_dataset_eda.py` (deterministic sampling and statistics
implementation). Companion to `dataset_structure.md`, which holds the
corpus/label/geometry findings this methodology produced.

## Deterministic sampling rule

`run_dataset_eda.py` fixes `SEED = 20260919` for every random-looking
selection in the script. Deterministic subsets are drawn with:

```python
def deterministic_subset(items: list[T], limit: int, salt: str) -> list[T]:
    return sorted(
        items,
        key=lambda item: hashlib.sha256(f"{SEED}:{salt}:{item}".encode("utf-8")).digest(),
    )[:limit]
```

This is a hash-sort selection, not `random.sample`: it is reproducible across
runs/machines without seeding a stateful RNG, and different `salt` values
(e.g. `f"geometry-{split}"`, `f"pixels-{split}"`) draw independent,
non-correlated subsets from the same item list. Geometry sampling used
`--geometry-series-sample` (300 training series audited) and pixel decoding
used `--pixel-samples` (79 deterministic samples decoded). Re-running the
script with the same seed and sample sizes reproduces the identical subset.

## Physical slice ordering (mandatory, never by filename)

Zero of 315 audited series were spatially monotonic when sorted by filename
(filenames are SOP UIDs, not sequence numbers). The required order is:

```text
row_dir = IOP[:3]
col_dir = IOP[3:]
normal = cross(row_dir, col_dir)
slice_position = dot(IPP, normal)
sort by slice_position
```

For pixel index `[row, column]`, project to patient coordinates with:

```text
P = IPP
  + column × PixelSpacing[1] × row_dir
  + row    × PixelSpacing[0] × col_dir
```

This is the same projection implemented in `roi_geometry.SliceGeometry.
pixel_to_patient` (see `../roi_cache/file_contracts.md`) — the EDA script and
the production ROI pipeline must stay derived from one formula.

## Wilson binomial interval logic

Label prevalence (58 fully labeled studies) is reported with a 95% Wilson
score interval rather than a naive normal approximation, since normal
intervals misbehave at small `n` and prevalence near 0 or 1:

```python
def wilson_interval(positives: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    ...
```

`z = 1.959963984540054` is the exact two-sided 95% normal quantile. Every
`target_prevalence.csv` row's `wilson_95_low`/`wilson_95_high` columns come
from this function; see `dataset_structure.md`'s label table for the
resulting intervals. Treat these as uncertainty bounds, not tight estimates —
with `n=58`, a handful of studies can shift a coefficient substantially.

## Integrity and schema checks the script performs

- `require_layout(root)`: confirms `train.csv`, `train_series.csv`,
  `train_series/`, `test.csv`, `test_series.csv`, `test_series/`, and
  `sample_submission.csv` exist before any analysis begins.
- `sha256_file`: fingerprints CSV inputs for reproducibility tracking.
- `dataframe_schema`: records column names/dtypes for the schema audit.
- `analyze_csvs`: reconciles study/series IDs between `train.csv`,
  `train_series.csv`, and `sample_submission.csv`; computes the Wilson
  interval per target from the 58 fully labeled rows.
- `representative_header_record` / `geometry_record` / `decode_pixel_sample`:
  the per-series header read, per-series geometry audit, and per-sample pixel
  decode that back the corpus-integrity and geometry claims in
  `dataset_structure.md`.
- `normalized_manufacturer`: collapses raw vendor string aliases into the
  five normalized families reported in `dataset_structure.md`.
- `reconcile_filesystem_csv`: confirms CSV-declared study/series IDs match
  the on-disk directory tree; writes `filesystem_csv_mismatches.csv`
  (header-only when no mismatches are found).

## Evidence map (generated outputs)

The most useful generated files, all under `eda_outputs/`:

- `summary.json`: headline machine-readable facts and audit pass counts.
- `dicom_series_with_declared_metadata.csv.gz`: one row per series combining
  filesystem, representative-header, and competition metadata.
- `dicom_geometry_audit.csv`: slice-level geometry audit results.
- `dicom_pixel_decode_sample.csv`: deterministic decode validation.
- `slot_coverage.csv` and `series_depth_by_slot.csv`: model-input coverage.
- `target_prevalence.csv`, `target_phi_correlations.csv`, and
  `target_co_positive_counts.csv`: gold-label evidence.
- `report_statistics.csv.gz`: lengths and privacy-preserving normalized
  hashes.
- `study_laterality_availability.csv`: laterality usability by study.
- `filesystem_csv_mismatches.csv`: currently header-only because no
  mismatches were found.

The generated plots (`make_plots`) are summaries, not additional evidence.
The CSV/JSON files are authoritative when exact values matter.

## Limits of this EDA

- Pixel content was decoded for 79 deterministic samples, not visually
  reviewed across the entire corpus.
- Full slice geometry was audited for 315 series rather than all 24,386;
  every representative series header was still inspected.
- Target statistics have only 58 labeled observations.
- Language fingerprints are overlapping regex probes, not validated language
  identification.
- DICOM metadata can be anonymized, missing, or semantically noisy even when
  syntactically valid.
- The local test set is a three-study code fixture and may not describe the
  scoring test set.

## When to re-run

Re-run `run_dataset_eda.py` after any data replacement, extraction change, or
competition refresh. Compare `summary.json`, the mismatch table, and
laterality coverage before reusing existing caches or assumptions.
