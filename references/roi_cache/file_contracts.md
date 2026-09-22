# ROI cache: file contracts and function/class inventories

Source: `ROI_CACHE_REFERENCE.md` (file contracts, end-to-end data flow,
function/class inventories, output schemas, test inventory). Companion to
`laterality_rules.md`, which holds the target/laterality contract and the
chronological correction history.

## End-to-end data flow

```text
train_series.csv + DICOM headers
        |
        +--> select-pilot --> 20-study manifest + 80-row annotation template
        |                         |
        |                         +--> ROIAnnotator --> patient-space landmarks
        |                                              |
        |                                              +--> LOO pilot priors/gate
        |
        +--> select-cohort --> frozen 100-study manifest + 400-row template
                                      |
                                      +--> annotations --> 80-fit/20-audit priors
                                                                |
ids_shard<S>.npy + DICOMs + accepted priors --------------------+
        |
        +--> geometry --> origin --> candidate centers --> physical crops
        +--> uint8 arrays + masks + provenance + metadata
        +--> validate --> Arm C loader (which checks IDs again)
```

## File contracts

### `roi_geometry.py`

Pure, split-agnostic geometry. Input is DICOM header metadata or NumPy
vectors; output is validated geometry, coordinates, crop bounds, triplets, or
series choices. It performs no pixel decoding, pathology-label access, or
writing.

### `build_roi_cache.py`

CLI orchestrator. Inputs are the competition root, annotations/manifests,
accepted priors, and authoritative IDs. Outputs are selection bundles, JSON
priors, five image arrays, five masks, provenance, metadata, validation JSON,
or a smoke PNG.

### `annotate_roi_pilot.py`

Local Matplotlib desktop UI. Inputs are root, selection manifest, annotation
CSV, and annotator name. Each save atomically records source series/SOP,
pixel position, patient XYZ, side, annotator, and QC. Restart resumes the
first incomplete task; no notebook runtime is used.

### `run_local_pilot.py`

Single local entry point. It creates or resumes the selection, opens the
local annotator when needed, verifies all 80 rows, fits LOO priors, renders a
real smoke image, and writes the complete pilot as one shard. An interrupted
run is resumed by executing the same command again.

## Function and class inventory: `roi_geometry.py`

| Symbol | Input | Output | Purpose |
|---|---|---|---|
| `_unit` | direction vector | normalized float64 vector | Reject zero/non-finite directions. |
| `_finite_vector` | object, length, name | validated float64 vector | Shared coordinate guard. |
| `SliceGeometry` | path/UID/IOP/IPP/spacing/matrix/position | immutable record | Binds projection to one physical slice. |
| `SliceGeometry.from_header` | DICOM path/header | `SliceGeometry` | Validates and derives row/column/normal axes. |
| `SliceGeometry.pixel_to_patient` | row, column | LPS XYZ mm | DICOM affine used for annotation/crops. |
| `SliceGeometry.patient_to_pixel` | LPS XYZ | row, column | True in-plane inverse (solved via `numpy.linalg.lstsq` for both in-plane basis vectors together; see `../web_annotator/backend.md`'s "Non-obvious implementation facts" for the 2026-09-21 correction from an earlier per-vector dot-product shortcut). |
| `SliceGeometry.center` | none | physical image center | Study-origin candidate. |
| `SeriesGeometry` | series metadata and slices | immutable record | One validated acquisition. |
| `SeriesGeometry.middle` | none | middle slice | Stable physical reference. |
| `SeriesGeometry.center` | none | physical center | Study-origin input. |
| `SeriesGeometry.nearest_slice_index` | patient XYZ | slice index | Nearest physical plane. |
| `orientation_angle_degrees` | two vectors | degrees | 1-degree consistency gate. |
| `sort_deduplicate_slices` | slices/tolerances | ordered unique tuple | One-normal projection, deterministic ties, 0.1 mm deduplication. |
| `normalize_laterality` | arbitrary value | `L`, `R`, or `None` | Accepts only explicit side tokens. |
| `resolve_laterality` | votes | side and source/status | Preserves absent/conflicting metadata. |
| `canonical_axes` | known side | medial/anterior/superior axes | Side-normalizes priors. |
| `study_origin` | valid series | median XYZ | Robust acquisition-wide origin. |
| `patient_to_canonical` | point, origin, side | canonical offset | Fits priors. |
| `canonical_to_patient` | offset, origin, side | patient XYZ | Projects known/hypothetical side. |
| `centerline_offset_to_patient` | offset, origin | patient XYZ | Consistent PF/suprapatellar center without side. |
| `anatomical_crop_axes` | plane, optional side | width/height axes | Maps anatomical extent into acquisition. |
| `crop_bounds` | slice, center, plane, mm size, side | pixel bounds | Physical rather than fixed-pixel crop. |
| `padding_fraction` | bounds, image size | outside fraction | Enforces 20% limit. |
| `crop_with_padding` | image, bounds | padded crop | Preserves requested extent at edges. |
| `triplet_indices` | center, depth | two overlapping triplets | PF/synovitis slice strategy. |
| `centered_triplet_indices` | center, depth | centered triplet | One triplet per side candidate. |
| `nearest_distinct_triplets` | center, depth | two nearest distinct triplets | Synovitis 25 mm policy: enumerates every valid consecutive three-slice group and picks the two nearest the landmark; see `laterality_rules.md`'s synovitis audit. |
| `pick_series` | rows, plane, fluid, fallback | row or `None` | Deterministic first match, never changes plane. |
| `assert_exact_id_alignment` | global/ROI IDs | exception or none | Prevents silent row mismatch. |

`LANDMARKS` fixes four annotation names. `TARGET_SPECS` is the single source
for target name, landmark, crop mm, side dependence, fallback, and two slots
(see `laterality_rules.md` for the full table).

## Function inventory: `build_roi_cache.py`

### Discovery, decoding, and geometry

| Function | Input | Output / side effect | Purpose |
|---|---|---|---|
| `log` | message | timestamped stdout | Long-build progress. |
| `sha256_file` | path | SHA-256 | Input fingerprints. |
| `find_competition_root` | optional path | validated root | Local path or safe Kaggle discovery. |
| `find_named_file` | name, roots | deterministic path | Finds cache64 IDs; logs ambiguity. |
| `_safe_float` | value | finite float/`None` | Optional header values. |
| `read_series_geometry` | study, series row, root | `SeriesGeometry` | Reads headers and physically orders slices. |
| `_read_pixels` | DICOM path | float32 image | Modality LUT/inversion/finiteness. |
| `intensity_limits` | series | low/high | 16 physical samples, 2nd/98th percentiles. |
| `_scale_uint8` | image, limits | uint8 image | Compact deterministic scaling. |
| `load_study_geometry` | study, all rows, root | geometries/failures | Reads all series for stable origin; contains failures. |

### Build and multiprocessing

| Function | Input | Output / side effect | Purpose |
|---|---|---|---|
| `_blank_provenance` | study/target/window/status | invalid row | Schema-complete audit trail. |
| `_landmark_point` | landmark/prior/origin/side/mode | patient XYZ | Side or centerline projection. |
| `_window_candidates` | spec/priors/origin/side | candidate labels/points | Paired L/R or one center. |
| `build_study` | ID, series rows, root, priors | five arrays, masks, 20 rows | Deterministic failure-contained unit. |
| `_worker_initialize` | grouped rows/root/priors | worker context | Avoids repeated serialization. |
| `_worker_job` | index, study | indexed result | Converts study exceptions to masked output. |
| `command_build` | build args | shard artifacts | Memmaps, provenance, fingerprints, deterministic rebuild. |

### Annotation selection

| Function | Input | Output / side effect | Purpose |
|---|---|---|---|
| `_header_summary` | representative DICOM | metadata dict | Cheap pixel-free stratification. |
| `_normalize_manufacturer` | raw vendor | normalized family | Collapses aliases. |
| `_range_bin` | value/edges | interval label | Stable coarse protocol strata. |
| `_annotation_template` | study records | four-row/study frame | Exact annotation schema. |
| `_collect_annotation_candidates` | root/series frame | eligible studies | Label-free metadata aggregation; trusted side and reference views required. |
| `_selection_tiebreak` | UID/salt | seeded hash | Stable, genuinely seed-dependent ties. |
| `_extend_representative_selection` | pool/current/count/side quotas/population/salt | mutates selection | Population matching without rare-stratum over-sampling. |
| `_write_annotation_selection` | records/output names | manifest/NPY/CSV | Consistent selection bundle. |
| `command_select_pilot` | root/output | 20-study bundle | 16/4 coverage, 10/10 side, rare-vendor stress. |
| `command_select_cohort` | pilot/root/output | 100-study bundle | Retains pilot and freezes 80/20 before annotation. |

### Priors, validation, and CLI

| Function | Input | Output / side effect | Purpose |
|---|---|---|---|
| `command_build_pilot` | manifest/priors/output | 20-ID production cache | Tests exact builder before expansion. |
| `_coverage_for_offset` | actual/predicted/side/target | Boolean | Same extent logic as production. |
| `_annotation_offsets` | annotations/metadata/root | offsets/sides | Recomputes origins; rejects incomplete data. |
| `_prior_statistics` | fit IDs/offsets | median/MAD/p95/n | Robust prior and dispersion. |
| `_audit_coverage` | audit/fit/offsets/sides/LOO | coverage/mirror report | 90% pilot and 95% final gates. |
| `command_fit_priors` | phase/annotations/manifest/root/output | priors JSON | LOO pilot; frozen final split, no reshuffle. |
| `command_validate` | cache/IDs | JSON or exception | IDs, tensors, masks, provenance, strategies, ordering, coverage, padding. |
| `command_smoke` | DICOM/priors/root/PNG | visual plus double-build | Human/determinism gate. |
| `parse_arguments` | argv | namespace | Defines all CLI workflows. |

Fixed constants are `SEED=20260919`, `IMAGE_SIZE=336`, `WINDOWS_PER_TARGET=4`,
and `PADDING_LIMIT=0.20`. `PROVENANCE_FIELDS` is the authoritative
compressed-CSV schema.

## Class and function inventory: `annotate_roi_pilot.py`

| Symbol | Input | Output / behavior | Purpose |
|---|---|---|---|
| `REFERENCE_SLOT` | landmark | ordered reference choices | Coronal joint line, axial PF, sagittal suprapatellar; single source of truth also re-exported by `services.py` in the web annotator (see `../web_annotator/backend.md`). |
| `_display_pixels` | DICOM path | normalized image | Notebook display with LUT/inversion. |
| `ROIAnnotator.__init__` | root/manifest/CSV/name | widget/task state | Loads data, resumes incomplete task, wires callbacks. |
| `_task_rows` | study | series rows | Safe lookup. |
| `_reference_row` | rows/landmark | series row | Explicit reference priority. |
| `_load_task` | current index | UI/series state | Contains per-task failures. |
| `_render` | current slice | image/cleared marker | Keeps click bound to displayed SOP. |
| `_on_slice_change` | widget event | rerender | Prevents stale click save. |
| `_set_status` | text and error flag | updated status line | One consistent visible feedback path. |
| `_on_click` | Matplotlib mouse event | stored pixel/XYZ and marker | Immediate round-trip verification. |
| `_save_and_next` | button event | atomic CSV and advance | Exactly one resumable record/task. |
| `_move` | signed delta | clamped navigation | Safe previous/next. |
| `_on_key` | Matplotlib key event | slice/task navigation or save | Efficient local keyboard workflow. |
| `show` | none | blocking desktop window | Local entry point. |
| `parse_arguments` | CLI argv | namespace | Defines local annotator inputs. |
| `main` | none | constructs and shows annotator | Script entry point. |

## Function inventory: `run_local_pilot.py`

| Function | Input | Output / behavior | Purpose |
|---|---|---|---|
| `annotations_complete` | annotation CSV | Boolean and progress message | Requires exactly 80 approved finite landmark rows. |
| `first_dicom_for_manifest` | root and pilot manifest | example DICOM path | Deterministic real-pixel smoke input. |
| `parse_arguments` | CLI argv | namespace | Root, work directory, annotator, workers, and headless check. |
| `main` | none | selection/UI/priors/smoke/single shard | Resumable all-local pilot orchestration. |

## Output schemas

```text
roi_<target>_shard<S>.npy        uint8 [N,4,3,336,336]
roi_<target>_masks_shard<S>.npy  uint8 [N,4]
roi_ids_shard<S>.npy             exact authoritative IDs
roi_provenance_shard<S>.csv.gz   20 rows per study
roi_meta_shard<S>.json           fingerprints, contract, coverage, timing
roi_priors.json                  accepted priors and frozen IDs
```

Valid provenance has three distinct SOP UIDs with increasing positions.
`candidate_laterality` is `L`/`R` for paired targets and `CENTER` otherwise.
`window_strategy` is `paired_side_centered` or `center_overlapping`. Invalid
rows are zero-filled, mask 0, with a non-empty `failure_reason`.

## Tests

`test_roi_geometry.py` covers projection, orientation, physical ordering,
deduplication, crop/padding math, triplets, series selection, and ID
alignment. `test_roi_builder.py` covers failure containment, tensor
contracts, paired-side candidates, and side-independent centers.
`test_roi_revision.py` covers direction-token handling, explicit conflict
rejection, and QC-list composition (see `laterality_rules.md` for the
final-r1 revision it protects).

Test helper/class inventory:

- `make_slice` constructs synthetic `SliceGeometry` from a minimal header.
- `GeometryTests.test_pixel_patient_round_trip` checks the affine inverse.
- `test_left_and_right_medial_axes` checks LPS canonical axes and vote
  handling.
- `test_reversed_filenames_sort_by_physical_position` prevents filename
  sorting.
- `test_position_deduplication_uses_point_one_mm` fixes the dedup tolerance.
- `test_reversed_orientation_is_rejected` checks the orientation gate.
- `test_oblique_projection_round_trip` covers non-cardinal IOP.
- `test_crop_extent_and_padding` checks mm bounds and outside-area fraction.
- `test_missing_series_and_boundary_windows_are_invalid` checks masks/triplets.
- `test_series_selection_is_deterministic_and_exact` checks fallback policy.
- `test_exact_id_alignment_rejects_reordering` checks cache row safety.
- `BuilderFailureTests.test_laterality_dependent_targets_always_get_two_side_hypotheses`
  checks symmetric candidates without metadata.
- `test_side_independent_target_ignores_laterality` checks centerline
  semantics.
- `test_mirror_audit_compares_landmarks_not_origin_sign` prevents false
  mirror failures from an off-center origin.
- `test_missing_series_produces_zero_windows_and_complete_provenance` checks
  failure containment and all output shapes/fields.
- `test_near_orthogonal_header_uses_a_true_in_plane_inverse` protects the
  2026-09-21 `patient_to_pixel` geometry fix (see `laterality_rules.md`).

## Validation completed in this review

- 14 unit tests passed with real NumPy/OpenCV/pandas dependencies.
- `py_compile` passed for every ROI Python file.
- Selectors completed against all local study metadata; exact cohort counts
  and the frozen 80/20 split were verified.
- One known- and one unknown-laterality study were each built twice from real
  pixels. All five targets had expected strategies, 20 provenance rows, and
  byte-identical arrays/masks.
- The unknown-side study produced valid paired windows for all three
  side-dependent targets, removing the EDA blocker.
