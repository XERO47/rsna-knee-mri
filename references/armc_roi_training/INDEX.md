# Arm C ROI training domain index

Files in this folder, split from the archived Arm16 training reference
`../archive/`).

| File | Read this for |
|---|---|
| `architecture.md` | Why Arm 13 (CoAtNet-RMLP-2 @384) is the preserved global baseline; the exact global-path and local-path (ROI residual) diagrams; the zero-init/zero-residual guarantee that keeps a freshly-loaded Arm 13 checkpoint bit-identical before training; the data contract (cache64 + ROI cache + label CSVs + Arm 13 checkpoint); and the `train_armc_roi_kaggle.py`/`model.py`/`check_roi_cache.py` symbol map. |
| `training_policy.md` | The 10-epoch schedule, 2 frozen-global-epoch warm-up, learning-rate table per parameter group, loss composition (primary BCE + 0.25-weighted auxiliary global BCE), no-flip policy, the pre-epoch-0 sanity check, per-epoch logging fields, top-3-vs-best-single model selection, the 5-point credibility checklist for calling a run a real Arm 13 upgrade, and the checkpoint/recovery contract. |
| `launch.md` | Exact Kaggle input basenames, merged-cache alignment requirements, labels/checkpoint contract, run order, ZIP extraction example, outputs/recovery, and the boundary between Arm C ROI and the separate `kaggle/arm16/train_arm16.py` experiment. |

Read `architecture.md` before changing model code or data loading;
`training_policy.md` before changing hyperparameters, loss, schedule, or
checkpoint/selection logic.
