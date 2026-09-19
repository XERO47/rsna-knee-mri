# Arm 16 / Arm C — global + local residual MIL

This is the new model family finalized in the architecture review. It retains
Arm 12/15's ConvNeXt global bag and twelve finding-specific attention queries,
then adds local evidence only for:

- MCL
- Lateral Meniscus
- Lateral OA
- PF OA
- Synovitis

Each local bag has its own gated-attention pooler. Its scalar output is added
only to the matching global logit through a gate initialized to
`sigmoid(-2) ≈ 0.119`. The backbone is shared across global and local windows.

## Tensor API

```python
from model import make_model

model = make_model(pretrained=True, encoder_chunk_size=32)
logits = model(
    global_windows,  # [B, 12, 3, 336, 336]
    global_mask,     # [B, 12]
    roi_windows={"MCL": mcl_windows, "PF OA": pf_windows},
    roi_masks={"MCL": mcl_mask, "PF OA": pf_mask},
)
```

At inference, pass all available windows rather than training's random sample.
The implementation encodes only mask-valid windows and can chunk backbone
forwards to control memory.

## Warm-start the proven global arm

```python
from model import load_global_checkpoint, make_model

model = make_model(pretrained=False)
summary = load_global_checkpoint(model, "/path/to/arm12_swa.pt")
```

The loader maps Arm 12/15's `backbone`, `norm`, `att`, `clsW`, and `clsb`
weights into Arm C. ROI heads and their beta gates intentionally remain new.

## ROI data contract

ROI crop generation is intentionally outside this module. Do not guess crop
centres from normalized image coordinates. Build them from the agreed physical
coordinate priors and serialize provenance per study. A dataset may omit a
whole ROI key, or mark unavailable sample windows false in the mask; the model
then makes the ROI path an exact no-op.

Planned bags, all resized to 336 px:

| Target | Preferred input | Physical crop |
|---|---|---|
| MCL | coronal + axial fluid-sensitive | 90 × 120 mm medial |
| Lateral Meniscus | sagittal + coronal PD/intermediate | 80 × 80 mm lateral joint line |
| Lateral OA | coronal PD/T1 + sagittal PD | 85 × 75 mm lateral compartment |
| PF OA | axial + sagittal PD/FS | 95 × 85 mm patellofemoral |
| Synovitis | axial/sagittal fluid-sensitive | 100 × 100 mm suprapatellar/peripatellar |

Before training, enforce the architecture review's safeguards: ≥95% held-out
crop coverage, serialized crop provenance, fold-isolated image-teacher labels,
and no report access from the inference dataset.

## Local verification

```bash
cd kaggle/arm16
python -m unittest -v test_model.py
```

## One-cell Kaggle training

[`train_arm16_kaggle.py`](train_arm16_kaggle.py) is self-contained and does not
import this package. Paste it into one Kaggle cell, or upload the file and run:

```python
exec(open('/kaggle/working/train_arm16_kaggle.py').read())
```

Set `WANDB_API_KEY` as a Kaggle secret/environment variable and optionally set
`WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_RUN_NAME`. The script logs training
loss, gold macro/per-target AUC, learning rates, epoch time, and ROI gate values.
Every second epoch and the final epoch are uploaded as W&B model artifacts. Each
`.pt` artifact contains model weights, optimizer state, scheduler state, AMP
scaler, history, RNG state, configuration, and metrics, so training can resume
with `ARM16_RESUME=/path/to/checkpoint.pt`.

The existing cache64 and label files are required. ROI files are optional for a
global smoke run; to activate each local branch, attach the corresponding
`roi_<target>_shard0/1.npy` and `roi_<target>_masks_shard0/1.npy` files described
in the standalone trainer's docstring.

Run [`check_roi_cache.py`](check_roi_cache.py) in Kaggle before training to
confirm whether those files are actually attached. ROI caches are not supplied
by the competition or created by Arm 12/15: they must be produced separately
from physically aligned DICOM local crops and their anatomy-coordinate priors.
