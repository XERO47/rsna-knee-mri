# RSNA Knee Abnormality Detection

Multi-label detection of 12 knee findings from DICOM MRI, built for the Kaggle
competition
[rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
(Research category, $77,000 prize pool, 6,425 teams).

**Final result:** 0.937 public LB, 0.920 private LB (macro ROC AUC).

## The problem

This is not a plain image classifier. `train.csv` has 4,407 studies, each with
a free-text radiology `Report` (multilingual), but structured 0/1 labels for
only **58 of them (1.3 percent)**. `test.csv` has no `Report` column at all, so
at inference the model only ever sees images.

That creates two chained sub-problems:

1. **Label mining (text to labels, training-time only).** The `Report` field is
   the only label source for 98.7 percent of training studies. It can be used
   to derive training targets, never as a model input, because it does not
   exist at test time.
2. **Multi-instance, multi-view vision (images to 12 probabilities).** Each
   study has multiple DICOM series (sagittal, coronal, axial planes and
   different sequences), each a stack of slices. The model pools evidence
   across a variable number of volumes per patient into one study-level
   multi-label prediction.

The 12 target columns: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`,
`Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`,
`Contusion`, `Fracture`.

## Evaluation convention

The 58 expert-labeled studies are the only ground truth, so they are held out
as an internal gold test set and never trained on. This matches what the host
confirmed: image labels were assigned independently of the reports, the image
is authoritative, and report-derived labels systematically over-call relative
to the stricter image labels. That over-call puts a soft ceiling near 0.90 on
any purely report-derived label set, which is the central difficulty of the
competition.

## Approach

### 2.5D representation

Volumes are resampled into a fixed cache of shape `(4407, 64, 336, 336)` with
five plane and sequence slots. The model consumes 3-slice windows (a center
slice plus its two neighbors) as 3-channel inputs. This 2.5D design keeps most
of the signal of a 3D volume while staying cheap enough to train many models.
A full-3D baseline (`kaggle/vol3d`) was tried and scored lower (0.871), so 2.5D
was kept for every production model.

### KneeNet: per-finding attention MIL

Each study is a bag of windows across its series. `KneeNet` applies a shared
backbone to every window, then a per-finding attention head pools the windows
into one vector per finding, and a per-finding linear head produces the 12
logits. Training uses masked binary cross-entropy (the "report never addressed
this finding" prior is masked out of the loss), stochastic weight averaging
over the final epochs, distributed data parallel, and per-finding positive
weighting.

### Label strategy (the main lever)

Label quality, not architecture, moved the score. Three generations:

1. **Report-mined labels.** Multilingual report text parsed into 0/1 targets.
   Caps near 0.89 gold because reports over-call.
2. **Evidence-fusion (evf) labels.** Cleaner public soft labels (0.929 gold).
   A single model trained on these reached 0.9006 gold and 0.929 public LB on
   its own.
3. **Pseudo-labels.** Out-of-fold image predictions blended into the evf labels
   at a fixed ratio, then retrained. This gave a consistent per-fold gain and
   the best ensemble.

### Ensemble

The final submission blends, by per-finding rank averaging:

- Four base models: two ConvNeXt-small and two CoAtNet-RMLP-2 at 384, trained
  on report and cache64 corpora.
- Five cross-validation folds trained on pseudo or evidence-fusion labels.
- A crop specialist that votes only on five findings, using anatomical ROI
  crops rebuilt from test DICOM by the same builder that made its training
  cache.
- A retrieval member with no classifier head: for each finding it takes a
  softmax-weighted vote over the nearest training studies in that finding's own
  attention-pooled embedding space.

Two extra decorrelated backbones (an eva02 ViT and a caformer MetaFormer) were
trained on pseudo labels and both gated positive, but the weekly GPU quota ran
out before their blend could be scored.

## Results

| Milestone                                | Public LB |
|------------------------------------------|-----------|
| Single evidence-fusion model             | 0.929     |
| Six-member blend                         | 0.932     |
| Eight-member blend                       | 0.934     |
| Nine-member (evf folds)                  | 0.936     |
| Eleven-member (pseudo folds)             | 0.937     |

Final private LB: 0.920. The public to private drop was uniform (about 0.017)
across every submission, which points to a private-distribution shift that hit
all models equally rather than a selection error.

### What did not work

- **Training only on the 58 gold studies.** Even with warm-starting from the
  best model, fine-tuning on 58 clean studies overfit badly (training AUC 0.98,
  held-out gold 0.88, a drop of about 0.04 below the base). Clean labels cannot
  overcome tiny data volume.
- **Full-3D models.** Lower than 2.5D (0.871).
- **Architecture and input-resolution tweaks at 336px.** The blend saturated
  near 0.937; only a decorrelated backbone family moved it, and only slightly.

### What would come next

The honest ceiling of this pipeline is capacity and localization, not labels.
The path to the 0.95 tier is a two-stage design: a learned coordinate
localization stage per finding, tight higher-resolution crops around the
predicted landmarks, and a multi-instance severity classifier on those crops.
That is a multi-week rebuild, not a single-run change.

## Repository layout

```
kaggle/            training and inference kernels (one directory per experiment arm)
  arm*/            model variants (backbone, label set, recipe) with kernel-metadata.json
  sub_*/           ensemble inference kernels that build submission.csv
  roi_cache/       ROI crop geometry, cache builder, and the web annotation tool
  armc_roi/        ROI residual-head model training
  vol3d/           full-3D baseline (kept for reference)
references/        index-driven documentation tree (start at references/INDEX.md)
scripts/           dataset preparation and EDA
src/               shared library code
data/              local data root (contents gitignored; fetched via scripts)
models/            local checkpoints (gitignored)
```

Agent and contributor guidance lives in `AGENTS.md` and `CLAUDE.md`. The
reference tree under `references/` holds function inventories, data contracts,
and incident history; start at `references/INDEX.md` and follow it to the
relevant domain.

## Reproduction notes

- Data is downloaded separately (see `scripts/`), never committed.
- Checkpoints, cache shards, and prediction dumps are gitignored.
- Each `kaggle/<arm>/` directory is a self-contained Kaggle kernel with its own
  `kernel-metadata.json`; training runs on Kaggle GPU or RunPod.
- Submission is notebook-only (code competition): the inference kernel runs on
  the hidden test inside Kaggle and is submitted through the web interface.

## License

Code is provided as-is for research and educational reference.
