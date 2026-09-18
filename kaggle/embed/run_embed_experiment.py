"""Small experiment: do frozen, unsupervised DINOv2 embeddings of knee MRI
slices group studies in a way that lines up with the findings we care about,
and can that grouping help clean up our report-mined pseudo-labels?

Not training our own DINOv3 -- that's a much bigger project (self-supervised
pretraining needs a large batch, many epochs, and real infra time). Instead:
use an off-the-shelf frozen DINOv2 ViT as a generic visual feature extractor
(never seen an MRI, but self-supervised ViT features transfer surprisingly
well to out-of-domain images for structure/texture), and see what falls out.

Pipeline:
  1. Sample studies: all 58 gold + a random slice of the pool.
  2. Per study: one representative (mid-stack) slice per DICOM series ->
     frozen DINOv2 embedding per slice.
  3. Converge to ONE vector per study: mean-pool across that study's series
     embeddings. Mean pooling (not max, not attention) is the deliberate
     choice here -- there's no label to train an attention pool against,
     and mean pooling is the standard unsupervised aggregate for "what does
     this whole study look like on average."
  4. PCA -> t-SNE to 2D, colored by label, to see if it groups meaningfully.
  5. A concrete, measurable use for training: k-nearest-neighbor label
     agreement. For each study, do its embedding-neighbors' pseudo-labels
     predict its own label better than random? If yes, neighbor-averaging
     in this embedding space is a legitimate way to denoise/smooth the
     report-mined pool labels -- a real technique (label propagation), not
     just a nice picture.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]
N_POOL_SAMPLE = 350
SEED = 42
IMG_SIZE = 224
K_NEIGHBORS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def find_root(input_root: Path, marker: str, skip_dirs=("train_series", "test_series")) -> Path:
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if marker in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{marker!r} not found under {input_root}")


INPUT_ROOT = Path("/kaggle/input")
COMP_ROOT = find_root(INPUT_ROOT, "train_series.csv")
LABELS_ROOT = find_root(INPUT_ROOT, "gold_test.csv")
WORK = Path("/kaggle/working")
SERIES_ROOT = COMP_ROOT / "train_series"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def load_dinov2():
    import timm
    for name in ["vit_small_patch14_dinov2.lvd142m", "vit_small_patch14_reg4_dinov2.lvd142m"]:
        try:
            m = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
            print("loaded timm model:", name, "embed dim", m.num_features)
            return m, m.num_features
        except Exception as e:
            print(f"  timm {name} failed: {e}")
    print("falling back to torch.hub facebookresearch/dinov2")
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    return m, 384


backbone, EMBED_DIM = load_dinov2()
backbone.eval().to(device)
for p in backbone.parameters():
    p.requires_grad_(False)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


def load_mid_slice(series_dir: Path) -> np.ndarray:
    files = list(series_dir.glob("*.dcm"))
    if not files:
        return None
    tagged = []
    for f in files:
        try:
            n = pydicom.dcmread(str(f), stop_before_pixels=True).InstanceNumber
        except Exception:
            n = 0
        tagged.append((int(n), f))
    tagged.sort(key=lambda t: t[0])
    mid = tagged[len(tagged) // 2][1]
    dcm = pydicom.dcmread(str(mid))
    arr = dcm.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    arr -= arr.min()
    denom = arr.max()
    if denom > 0:
        arr /= denom
    import cv2
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return arr


@torch.no_grad()
def embed_batch(slices: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(slices.astype(np.float32)).unsqueeze(1).repeat(1, 3, 1, 1).to(device)  # (B,3,H,W)
    x = (x - _MEAN) / _STD
    feats = backbone(x)
    return feats.cpu().numpy()


def main():
    gold = pd.read_csv(LABELS_ROOT / "gold_test.csv")
    pool = pd.read_csv(LABELS_ROOT / "train_pool.csv")
    pool_sample = pool.sample(n=min(N_POOL_SAMPLE, len(pool)), random_state=SEED).reset_index(drop=True)

    gold = gold.copy(); gold["split"] = "gold"
    pool_sample = pool_sample.copy(); pool_sample["split"] = "pool"
    studies = pd.concat([gold, pool_sample[["StudyInstanceUID", *LABELS, "split"]]], ignore_index=True)
    print(f"embedding {len(studies)} studies ({len(gold)} gold + {len(pool_sample)} pool)")

    series_df = pd.read_csv(COMP_ROOT / "train_series.csv")
    series_by_study = {uid: g for uid, g in series_df.groupby("StudyInstanceUID")}

    study_uids, study_embeds, n_series_used = [], [], []
    t_report = 0
    for i, row in studies.iterrows():
        uid = row["StudyInstanceUID"]
        sdf = series_by_study.get(uid)
        if sdf is None or len(sdf) == 0:
            continue
        slice_imgs = []
        for _, srow in sdf.iterrows():
            arr = load_mid_slice(SERIES_ROOT / uid / srow["SeriesInstanceUID"])
            if arr is not None:
                slice_imgs.append(arr)
        if not slice_imgs:
            continue
        embs = embed_batch(np.stack(slice_imgs))
        study_embeds.append(embs.mean(axis=0))   # converge to ONE vector: mean-pool over series
        study_uids.append(uid)
        n_series_used.append(len(slice_imgs))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(studies)} studies embedded", flush=True)

    emb_matrix = np.stack(study_embeds)
    meta = studies[studies.StudyInstanceUID.isin(study_uids)].set_index("StudyInstanceUID").loc[study_uids].reset_index()
    meta["n_series"] = n_series_used
    print(f"final embedding matrix: {emb_matrix.shape}")

    np.save(WORK / "study_embeddings.npy", emb_matrix)
    meta.to_csv(WORK / "study_embeddings_meta.csv", index=False)

    # ---- 2D projection ----
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca = PCA(n_components=min(50, emb_matrix.shape[0] - 1, emb_matrix.shape[1]))
    reduced = pca.fit_transform(emb_matrix)
    print("PCA explained variance (first 10):", np.round(pca.explained_variance_ratio_[:10], 3))

    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca")
    xy = tsne.fit_transform(reduced)
    meta["tsne_x"], meta["tsne_y"] = xy[:, 0], xy[:, 1]
    meta.to_csv(WORK / "study_embeddings_meta.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    for split, color in [("pool", "#9db2ae"), ("gold", "#1f6f68")]:
        m = meta.split == split
        ax.scatter(meta.tsne_x[m], meta.tsne_y[m], s=14, c=color, label=split, alpha=0.75)
    ax.set_title("colored by split (pool vs gold)")
    ax.legend()

    ax = axes[1]
    sc = ax.scatter(meta.tsne_x, meta.tsne_y, s=14, c=meta["Effusion"], cmap="viridis")
    ax.set_title("colored by Effusion label (0-1)")
    plt.colorbar(sc, ax=ax)

    ax = axes[2]
    sc = ax.scatter(meta.tsne_x, meta.tsne_y, s=14, c=meta["n_series"], cmap="plasma")
    ax.set_title("colored by # series (protocol variation)")
    plt.colorbar(sc, ax=ax)

    fig.suptitle("Frozen DINOv2 embeddings of knee MRI studies, mean-pooled per study, t-SNE'd")
    fig.tight_layout()
    fig.savefig(WORK / "embedding_scatter.png", dpi=130, bbox_inches="tight")
    print("saved embedding_scatter.png")

    # ---- kNN label-agreement: is this embedding space useful for labeling? ----
    from sklearn.neighbors import NearestNeighbors

    Y = meta[LABELS].to_numpy()
    nn_model = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(emb_matrix)
    _, idx = nn_model.kneighbors(emb_matrix)
    idx = idx[:, 1:]  # drop self

    neighbor_pred = Y[idx].mean(axis=1)  # average label of k nearest neighbors
    rng = np.random.default_rng(SEED)
    random_idx = rng.integers(0, len(Y), size=idx.shape)
    random_pred = Y[random_idx].mean(axis=1)

    from sklearn.metrics import r2_score
    print("\nper-finding R^2 of (own label) ~ (mean label of k-NN embedding neighbors), vs random baseline:")
    results = {}
    for j, name in enumerate(LABELS):
        r2_knn = r2_score(Y[:, j], neighbor_pred[:, j])
        r2_rand = r2_score(Y[:, j], random_pred[:, j])
        results[name] = {"r2_knn_neighbors": round(float(r2_knn), 4),
                          "r2_random_pairs": round(float(r2_rand), 4)}
        print(f"  {name:20s} knn R^2={r2_knn:+.4f}   random R^2={r2_rand:+.4f}")

    with open(WORK / "knn_label_agreement.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\ndone")


if __name__ == "__main__":
    main()
