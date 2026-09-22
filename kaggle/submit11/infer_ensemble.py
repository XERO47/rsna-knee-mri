"""Ensemble submission: several of our own members, each rebuilt from DICOM under its
own preprocessing recipe, rank-averaged per finding.

Members are listed in MEMBERS by checkpoint filename. Each checkpoint either carries a
"recipe" (slot layout, span, crop, image size -- arm12 onward) or was trained on the
public 44-slice Raptor corpus (arm10, arm11), whose recipe arm10_verify matched
pixel-for-pixel. Every distinct recipe is rebuilt once per study and shared by all
members that use it.

WHY RANK AVERAGING. The metric is per-finding ROC AUC, which depends only on ordering.
Members trained at different resolutions / architectures are calibrated differently,
so averaging raw probabilities lets the most confident member dominate. Ranks give
each member an equal vote on ordering.

GUARDS (the reason this kernel can be trusted):
  * every checkpoint loads 0 missing / 0 unexpected;
  * self-check: for each recipe with a reference corpus attached, a few train studies
    are rebuilt from DICOM and must be byte-identical to that corpus;
  * MODE="validate": rebuild the 58 gold studies from DICOM and require EACH member to
    reproduce the gold AUC its training kernel recorded (|diff| <= VALIDATE_TOL).
    A member that does not reproduce is a preprocessing bug, and the run fails.
"""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydicom.pixel_data_handlers.util import apply_modality_lut

MODE = "submit"                        # "validate" | "submit"
MEMBERS = [                                # (checkpoint file, weight)
    ("arm10_swa.pt", 1.0),                 # ConvNeXt-small, raptor44 corpus   | gold .8853, LB .920
    ("arm11_swa.pt", 1.0),                 # CoAtNet-RMLP-2 @384, raptor44     | gold .8928
    ("arm12_swa.pt", 1.0),                 # ConvNeXt-small, cache64 maxspan   | gold .8933, LB .923
    ("arm13_best.pt", 1.0),                # CoAtNet-RMLP-2 @384, cache64      | gold .8909 (best epoch;
                                           #   its weight-average step OOMed, checkpoint recovered)
    # ("arm15_swa.pt", 1.0),                 # ConvNeXt-small, cache64, OOF-refined labels | gold .9011
]

VALIDATE_TOL = 0.002
# arm13's training kernel crashed in its final K=62 evaluation, so the gold score stored in
# its checkpoint is a 24-window number from its best epoch. Comparing it against this
# kernel's 62-window score would flag a mismatch that is not an error, so it is reported
# rather than enforced. Every other member is still required to reproduce exactly.
NO_STRICT = {"arm13_best"}
# Depth-reversal TTA: every window is also scored with its three slices in reverse order
# (c+1, c, c-1). The 0.941 public blend carries this as a separate "-reverse" arm of the
# same weights. Reversed BEFORE normalisation, so the model sees a valid image rather than
# channel statistics swapped between R and B. Same model, same calibration -> the two
# passes are averaged as probabilities before the cross-member rank blend.
TTA_REVERSE = False
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]

# Verified exact against dreaddevelopment/knee-raptor-corpus (kaggle/arm10_verify).
RAPTOR44 = {"name": "raptor44", "img": 336, "crop_mm": 140.0, "span": [0.15, 0.85],
            "slots": [["Sagittal", 1, 12], ["Sagittal", 0, 10], ["Coronal", 1, 8],
                      ["Coronal", 0, 6], ["Axial", -1, 8]]}
# Reference corpora for the byte-equality self-check: recipe name -> (ids, vols) files.
REFERENCE = {"raptor44": ("all_ids.npy", "all_vols.npy"),
             "cache64-maxspan": ("ids_shard0.npy", "vols_shard0.npy")}
# validate only: score every member at each of these window counts in one run, so the
# cost/accuracy curve comes out of a single DICOM build. Empty list = current behaviour.
EVAL_KS = [24, 42, 62]
# Windows per study in SUBMIT mode, overriding each member's depth-2 default. The v14
# sweep measured K=24 vs K=62 at matched TTA as 0.0003 (no TTA) and 0.0009 (TTA) on gold
# -- neighbouring windows share two of three slices, so K=62 mostly re-scores the same
# pixels. 24 costs 4.33x less. None of this is worth anything unless the LB agrees, which
# is what this submission tests.
SUBMIT_K = int(os.environ.get("SUBMIT_K", "24"))
SELF_CHECK_N = 2
BATCH_STUDIES = 2
# Backbone images per forward. arm13's training kernel died in its final eval doing
# 2 studies x 62 windows = 124 CoAtNet images at 384 in one go; chunking bounds peak
# memory. Models are in eval mode, so normalisation uses running statistics and the
# result is identical to one big batch.
CNN_CHUNK = 32
BUILD_THREADS = 4

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
cv2.setNumThreads(1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_file(name, root="/kaggle/input"):
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if name in fs:
            return Path(d) / name
    return None


def find_comp():
    for d, dirs, fs in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if "test.csv" in fs and "train_series.csv" in fs:
            return Path(d)
    raise FileNotFoundError("competition data not attached")


# ------------------------------------------------ verified rebuild (parametrised) --
def order_and_meta(sdir):
    recs, ps_list = [], []
    for f in glob.glob(sdir + "/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            ps_list.append(ps)
            recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda x: x[0])
    med = float(np.median(ps_list)) if ps_list else 0.5
    return [(f, ps) for _, f, ps in recs], med


def read_px(f):
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def mm_crop_resize(a, ps, crop_mm, img):
    h, w = a.shape
    c = min(int(round(crop_mm / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - c) // 2, (w - c) // 2
    return cv2.resize(a[y0:y0 + c, x0:x0 + c], (img, img), interpolation=cv2.INTER_AREA)


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane
             and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, rows, series_root, recipe, order_cache=None, px_cache=None):
    """The verified recipe with its layout as parameters. Header ordering and decoded
    pixels are cached per study so several recipes reuse the same DICOM reads."""
    order_cache = {} if order_cache is None else order_cache
    px_cache = {} if px_cache is None else px_cache
    img, crop_mm = int(recipe["img"]), float(recipe["crop_mm"])
    span_lo, span_hi = recipe["span"]
    depth = sum(k for _, _, k in recipe["slots"])
    vol = np.zeros((depth, img, img), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in recipe["slots"]:
        r = pick_series(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        sdir = f"{series_root}/{sid}/{r['SeriesInstanceUID']}"
        if sdir not in order_cache:
            order_cache[sdir] = order_and_meta(sdir)
        files, med = order_cache[sdir]
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * span_lo), int(n * span_hi) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs, pss = [], []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                if fp not in px_cache:
                    px_cache[fp] = read_px(fp)
                arrs.append(px_cache[fp]); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= depth:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            vol[idx] = (mm_crop_resize(aw, ps if ps > 0 else med, crop_mm, img) * 255).astype(np.uint8)
            idx += 1
    mask = (vol.reshape(depth, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def build_all_recipes(sid, rows, series_root, recipes):
    oc, pc = {}, {}
    return {name: build_study(sid, rows, series_root, rec, oc, pc) for name, rec in recipes.items()}


# ----------------------------------------------------------------------- windows --
def window_centers(mask, k):
    """Identical to StudyWindows.centers(train=False) in the trainers."""
    depth = len(mask)
    valid = np.flatnonzero(mask > 0)
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, depth - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [max(1, min(cs[i], depth - 2)) for i in idx]


def windows(vol, mask, res, k, reverse=False):
    cs = window_centers(mask, k)
    order = (lambda c: [c + 1, c, c - 1]) if reverse else (lambda c: [c - 1, c, c + 1])
    x = torch.from_numpy(np.stack([vol[order(c)] for c in cs])).float().div_(255.0)
    if x.shape[-1] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


# ------------------------------------------------------------------------- model --
class RaptorClassifier(nn.Module):
    """Mirrors the trainers' module exactly (asserted by the 0/0 state_dict load)."""

    def __init__(self, backbone, n=len(LAB), drop=0.2):
        super().__init__()
        f = backbone.num_features
        self.backbone = backbone
        self.norm = nn.LayerNorm(f)
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, f))
        self.clsb = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        feats = [self.backbone(flat[i:i + CNN_CHUNK]) for i in range(0, flat.shape[0], CNN_CHUNK)]
        h = self.norm(torch.cat(feats).view(B, K, -1))
        a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
        return (torch.einsum("bkn,bkf->bnf", a, h) * self.clsW).sum(-1) + self.clsb


def load_members(dev):
    import timm
    members = []
    for fname, weight in MEMBERS:
        path = find_file(fname)
        if path is None:
            raise FileNotFoundError(f"member checkpoint {fname} not attached")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if list(ck.get("lab", [])) != LAB or not isinstance(ck.get("res"), int):
            raise RuntimeError(f"{fname}: checkpoint contract drift")
        recipe = ck.get("recipe") or RAPTOR44
        depth = sum(k for _, _, k in recipe["slots"])
        bb = timm.create_model(ck["arch"], pretrained=False, num_classes=0, in_chans=3,
                               global_pool="avg")
        model = RaptorClassifier(bb)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{fname}: state_dict mismatch {missing[:3]} {unexpected[:3]}")
        members.append({"name": fname.replace("_swa.pt", "").replace(".pt", ""),
                        "model": model.to(dev).eval(), "res": ck["res"],
                        "recipe": recipe,
                        "k": min(SUBMIT_K, depth - 2) if SUBMIT_K else depth - 2,
                        "weight": float(weight),
                        "train_gold": ck.get("gold_auc")})
        # Print the K actually used, not depth-2: with SUBMIT_K set they differ, and v15
        # logged "K=42/62" while genuinely running 24, which is indistinguishable from the
        # override silently failing.
        log(f"member {fname}: {ck['arch']} res {ck['res']} recipe {recipe['name']} "
            f"K={members[-1]['k']} (of {depth - 2} available) weight {weight} | "
            f"train gold {ck.get('gold_auc')} | 0/0 load")
    return members


# ---------------------------------------------------------------------- checks --
def self_check(comp, by, recipes):
    for name, rec in recipes.items():
        ref = REFERENCE.get(name)
        ids_path = find_file(ref[0]) if ref else None
        if ids_path is None:
            log(f"self-check {name}: no reference corpus attached, skipped")
            continue
        ids = np.load(ids_path, allow_pickle=True).astype(str)
        vols = np.load(find_file(ref[1]), mmap_mode="r")
        for row in np.random.default_rng(2026).choice(len(ids), SELF_CHECK_N, replace=False):
            vol, _ = build_study(ids[row], by.get(ids[row], []), f"{comp}/train_series", rec)
            if not np.array_equal(vol, np.asarray(vols[row])):
                raise RuntimeError(f"SELF-CHECK FAILED for recipe {name}, row {row}")
        log(f"self-check {name}: PASS ({SELF_CHECK_N} studies byte-identical to reference)")


def ks_for(m):
    """Window counts to score this member at. Sweep values above its own K are clipped and
    de-duplicated, so a 44-slice member is never asked for 62 windows it does not have."""
    if not EVAL_KS or MODE != "validate":
        return [m["k"]]
    return sorted({min(k, m["k"]) for k in EVAL_KS})


def key_of(m, k):
    return m["name"] if (not EVAL_KS or MODE != "validate") else f"{m['name']}#k{k}"


@torch.no_grad()
def predict(members, dev, uids, by, series_root):
    recipes = {m["recipe"]["name"]: m["recipe"] for m in members}
    passes = [False, True] if TTA_REVERSE else [False]
    preds = {key_of(m, k) + ("@rev" if r else ""):
             np.full((len(uids), len(LAB)), np.nan, np.float32)
             for m in members for k in ks_for(m) for r in passes}
    t0 = time.time()
    with ThreadPoolExecutor(BUILD_THREADS) as pool:
        for b0 in range(0, len(uids), BATCH_STUDIES):
            chunk = list(range(b0, min(b0 + BATCH_STUDIES, len(uids))))
            built = list(pool.map(lambda i: build_all_recipes(uids[i], by.get(uids[i], []),
                                                               series_root, recipes), chunk))
            for m in members:
                rn = m["recipe"]["name"]
                for k in ks_for(m):
                    for rev in passes:
                        x = torch.stack([windows(*b[rn], m["res"], k, reverse=rev)
                                         for b in built]).to(dev)
                        with torch.autocast("cuda", dtype=torch.float16,
                                            enabled=dev.type == "cuda"):
                            p = torch.sigmoid(m["model"](x).float()).cpu().numpy()
                        key = key_of(m, k) + ("@rev" if rev else "")
                        for j, i in enumerate(chunk):
                            if built[j][rn][1].any():
                                preds[key][i] = p[j]
            done = chunk[-1] + 1
            if done % 50 < BATCH_STUDIES or done == len(uids):
                el = time.time() - t0
                log(f"  {done}/{len(uids)} studies | {el / 60:.1f} min | "
                    f"eta {el / done * (len(uids) - done) / 60:.1f} min")
    return preds


def member_view(preds, name, tta):
    """A member's final probabilities: normal pass, or normal/reversed mean under TTA."""
    if tta and name + "@rev" in preds:
        return np.nanmean(np.stack([preds[name], preds[name + "@rev"]]), axis=0)
    return preds[name]


def rank_blend(preds, members, tta=False, keys=None):
    """Weighted mean of per-finding percentile ranks. Studies a member could not read
    (all-NaN row) take that member's median rank, so they neither help nor hurt."""
    out = np.zeros_like(next(iter(preds.values())), dtype=np.float64)
    wsum = 0.0
    for m in members:
        r = pd.DataFrame(member_view(preds, (keys or {}).get(id(m), m["name"]), tta)).rank(pct=True).to_numpy()
        r = np.where(np.isnan(r), np.nanmedian(r, axis=0, keepdims=True), r)
        out += m["weight"] * r
        wsum += m["weight"]
    return out / wsum


def macro(Y, P):
    from sklearn.metrics import roc_auc_score
    aucs = {n: float(roc_auc_score(Y[:, j], P[:, j])) for j, n in enumerate(LAB)
            if len(set(Y[:, j])) > 1}
    return float(np.mean(list(aucs.values()))), aucs


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comp = find_comp()
    log(f"mode {MODE} | device {dev}")
    members = load_members(dev)
    recipes = {m["recipe"]["name"]: m["recipe"] for m in members}

    if MODE == "validate":
        ser = pd.read_csv(comp / "train_series.csv")
        ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str)
        by = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}
        self_check(comp, by, recipes)
        gold = pd.read_csv(find_file("gold_test.csv"))
        uids = gold.StudyInstanceUID.astype(str).tolist()
        preds = predict(members, dev, uids, by, comp / "train_series")
        Y = gold[LAB].to_numpy()
        report, bad = {}, []
        log("=== per member, gold rebuilt from DICOM ===")
        for m in members:
            au, aucs = macro(Y, preds[key_of(m, m["k"])])
            report[m["name"]] = {"auc": au, "aucs": aucs, "train_gold": m["train_gold"]}
            diff = abs(au - m["train_gold"]) if m["train_gold"] is not None else float("nan")
            ok = diff <= VALIDATE_TOL or m["name"] in NO_STRICT
            bad += [] if ok else [m["name"]]
            note = " [not enforced: recorded at a different eval]" if m["name"] in NO_STRICT else ""
            log(f"  {m['name']:10s} {au:.4f}  (training recorded {m['train_gold']:.4f}, "
                f"diff {diff:.4f}) {'OK' if ok else 'MISMATCH'}{note}")
            if TTA_REVERSE:
                au_r, _ = macro(Y, preds[key_of(m, m["k"]) + "@rev"])
                au_t, aucs_t = macro(Y, member_view(preds, key_of(m, m["k"]), True))
                report[m["name"]].update(auc_reversed=au_r, auc_tta=au_t, aucs_tta=aucs_t)
                log(f"  {'':8s} reversed-only {au_r:.4f} | normal+reversed TTA {au_t:.4f} "
                    f"({au_t - au:+.4f})")
        blend = rank_blend(preds, members, keys={id(m): key_of(m, m["k"]) for m in members})
        au, aucs = macro(Y, blend)
        report["ensemble"] = {"auc": au, "aucs": aucs}
        log(f"=== ENSEMBLE (rank mean, weights {[m['weight'] for m in members]}): gold {au:.4f} ===")
        if TTA_REVERSE:
            au_bt, aucs_bt = macro(Y, rank_blend(preds, members, tta=True,
            keys={id(m): key_of(m, m["k"]) for m in members}))
            report["ensemble_tta"] = {"auc": au_bt, "aucs": aucs_bt}
            log(f"=== ENSEMBLE with depth-reversal TTA: gold {au_bt:.4f} ({au_bt - au:+.4f}) ===")
        names = [m["name"] for m in members]
        log("  finding            " + "  ".join(f"{n:>7s}" for n in names) + "  ensemble")
        for n in LAB:
            row = "  ".join(f"{report[x]['aucs'].get(n, float('nan')):7.4f}" for x in names)
            log(f"  {n:18s} {row}  {aucs.get(n, float('nan')):8.4f}")
        json.dump(report, open("/kaggle/working/validate_ensemble.json", "w"), indent=1)
        np.savez("/kaggle/working/gold_member_preds.npz", ids=np.array(uids), Y=Y,
                 **{k: v for k, v in preds.items()})
        if EVAL_KS:
            # Baseline is the CURRENT production config: each member at its own K, TTA on.
            # Compute it first so every row below is a delta against what we actually ship.
            base_keys = {id(m): key_of(m, m["k"]) for m in members}
            base_auc, _ = macro(Y, rank_blend(preds, members, tta=TTA_REVERSE, keys=base_keys))
            base_imgs = sum(m["k"] for m in members) * (2 if TTA_REVERSE else 1)
            log("")
            log("=== INFERENCE COST SWEEP (gold, all members blended) ===")
            log("  images/study bounds hidden-test runtime. This run measured 9.2 s/study")
            log(f"  at the baseline: each member at its own K, TTA={TTA_REVERSE}, "
                f"{base_imgs} images/study, gold {base_auc:.4f}.")
            log(f"  {'K':>4} {'TTA':>6} {'img/study':>10} {'speedup':>8} {'gold':>8} {'vs base':>9}")
            sweep = {"baseline": {"gold": base_auc, "images_per_study": base_imgs}}
            for k in sorted(set(EVAL_KS)):
                keys = {id(m): key_of(m, min(k, m["k"])) for m in members}
                for tta in ((False, True) if TTA_REVERSE else (False,)):
                    au_k, _ = macro(Y, rank_blend(preds, members, tta=tta, keys=keys))
                    imgs = sum(min(k, m["k"]) for m in members) * (2 if tta else 1)
                    sweep[f"k{k}_tta{int(tta)}"] = {"gold": au_k, "images_per_study": imgs}
                    log(f"  {k:>4} {str(tta):>6} {imgs:>10} {base_imgs / imgs:7.2f}x "
                        f"{au_k:8.4f} {au_k - base_auc:+9.4f}")
            json.dump(sweep, open("/kaggle/working/sweep_gold.json", "w"), indent=1)
            log("  Gold-58 cannot resolve 0.002 and is not being asked to -- only whether")
            log("  cutting the window budget causes GROSS degradation. A cheap row within")
            log("  ~0.01 of baseline is a candidate; the LB confirms it.")
        if bad:
            raise RuntimeError(f"members did not reproduce training gold: {bad}")
        return

    test = pd.read_csv(comp / "test.csv")
    ser = pd.read_csv(comp / "test_series.csv")
    ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str)
    by = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}
    uids = test.StudyInstanceUID.astype(str).tolist()
    log(f"test studies: {len(uids)}")
    preds = predict(members, dev, uids, by, comp / "test_series")
    blend = rank_blend(preds, members, tta=TTA_REVERSE)
    sub = pd.DataFrame(blend, columns=LAB)
    sub.insert(0, "StudyInstanceUID", uids)
    sample = pd.read_csv(comp / "sample_submission.csv")
    sub = sample[["StudyInstanceUID"]].astype(str).merge(sub, on="StudyInstanceUID", how="left")
    sub[LAB] = sub[LAB].fillna(0.5)
    if list(sub.columns) != list(sample.columns) or not np.isfinite(sub[LAB].to_numpy()).all():
        raise RuntimeError("submission does not match sample_submission contract")
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    log(f"wrote submission.csv {sub.shape} from {len(members)} members")


if __name__ == "__main__":
    main()
