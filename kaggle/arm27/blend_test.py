"""arm27e: retrieval beat a single head by +0.0031. Does it survive an ENSEMBLE?

Beating one model is the easy test. Every idea that died this month died at the next step,
because a blend of six heads already averages away most of what one head gets wrong. The
question that decides whether retrieval reaches a submission is:

    Does a kNN vote over arm20's per-finding pooled embeddings add anything to a blend that
    ALREADY CONTAINS arm20's own head?

This is answerable locally and exactly, because ten fold-0 models share the same 870
held-out studies. Four of them (arm14_f0_attn, arm16_f0_base, arm18_f0_base,
arm19_f0_noaug) turn out to be bit-identical -- they are the same control arm re-run -- so
they count as one member.

BASELINE, chosen to be hostile to the new member rather than flattering:
    arm14_f0_attn   ConvNeXt-small, 10 epochs        0.8616
    arm16_f0_canon  + orientation canonicalisation   0.8630
    arm19_f0_aug    + geometric augmentation         0.8612
    arm20_f0_noaug  20 epochs  <-- the retrieval's OWN PARENT           0.8620
    arm20_f0_aug    20 epochs + augmentation         0.8628
    arm23_f0_448    448 px                           0.8627
    arm21_f0_crop   ROI crops, votes on 5 findings   0.8134
Keeping arm20_f0_noaug in the baseline is the conservative choice: retrieval must add
something beyond the head that produced its own embedding space, not merely re-import it.
The run also reports the variant with that parent removed, since the two bracket the truth.

BLENDING is the submission's own rule, so a gain here means the same thing it would mean in
`submit12`: per-finding percentile-rank mean over the members that vote on that finding,
each with weight 1. The retrieval member is added exactly like the crop member was.
"""
import io
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
CROP_LAB = ["MCL", "Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis"]
GLOBALS = [("arm14_f0_attn", "kaggle/arm14c/output/arm14_f0_attn_preds.npz"),
           ("arm16_f0_canon", "kaggle/arm16/output/arm16_f0_canon_preds.npz"),
           ("arm19_f0_aug", "kaggle/arm19/output/arm19_f0_aug_preds.npz"),
           ("arm20_f0_noaug", "kaggle/arm20/output/arm20_f0_noaug_preds.npz"),
           ("arm20_f0_aug", "kaggle/arm20/output/arm20_f0_aug_preds.npz"),
           ("arm23_f0_448", "kaggle/arm23/output/arm23_f0_448_preds.npz")]
CROP = ("arm21_f0_crop", "kaggle/arm21/output/arm21_f0_crop_preds.npz")
K, T, WEIGHTS = 32, 0.07, (0.5, 1.0)
N_BOOT = 2000


def ranks(v):
    """submit12's member view: percentile rank per column, NaN filled with the column median."""
    r = pd.DataFrame(v).rank(pct=True).to_numpy()
    med = np.nanmedian(r, axis=0, keepdims=True)
    return np.where(np.isnan(r), np.where(np.isnan(med), 0.5, med), r)


def rank_blend(members):
    """Exactly submit12.rank_blend: per-finding weighted mean of member percentile ranks."""
    out = np.zeros((len(members[0]["pred"]), 12))
    wsum = np.zeros(12)
    for m in members:
        r = ranks(m["pred"])
        for c in m["columns"]:
            out[:, c] += m["weight"] * r[:, c]
            wsum[c] += m["weight"]
    return out / wsum[None, :]


def aucs(Y, P):
    out = np.full(12, np.nan)
    for j in range(12):
        y = (Y[:, j] > 0.5).astype(int)
        if 0 < y.sum() < len(y):
            out[j] = roc_auc_score(y, P[:, j])
    return out


def knn_pool(pool_q, pool_r, Yref, k, T):
    out = np.empty((len(pool_q), 12))
    for j in range(12):
        zq = pool_q[:, j] / np.linalg.norm(pool_q[:, j], axis=1, keepdims=True)
        zr = pool_r[:, j] / np.linalg.norm(pool_r[:, j], axis=1, keepdims=True)
        sim = zq.astype(np.float64) @ zr.astype(np.float64).T
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        s = np.take_along_axis(sim, idx, 1)
        w = np.exp((s - s.max(1, keepdims=True)) / T)
        w /= w.sum(1, keepdims=True)
        out[:, j] = (w * (Yref[idx][:, :, j] > 0.5)).sum(1)
    return out


def main(emb_path, out_path):
    ref = np.load(GLOBALS[3][1], allow_pickle=True)
    ids = [str(u) for u in ref["oof_ids"]]
    Y = ref["oof_truth"]

    members = []
    for name, path in GLOBALS:
        d = np.load(path, allow_pickle=True)
        assert [str(u) for u in d["oof_ids"]] == ids, f"{name} is a different fold"
        members.append({"name": name, "pred": d["oof_pred"], "columns": list(range(12)),
                        "weight": 1.0})
    d = np.load(CROP[1], allow_pickle=True)
    assert [str(u) for u in d["oof_ids"]] == ids
    cp = np.full((len(ids), 12), np.nan)
    for i, lab in enumerate(CROP_LAB):
        col = d["oof_pred"][:, i].astype(float)
        col[~d["oof_valid"][:, i].astype(bool)] = np.nan
        cp[:, LAB.index(lab)] = col
    members.append({"name": CROP[0], "pred": cp,
                    "columns": [LAB.index(x) for x in CROP_LAB], "weight": 1.0})

    with np.load(emb_path, allow_pickle=True) as f:
        e = {k: f[k] for k in f.files}
    is_ref = e["is_reference"] == 1
    order = {str(u): i for i, u in enumerate(e["ids"])}
    qrows = [order[u] for u in ids]
    assert not is_ref[qrows].any(), "a query study is also in the reference database"
    retr = knn_pool(e["pool"][qrows], e["pool"][is_ref], e["y"][is_ref], K, T)
    print(f"retrieval member alone: macro {np.nanmean(aucs(Y, retr)):.4f} "
          f"(reference {int(is_ref.sum())} training studies, k={K}, T={T})")
    rho = np.nanmean([spearmanr(rank_blend(members)[:, j], retr[:, j]).statistic
                      for j in range(12)])
    print(f"Spearman(baseline blend, retrieval) = {rho:.3f}\n")

    out = {"k": K, "T": T, "retrieval_macro": float(np.nanmean(aucs(Y, retr))),
           "spearman_vs_blend": float(rho), "variants": {}}
    for drop_parent in (False, True):
        base_members = [m for m in members if not (drop_parent and m["name"] == "arm20_f0_noaug")]
        a_base = aucs(Y, rank_blend(base_members))
        macro = float(np.nanmean(a_base))
        tag = "parent removed" if drop_parent else "parent in the blend"
        print(f"--- {tag}: {len(base_members)} members, macro {macro:.4f}")
        v = {"n_members": len(base_members), "baseline_macro": macro, "weights": {}}
        for w in WEIGHTS:
            withr = base_members + [{"name": "retrieval", "pred": retr,
                                     "columns": list(range(12)), "weight": w}]
            a_new = aucs(Y, rank_blend(withr))
            delta = float(np.nanmean(a_new) - macro)
            boot = []
            rngb = np.random.default_rng(0)
            for _ in range(N_BOOT):
                i = rngb.integers(0, len(ids), len(ids))
                bm = [{**m, "pred": m["pred"][i]} for m in base_members]
                bw = bm + [{"name": "retrieval", "pred": retr[i], "columns": list(range(12)),
                            "weight": w}]
                boot.append(np.nanmean(aucs(Y[i], rank_blend(bw)))
                            - np.nanmean(aucs(Y[i], rank_blend(bm))))
            boot = np.array(boot)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            npos = int((a_new > a_base).sum())
            print(f"  + retrieval @ weight {w:<4} macro {np.nanmean(a_new):.4f}  "
                  f"delta {delta:+.4f}  95% CI {lo:+.4f}..{hi:+.4f}  "
                  f"P(>0) {(boot > 0).mean():.2f}  findings improved {npos}/12")
            v["weights"][str(w)] = {"macro": float(np.nanmean(a_new)), "delta": delta,
                                    "ci": [float(lo), float(hi)],
                                    "p_positive": float((boot > 0).mean()),
                                    "findings_improved": npos,
                                    "per_finding": {LAB[j]: float(a_new[j] - a_base[j])
                                                    for j in range(12)}}
        out["variants"][tag] = v

    strict = out["variants"]["parent in the blend"]["weights"]["1.0"]
    print(f"\nper-finding delta at weight 1.0 with the parent in the blend:")
    for j, name in enumerate(LAB):
        print(f"  {name:<18} {strict['per_finding'][name]:+.4f}")
    if strict["delta"] >= 0.0015 and strict["p_positive"] >= 0.90:
        verdict = ("SUBMIT -- retrieval adds to a 7-member blend that already contains its own "
                   "parent head. Next step is an LB submission, not more local analysis")
    elif strict["delta"] >= 0.0015:
        verdict = ("PROMISING BUT NOISY -- the point estimate clears the threshold, the CI does "
                   "not exclude zero on 870 studies; confirm on a second fold before submitting")
    elif out["variants"]["parent removed"]["weights"]["1.0"]["delta"] >= 0.0015:
        verdict = ("RE-IMPORTS ITS PARENT -- retrieval helps only when arm20_f0_noaug is absent, "
                   "so it is carrying that head's signal rather than new information")
    else:
        verdict = ("REJECT AT THE ENSEMBLE LEVEL -- the +0.0031 against a single head does not "
                   "survive blending; the ensemble already knows what the neighbourhood knows")
    print(f"\nVERDICT: {verdict}")
    out["verdict"] = verdict
    json.dump(out, io.open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["kaggle/arm27/output/arm27_embeddings.npz",
                            "kaggle/arm27/output/arm27_blend.json"]))
