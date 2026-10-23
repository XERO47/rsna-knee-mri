"""arm30-probe decision: is there local evidence the global pooling throws away, and is it
worth a real training run?

The probe's own macro AUC is not the decision. Retrieval taught us that a member's value must
be measured against the strongest cross-fitted ensemble available, because the marginal
contribution decays as the baseline improves and a fold-model delta is an upper bound
(EXPERIMENTS.md section 2b):

    >= +0.0040   LB candidate
    +0.0020-0.0040   interesting research, do not submit
    <  +0.0020   reject for deployment

But arm30-probe is a PROBE, not a candidate member, and its purpose is different: it decides
whether to spend a real training run on arm30. The pre-registered rule for that:

    patch-MIL adds >= +0.0015 to the strong ensemble   -> train arm30
    >= +0.0030                                          -> highest-priority architecture
    ~0                                                   -> kill the expensive version

That is a lower bar than deployment on purpose. A frozen linear read-out over patches of an
encoder that was never trained to localise anything is a floor on what a model trained for it
could do; the question is only whether the signal exists at all.

THREE COMPARISONS, in increasing strictness:
  1. patch-MIL vs the MATCHED control -- the real attention head at the SAME window count.
     Confounder-free: same encoder, same windows, only the pooling differs.
  2. patch-MIL added to the 7-member fold-0 blend.
  3. patch-MIL added to that blend plus all three retrieval members, which is the strongest
     cross-fitted thing we have (0.8759).

The ZERO-TRAINING arm is the one to read first. It scores patches with the model's OWN trained
clsW and pools the top-k. Nothing is fitted, so it cannot be overfitting 870 studies, and if
it beats the attention head on the focal findings then the evidence was already inside the
features and the pooling was discarding it.
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
FOCAL = ["Fracture", "Contusion", "Lateral Meniscus", "Medial Meniscus", "Baker's"]
GLOBALS = [("arm14_f0_attn", "kaggle/arm14c/output/arm14_f0_attn_preds.npz"),
           ("arm16_f0_canon", "kaggle/arm16/output/arm16_f0_canon_preds.npz"),
           ("arm19_f0_aug", "kaggle/arm19/output/arm19_f0_aug_preds.npz"),
           ("arm20_f0_noaug", "kaggle/arm20/output/arm20_f0_noaug_preds.npz"),
           ("arm20_f0_aug", "kaggle/arm20/output/arm20_f0_aug_preds.npz"),
           ("arm23_f0_448", "kaggle/arm23/output/arm23_f0_448_preds.npz")]
CROP = "kaggle/arm21/output/arm21_f0_crop_preds.npz"
TRAIN_ARM30, TOP_PRIORITY = 0.0015, 0.0030
N_BOOT = 2000


def ranks(v):
    r = pd.DataFrame(v).rank(pct=True).to_numpy()
    med = np.nanmedian(r, axis=0, keepdims=True)
    return np.where(np.isnan(r), np.where(np.isnan(med), 0.5, med), r)


def rank_blend(members):
    out = np.zeros((len(members[0]["pred"]), 12))
    wsum = np.zeros(12)
    for m in members:
        r = ranks(m["pred"])
        for c in m["columns"]:
            out[:, c] += m["weight"] * r[:, c]
            wsum[c] += m["weight"]
    return out / np.maximum(wsum, 1e-9)[None, :]


def aucs(Y, P):
    out = np.full(12, np.nan)
    for j in range(12):
        ok = np.isfinite(P[:, j])
        if ok.sum() < 20:
            continue
        y = (Y[ok, j] > 0.5).astype(int)
        if 0 < y.sum() < len(y):
            out[j] = roc_auc_score(y, P[ok, j])
    return out


def knn(pool_q, pool_r, Yref, cols, k, T, valid_q=None, valid_r=None):
    out = np.full((len(pool_q), 12), np.nan)
    for i, col in enumerate(cols):
        okr = np.ones(len(pool_r), bool) if valid_r is None else valid_r[:, i]
        okq = np.ones(len(pool_q), bool) if valid_q is None else valid_q[:, i]
        if okr.sum() <= k or not okq.any():
            continue
        zr = pool_r[okr, i].astype(np.float64)
        zr /= np.maximum(np.linalg.norm(zr, axis=1, keepdims=True), 1e-12)
        zq = pool_q[okq, i].astype(np.float64)
        zq /= np.maximum(np.linalg.norm(zq, axis=1, keepdims=True), 1e-12)
        sim = zq @ zr.T
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        s = np.take_along_axis(sim, idx, 1)
        w = np.exp((s - s.max(1, keepdims=True)) / T)
        w /= w.sum(1, keepdims=True)
        out[okq, col] = (w * (Yref[okr][:, i] > 0.5).astype(np.float64)[idx]).sum(1)
    return out


def main(probe_path, out_path):
    d = {k: v for k, v in np.load(probe_path, allow_pickle=True).items()}
    ids = [str(u) for u in d["oof_ids"]]
    Y = d["oof_truth"]
    kprim = int(d["topk"])

    members = []
    for name, path in GLOBALS:
        p = np.load(path, allow_pickle=True)
        assert [str(u) for u in p["oof_ids"]] == ids, f"{name} is a different fold/order"
        members.append({"name": name, "pred": p["oof_pred"], "columns": list(range(12)),
                        "weight": 1.0})
    p = np.load(CROP, allow_pickle=True)
    cp = np.full((len(ids), 12), np.nan)
    for i, lab in enumerate(CROP_LAB):
        col = p["oof_pred"][:, i].astype(float)
        col[~p["oof_valid"][:, i].astype(bool)] = np.nan
        cp[:, LAB.index(lab)] = col
    crop_cols = [LAB.index(x) for x in CROP_LAB]
    members.append({"name": "arm21_crop", "pred": cp, "columns": crop_cols, "weight": 1.0})

    # the three retrieval members, rebuilt exactly as arm27/29a/29b measured them
    load = lambda q: {k: v for k, v in np.load(q, allow_pickle=True).items()}
    e20 = load("kaggle/arm27/output/arm27_embeddings.npz")
    e23 = load("kaggle/arm29b/output/arm29b_embeddings.npz")
    c21 = load("kaggle/arm29a/output/arm29a_crop_embeddings.npz")
    retr = []
    for tag, e, cols, vq, vr in (
            ("R20", e20, list(range(12)), None, None),
            ("R23", e23, list(range(12)), None, None),
            ("R21", c21, crop_cols, "valid", "valid")):
        o = {str(u): i for i, u in enumerate(e["ids"])}
        q = [o[u] for u in ids]
        isr = e["is_reference"] == 1
        assert not isr[q].any(), f"{tag}: a query study is in its own reference database"
        R = knn(e["pool"][q], e["pool"][isr], e["y"][isr], cols, 32, 0.07,
                valid_q=e[vq][q] if vq else None, valid_r=e[vr][isr] if vr else None)
        retr.append({"name": tag, "pred": R, "columns": cols, "weight": 1.0})

    head = {"name": "head_k8", "pred": d["head_k8"], "columns": list(range(12)), "weight": 1.0}
    arms = {"zero-training": d["zero_patch"]}
    arms.update({f"trained k={k}": d[f"trained_k{k}"] for k in (4, 8, 16)
                 if f"trained_k{k}" in d})

    a_ctrl = aucs(Y, d["head_k8"])
    print(f"{len(ids)} studies | {int(d['windows'])} windows | primary k={kprim}\n")
    print("1. AGAINST THE MATCHED CONTROL (same encoder, same windows, only the pooling "
          "differs)")
    print(f"   the real attention head at K={int(d['windows'])}: {np.nanmean(a_ctrl):.4f}\n")
    print(f"{'arm':<16} {'macro':>8} {'vs ctrl':>9}   " +
          "  ".join(f"{f[:9]:>9}" for f in FOCAL))
    rows = {}
    for tag, P in arms.items():
        a = aucs(Y, P)
        rows[tag] = {"macro": float(np.nanmean(a)), "aucs": a}
        print(f"{tag:<16} {np.nanmean(a):>8.4f} {np.nanmean(a) - np.nanmean(a_ctrl):>+9.4f}   " +
              "  ".join(f"{a[LAB.index(f)] - a_ctrl[LAB.index(f)]:>+9.4f}" for f in FOCAL))
    print(f"{'(control itself)':<16} {np.nanmean(a_ctrl):>8.4f} {'':>9}   " +
          "  ".join(f"{a_ctrl[LAB.index(f)]:>9.4f}" for f in FOCAL))
    print("   the five columns are the FOCAL findings -- where the hypothesis says patch "
          "evidence should show up")

    print("\n2 & 3. ADDED TO THE STRONG CROSS-FITTED ENSEMBLES")
    base7 = [m for m in members]
    base10 = members + retr
    for tag, b in (("7 members", base7), ("7 + all three retrievals", base10)):
        print(f"   {tag}: {np.nanmean(aucs(Y, rank_blend(b))):.4f}")
    print(f"\n{'arm':<16} {'+7 members':>12} {'+7+retrieval':>14} {'CI (strongest)':>22} "
          f"{'P(>0)':>7}")
    rng = np.random.default_rng(0)
    for tag, P in arms.items():
        m = {"name": "patch", "pred": P, "columns": list(range(12)), "weight": 1.0}
        d7 = float(np.nanmean(aucs(Y, rank_blend(base7 + [m])))
                   - np.nanmean(aucs(Y, rank_blend(base7))))
        d10 = float(np.nanmean(aucs(Y, rank_blend(base10 + [m])))
                    - np.nanmean(aucs(Y, rank_blend(base10))))
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(ids), len(ids))
            bm = [{**x, "pred": x["pred"][i]} for x in base10]
            boot.append(np.nanmean(aucs(Y[i], rank_blend(bm + [{**m, "pred": P[i]}])))
                        - np.nanmean(aucs(Y[i], rank_blend(bm))))
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows[tag].update(delta_7=d7, delta_strong=d10, ci=[float(lo), float(hi)],
                         p_positive=float((boot > 0).mean()))
        print(f"{tag:<16} {d7:>+12.4f} {d10:>+14.4f}   {lo:>+9.4f}..{hi:<+9.4f} "
              f"{(boot > 0).mean():>7.2f}")

    # is the patch arm decorrelated from the head it came from?
    print(f"\n{'arm':<16} {'rho vs its own head':>21} {'rho vs the strong blend':>25}")
    sb = rank_blend(base10)
    for tag, P in arms.items():
        rh = np.nanmean([spearmanr(P[:, j], d["head_k8"][:, j]).statistic for j in range(12)])
        rb = np.nanmean([spearmanr(P[:, j], sb[:, j]).statistic for j in range(12)])
        rows[tag].update(rho_vs_head=float(rh), rho_vs_blend=float(rb))
        print(f"{tag:<16} {rh:>21.3f} {rb:>25.3f}")

    best = max(rows, key=lambda t: rows[t]["delta_strong"])
    bd = rows[best]["delta_strong"]
    focal_gain = float(np.nanmean([rows[best]["aucs"][LAB.index(f)] - a_ctrl[LAB.index(f)]
                                  for f in FOCAL]))
    if bd >= TOP_PRIORITY:
        v = (f"TRAIN arm30 AS THE TOP PRIORITY -- {best} adds {bd:+.4f} to the strongest "
             f"cross-fitted ensemble we have, from a FROZEN encoder never trained to localise")
    elif bd >= TRAIN_ARM30:
        v = (f"TRAIN arm30 -- {best} adds {bd:+.4f}, above the pre-registered +0.0015 for "
             f"committing a real run")
    elif focal_gain > 0.005:
        v = (f"MIXED -- the blend gain is only {bd:+.4f}, but patch-MIL beats the matched "
             f"control by {focal_gain:+.4f} on the focal findings. The local evidence exists "
             f"and is not additive to this ensemble; arm30 would have to justify itself on the "
             f"focal targets alone")
    else:
        v = (f"KILL arm30 -- the best patch arm adds {bd:+.4f} to the strong ensemble and "
             f"{focal_gain:+.4f} on the focal findings against a matched control. The pooling "
             f"is not discarding recoverable evidence at this granularity")
    print(f"\nVERDICT: {v}")
    json.dump({"n": len(ids), "windows": int(d["windows"]), "topk_primary": kprim,
               "matched_control_macro": float(np.nanmean(a_ctrl)),
               "baseline_7": float(np.nanmean(aucs(Y, rank_blend(base7)))),
               "baseline_7_plus_retrieval": float(np.nanmean(aucs(Y, rank_blend(base10)))),
               "arms": {t: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in r.items()} for t, r in rows.items()},
               "focal": FOCAL, "best_arm": best, "best_delta_strong": bd,
               "focal_gain_vs_control": focal_gain,
               "thresholds": {"train_arm30": TRAIN_ARM30, "top_priority": TOP_PRIORITY},
               "verdict": v}, io.open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["kaggle/arm30probe/output/arm30probe_preds.npz",
                            "kaggle/arm30probe/output/arm30probe_decision.json"]))
