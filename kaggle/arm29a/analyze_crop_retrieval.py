"""arm29a: does a SECOND retrieval geometry add anything beyond the first?

Pre-registered, with k=32 and T=0.07 frozen from arm27 and the pooled representation fixed
before any of these numbers existed. Nothing here is selected on its own results.

    R20  retrieval in arm20's per-finding pooled space          (12 findings, adopted)
    R21  retrieval in arm21's per-target crop-pooled space      (5 findings, this arm)
    B    the 7-member fold-0 blend (6 global heads + arm21's own crop head)

THE PRIMARY COMPARISON is the increment over a blend that ALREADY CONTAINS R20:

    delta = AUC(B + R20 + R21) - AUC(B + R20)

and NOT AUC(B + R21) - AUC(B). A second retrieval geometry can be very different from its own
head and still duplicate what the first geometry already contributed, so the correlation that
decides novelty is corr(R21, R20), not corr(R21, arm21's head).

ADOPTION, fixed in advance:
    >= +0.0010            strong adoption
    +0.0005 .. +0.0010    keep only if consistent across findings
    <  +0.0005            do not deploy, however good R21 looks on its own

SCOPE. arm21 predicts five of twelve findings and abstains on the rest, exactly as the crop
member does in the blend, so R21 can move at most 5/12 of the macro. A +0.0010 macro
increment needs about +0.0024 average across its own five findings; both are reported.
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
K, T, W = 32, 0.07, 1.0          # member weight in the rank blend, as arm27 was deployed
STRONG, KEEP = 0.0010, 0.0005
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
    """Per-finding AUC over the studies where the score EXISTS.

    A member that abstains leaves NaN -- on the seven findings arm21 does not predict, and on
    the handful of studies with no valid crop window for a target. Those rows are skipped,
    which is the convention arm21's own scorer uses; a finding with no score at all stays
    NaN and drops out of the macro."""
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
    """Per-finding softmax-weighted kNN label vote, with window-validity honoured.

    A reference study with no valid window for a target carries no usable embedding for it
    and is excluded from that target's database; a query study with none gets NaN, so the
    blend falls back to the other members exactly as the crop member already does."""
    n_q = len(pool_q)
    out = np.full((n_q, 12), np.nan)
    sim_nn = np.full((n_q, 12), np.nan)
    for i, col in enumerate(cols):
        okr = np.ones(len(pool_r), bool) if valid_r is None else valid_r[:, i]
        okq = np.ones(n_q, bool) if valid_q is None else valid_q[:, i]
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
        yj = (Yref[okr][:, i] > 0.5).astype(np.float64)
        out[okq, col] = (w * yj[idx]).sum(1)
        sim_nn[okq, col] = s.max(1)
    return out, sim_nn


def main(crop_path, emb_path, out_path):
    with np.load(crop_path, allow_pickle=True) as f:
        c = {k: f[k] for k in f.files}
    with np.load(emb_path, allow_pickle=True) as f:
        e = {k: f[k] for k in f.files}

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
    cp = np.full((len(ids), 12), np.nan)
    for i, lab in enumerate(CROP_LAB):
        col = d["oof_pred"][:, i].astype(float)
        col[~d["oof_valid"][:, i].astype(bool)] = np.nan
        cp[:, LAB.index(lab)] = col
    members.append({"name": CROP[0], "pred": cp,
                    "columns": [LAB.index(x) for x in CROP_LAB], "weight": 1.0})
    crop_head = cp

    # R20: arm20 pooled space, all 12 findings
    o20 = {str(u): i for i, u in enumerate(e["ids"])}
    q20 = [o20[u] for u in ids]
    is_ref20 = e["is_reference"] == 1
    assert not is_ref20[q20].any(), "a query study is in arm20's reference database"
    R20, nn20 = knn(e["pool"][q20], e["pool"][is_ref20], e["y"][is_ref20],
                    list(range(12)), K, T)

    # R21: arm21 crop-pooled space, its five targets
    o21 = {str(u): i for i, u in enumerate(c["ids"])}
    q21 = [o21[u] for u in ids]
    is_ref21 = c["is_reference"] == 1
    assert not is_ref21[q21].any(), "a query study is in arm21's reference database"
    crop_cols = [LAB.index(x) for x in [str(v) for v in c["target_lab"]]]
    R21, nn21 = knn(c["pool"][q21], c["pool"][is_ref21], c["y"][is_ref21], crop_cols, K, T,
                    valid_q=c["valid"][q21], valid_r=c["valid"][is_ref21])

    m20 = {"name": "R20", "pred": R20, "columns": list(range(12)), "weight": W}
    m21 = {"name": "R21", "pred": R21, "columns": crop_cols, "weight": W}

    def macro(P):
        return float(np.nanmean(aucs(Y, P)))

    a_solo21 = aucs(Y, R21)
    a_solo20 = aucs(Y, R20)
    print(f"R21 solo over its {len(crop_cols)} targets: "
          f"{np.nanmean(a_solo21[crop_cols]):.4f}   (arm21's own head: "
          f"{np.nanmean(aucs(Y, crop_head)[crop_cols]):.4f}, R20 on the same five: "
          f"{np.nanmean(a_solo20[crop_cols]):.4f})\n")

    ladder = {}
    combos = [("B", members), ("B+R20", members + [m20]), ("B+R21", members + [m21]),
              ("B+R20+R21", members + [m20, m21]),
              ("arm21 head + R21", [{"name": "h", "pred": crop_head, "columns": crop_cols,
                                     "weight": 1.0}, m21])]
    print(f"{'configuration':<20} {'macro(12)':>10} {'macro(5)':>10}")
    for tag, mem in combos:
        P = rank_blend(mem)
        a = aucs(Y, P)
        ladder[tag] = {"macro12": float(np.nanmean(a)),
                       "macro5": float(np.nanmean(a[crop_cols])),
                       "aucs": a}
        print(f"{tag:<20} {ladder[tag]['macro12']:>10.4f} {ladder[tag]['macro5']:>10.4f}")

    # ---- THE primary: increment over a blend that already contains R20 ----------------
    base = ladder["B+R20"]["aucs"]
    new = ladder["B+R20+R21"]["aucs"]
    d12 = float(np.nanmean(new) - np.nanmean(base))
    d5 = float(np.nanmean(new[crop_cols]) - np.nanmean(base[crop_cols]))
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(ids), len(ids))
        bm = [{**m, "pred": m["pred"][i]} for m in members + [m20]]
        bn = bm + [{**m21, "pred": R21[i]}]
        boot.append(np.nanmean(aucs(Y[i], rank_blend(bn)))
                    - np.nanmean(aucs(Y[i], rank_blend(bm))))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nPRIMARY  AUC(B+R20+R21) - AUC(B+R20) = {d12:+.4f} macro over 12  "
          f"({d5:+.4f} over R21's own 5)")
    print(f"         95% CI {lo:+.4f}..{hi:+.4f}, P(>0) {(boot > 0).mean():.2f}")
    print(f"secondary AUC(B+R21) - AUC(B)        = "
          f"{ladder['B+R21']['macro12'] - ladder['B']['macro12']:+.4f}  "
          f"(the comparison that would OVERSTATE novelty)")

    # ---- the two correlations, the second being the one that matters -------------------
    print(f"\n{'finding':<18} {'R21':>7} {'R20':>7} {'d(12)':>8}  "
          f"{'rho(R21,head21)':>16} {'rho(R21,R20)':>13}")
    rows = {}
    for col in crop_cols:
        ok = np.isfinite(R21[:, col]) & np.isfinite(crop_head[:, col])
        r_head = float(spearmanr(R21[ok, col], crop_head[ok, col]).statistic)
        ok2 = np.isfinite(R21[:, col]) & np.isfinite(R20[:, col])
        r_r20 = float(spearmanr(R21[ok2, col], R20[ok2, col]).statistic)
        rows[LAB[col]] = {"r21_auc": float(a_solo21[col]), "r20_auc": float(a_solo20[col]),
                          "delta12": float(new[col] - base[col]),
                          "rho_vs_head21": r_head, "rho_vs_R20": r_r20}
        print(f"{LAB[col]:<18} {a_solo21[col]:>7.4f} {a_solo20[col]:>7.4f} "
              f"{new[col] - base[col]:>+8.4f}  {r_head:>16.3f} {r_r20:>13.3f}")
    mean_rho_r20 = float(np.mean([v["rho_vs_R20"] for v in rows.values()]))
    print(f"mean rho(R21, R20) = {mean_rho_r20:.3f} -- the novelty measure; "
          f"mean rho(R21, arm21's head) = "
          f"{np.mean([v['rho_vs_head21'] for v in rows.values()]):.3f}")

    consistent = sum(1 for v in rows.values() if v["delta12"] > 0)
    if d12 >= STRONG:
        v = ("STRONG ADOPTION -- a second retrieval geometry adds beyond the first; extract "
             "arm23_f0_448 next and build the constrained multi-retrieval stack")
    elif d12 >= KEEP and consistent >= 4:
        v = (f"KEEP -- {d12:+.4f} with {consistent}/{len(crop_cols)} of its own findings "
             f"positive, consistent enough to carry forward")
    elif d12 >= KEEP:
        v = (f"BORDERLINE AND INCONSISTENT -- {d12:+.4f} but only {consistent}/"
             f"{len(crop_cols)} findings positive; do not deploy on this evidence")
    else:
        v = ("DO NOT DEPLOY -- below +0.0005 over a blend already containing R20. If R21 solo "
             "looks good, that is duplication of what R20 already contributed")
    print(f"\nVERDICT: {v}")

    # ---- the free extension: meta-features for the constrained stack ------------------
    np.savez_compressed(out_path.replace(".json", "_meta_features.npz"),
                        ids=np.array(ids), y=Y,
                        head_blend=rank_blend(members), R20=R20, R21=R21,
                        diff=R20 - R21, absdiff=np.abs(R20 - R21),
                        nn_sim_R20=nn20, nn_sim_R21=nn21,
                        crop_cols=np.array(crop_cols), lab=np.array(LAB))
    json.dump({"k": K, "temperature": T, "weight": W,
               "ladder": {k: {"macro12": v["macro12"], "macro5": v["macro5"]}
                          for k, v in ladder.items()},
               "primary_delta_macro12": d12, "primary_delta_macro5": d5,
               "primary_ci": [float(lo), float(hi)],
               "primary_p_positive": float((boot > 0).mean()),
               "secondary_delta_over_B": ladder["B+R21"]["macro12"] - ladder["B"]["macro12"],
               "mean_rho_R21_R20": mean_rho_r20,
               "per_finding": rows, "thresholds": {"strong": STRONG, "keep": KEEP},
               "verdict": v}, io.open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path} and the meta-feature file for the constrained stack")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["kaggle/arm29a/output/arm29a_crop_embeddings.npz",
                            "kaggle/arm27/output/arm27_embeddings.npz",
                            "kaggle/arm29a/output/arm29a_retrieval.json"]))
