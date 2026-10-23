"""arm27d: the mean-embedding result split the findings cleanly. Is the split a mechanism?

WHAT arm27a/b/c FOUND. A weighted kNN vote in arm20's study-MEAN embedding scores 0.8230
macro on its own -- the representation really is organised by pathology -- but blending it
into arm20's own head is worth -0.0002 macro, so the whole thing was rejected. The
per-finding column, however, did not look like noise:

    helped (w=0.20)   PF OA +0.0058, ACL +0.0042, Effusion +0.0042, Synovitis +0.0038,
                      Medial OA +0.0021, Lateral OA +0.0018, MCL +0.0005
    hurt              Fracture -0.0073, Contusion -0.0071, Lateral Meniscus -0.0058,
                      Medial Meniscus -0.0025, Baker's -0.0023

Every finding that gained is DIFFUSE -- osteoarthritis, effusion, synovitis, a ligament
that changes the whole joint's appearance. Every finding that lost is FOCAL -- a fracture
line, a bone bruise, a meniscal tear, a cyst. That is exactly what a study-MEAN embedding
should do: averaging 24 windows preserves what the whole knee looks like and washes out a
small lesion in two slices, so the nearest neighbours of a fractured knee are knees that
look broadly similar, not knees that are also fractured.

THE PREDICTION, written down before this script was run. arm27 also saved `pool`, the
per-finding attention-pooled vector -- the very thing arm20's classifier reads, and the one
place where a focal finding is NOT averaged away, because the attention concentrated on the
windows that matter for that finding. If the diffuse/focal story is right:

    1. retrieval in finding j's OWN pooled space must beat mean-space retrieval BY MORE on
       the focal findings than on the diffuse ones;
    2. the focal group's mean blend delta must move up from its mean-space value of
       -0.0050.

If the focal findings stay negative in pooled space, the diffuse/focal story is WRONG and
the real explanation is something else -- most likely that those findings are simply rare
and their neighbourhoods are dominated by negatives.

SELECTION. Picking the findings that won on these same 870 studies is exactly the trap the
pre-registration exists to avoid, so the per-finding selective blend is reported WITH its
null: the same selection applied to permuted scores, which says how much a chooser gains
from noise alone. A selective blend is only interesting if it clears that null.
"""
import importlib.util
import io
import json
import sys

import numpy as np
from scipy.stats import spearmanr

spec = importlib.util.spec_from_file_location(
    "ar", str(__file__).replace("analyze_retrieval_pool.py", "analyze_retrieval.py"))
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)

# fixed before running, from the arm27a/b/c per-finding column
FOCAL = ["Fracture", "Contusion", "Lateral Meniscus", "Medial Meniscus", "Baker's"]
K, T, W = 32, 0.07, 0.20
N_NULL = 400


def per_finding_scores(pool_q, pool_r, Yref, k, T):
    """Score A, but finding j's neighbours are found in finding j's own pooled space."""
    out = np.empty((len(pool_q), Yref.shape[1]), np.float64)
    for j in range(Yref.shape[1]):
        zq, zr = ar.unit(pool_q[:, j].astype(np.float64)), ar.unit(pool_r[:, j].astype(np.float64))
        sim = zq @ zr.T
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        s = np.take_along_axis(sim, idx, 1)
        w = np.exp((s - s.max(1, keepdims=True)) / T)
        w /= w.sum(1, keepdims=True)
        out[:, j] = (w * (Yref[idx][:, :, j] > 0.5)).sum(1)
    return out


def main(emb_path, base_path, out_path):
    with np.load(emb_path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    LAB = [str(x) for x in d["lab"]]
    ref = d["is_reference"] == 1
    Yref, Yq = d["y"][ref], d["y"][~ref]
    with np.load(base_path, allow_pickle=True) as f:
        b = {k: f[k] for k in f.files}
    pos = {str(u): i for i, u in enumerate(b["oof_ids"])}
    base = b["oof_pred"][[pos[str(u)] for u in d["ids"][~ref]]]
    base_rank = np.stack([ar.pct_rank(base[:, j]) for j in range(12)], 1)
    a_base = ar.aucs(Yq, base)
    macro_base = float(np.nanmean(a_base))

    s_mean = ar.score_A(ar.unit(d["z"][~ref].astype(np.float64))
                        @ ar.unit(d["z"][ref].astype(np.float64)).T, Yref, K, T)
    s_pool = per_finding_scores(d["pool"][~ref], d["pool"][ref], Yref, K, T)

    res = {}
    for tag, s in (("mean", s_mean), ("pool", s_pool)):
        res[tag] = {"auc": ar.aucs(Yq, s),
                    "rho": np.array([spearmanr(base[:, j], s[:, j]).statistic for j in range(12)]),
                    "delta": ar.blend_auc(Yq, base_rank, s, W) - a_base,
                    "score": s}

    focal = np.array([n in FOCAL for n in LAB])
    print(f"baseline macro {macro_base:.4f} | k={K} T={T} weight {W:.0%}\n")
    print(f"{'finding':<18} {'kind':<8} {'retr(mean)':>10} {'retr(pool)':>10} "
          f"{'d(mean)':>9} {'d(pool)':>9} {'change':>9}")
    for j, name in enumerate(LAB):
        dm, dp = res["mean"]["delta"][j], res["pool"]["delta"][j]
        print(f"{name:<18} {'FOCAL' if focal[j] else 'diffuse':<8} "
              f"{res['mean']['auc'][j]:>10.4f} {res['pool']['auc'][j]:>10.4f} "
              f"{dm:>+9.4f} {dp:>+9.4f} {dp - dm:>+9.4f}")
    for tag in ("mean", "pool"):
        print(f"{'MACRO ' + tag:<18} {'':<8} {np.nanmean(res[tag]['auc']):>10.4f} "
              f"{'':>10} {np.nanmean(res[tag]['delta']):>+9.4f}"
              if tag == "mean" else
              f"{'MACRO ' + tag:<18} {'':<8} {'':>10} {np.nanmean(res[tag]['auc']):>10.4f} "
              f"{'':>9} {np.nanmean(res[tag]['delta']):>+9.4f}")

    # ---- prediction 1: pooled space must help the FOCAL findings more -----------
    gain = res["pool"]["delta"] - res["mean"]["delta"]
    g_focal, g_diff = float(gain[focal].mean()), float(gain[~focal].mean())
    print(f"\nPREDICTION 1  pooled space helps focal findings more than diffuse ones")
    print(f"  focal   {g_focal:+.4f}   diffuse {g_diff:+.4f}   "
          f"-> {'CONFIRMED' if g_focal > g_diff else 'FALSIFIED'}")
    # ---- prediction 2: the focal group must stop losing --------------------------
    f_mean, f_pool = float(res["mean"]["delta"][focal].mean()), float(res["pool"]["delta"][focal].mean())
    print(f"PREDICTION 2  the focal group stops losing")
    print(f"  focal mean-space {f_mean:+.4f} -> pooled {f_pool:+.4f}   "
          f"-> {'CONFIRMED' if f_pool > f_mean else 'FALSIFIED'}"
          f"{' (still negative)' if f_pool < 0 else ''}")

    # ---- the selective blend, against the selection null ---------------------------
    rng = np.random.default_rng(0)
    print(f"\nSELECTIVE BLEND -- retrieval votes only on the findings where it gained here")
    out = {"baseline_macro": macro_base, "k": K, "T": T, "weight": W,
           "focal": FOCAL, "prediction_1_focal_gain": g_focal,
           "prediction_1_diffuse_gain": g_diff, "prediction_2_focal_before": f_mean,
           "prediction_2_focal_after": f_pool, "per_finding": {}, "selective": {}}
    for j, name in enumerate(LAB):
        out["per_finding"][name] = {
            "focal": bool(focal[j]),
            "retrieval_auc_mean": float(res["mean"]["auc"][j]),
            "retrieval_auc_pool": float(res["pool"]["auc"][j]),
            "delta_mean": float(res["mean"]["delta"][j]),
            "delta_pool": float(res["pool"]["delta"][j]),
            "spearman_mean": float(res["mean"]["rho"][j]),
            "spearman_pool": float(res["pool"]["rho"][j])}
    for tag in ("mean", "pool"):
        s = res[tag]["score"]
        pick = res[tag]["delta"] > 0
        sel = float(np.nanmean(np.where(pick, ar.blend_auc(Yq, base_rank, s, W), a_base))) - macro_base
        null = []
        for _ in range(N_NULL):
            sp = s[rng.permutation(len(Yq))]
            dnull = ar.blend_auc(Yq, base_rank, sp, W) - a_base
            null.append(float(np.nanmean(np.where(dnull > 0, a_base + dnull, a_base))) - macro_base)
        null = np.array(null)
        p = float((null >= sel).mean())
        print(f"  {tag:<5} keeps {int(pick.sum())}/12 findings -> {sel:+.4f} macro | "
              f"choosing the winners from PURE NOISE gives {null.mean():+.4f} "
              f"(95th pct {np.percentile(null, 95):+.4f}) | p = {p:.3f}")
        out["selective"][tag] = {"n_kept": int(pick.sum()),
                                 "kept": [LAB[j] for j in range(12) if pick[j]],
                                 "macro_delta": sel, "null_mean": float(null.mean()),
                                 "null_p95": float(np.percentile(null, 95)), "p_value": p}

    best = max(out["selective"].values(), key=lambda v: v["macro_delta"])
    if best["macro_delta"] >= ar.ADOPT and best["p_value"] < 0.05:
        v = ("WORTH A CONFIRMATION RUN -- the selective blend beats the selection null. It is "
             "still chosen on the same 870 studies, so it must be re-measured on another fold "
             "before it goes near a submission")
    elif g_focal > g_diff and f_pool > f_mean:
        v = ("MECHANISM CONFIRMED, GAIN TOO SMALL -- pooled space rescues the focal findings "
             "exactly as predicted, but not by enough to clear the threshold")
    else:
        v = ("REJECT -- retrieval in this representation does not beat arm20's own head, in "
             "either space, and the diffuse/focal explanation did not survive its test")
    print(f"\nVERDICT: {v}")
    out["verdict"] = v
    json.dump(out, io.open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["kaggle/arm27/output/arm27_embeddings.npz",
                            "kaggle/arm20/output/arm20_f0_noaug_preds.npz",
                            "kaggle/arm27/output/arm27_pool.json"]))
