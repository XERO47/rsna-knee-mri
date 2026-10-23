"""arm31 Phase 0: does a second legitimate acquisition of the same knee carry independent evidence?

Pre-registered reading, fixed before the numbers existed:

    A weaker than P alone            EXPECTED, not a rejection. An arbitrary first candidate
                                     beats an arbitrary second only by chance.
    rho(A, P) below ~0.90            the concept is alive
    strongest ensemble + A positive  the concept is worth training on

Decision thresholds, revised after the retrieval deployment failure (EXPERIMENTS.md 2b):

    <  +0.0020   kill for deployment
    +0.0020 .. +0.0040   investigate / replicate
    >= +0.0040   serious submission candidate

and the gain must come from several findings, not one noisy class.

THE COVERAGE CORRECTION, applied throughout. Only ~44% of the fold-0 query studies HAVE an
alternate assignment; for the rest A is byte-identical to P. So:

  * a correlation over all 870 is inflated toward 1 by rows that cannot differ -- it measures
    coverage, not agreement. The mechanistic number is rho on the CHANGED subset.
  * a blend delta over all 870 is diluted by the unchanged rows -- but that IS the deployable
    quantity, because deployment would also only change those studies.

Both are reported for every statistic. Quoting only the first understates the mechanism;
quoting only the second overstates what a submission would get.
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
CROP = "kaggle/arm21/output/arm21_f0_crop_preds.npz"
KILL, INVESTIGATE, SERIOUS = 0.0020, 0.0020, 0.0040
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


def aucs(Y, P, sub=None):
    out = np.full(12, np.nan)
    for j in range(12):
        ok = np.isfinite(P[:, j]) if sub is None else (np.isfinite(P[:, j]) & sub)
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
        srt = np.take_along_axis(sim, idx, 1)
        w = np.exp((srt - srt.max(1, keepdims=True)) / T)
        w /= w.sum(1, keepdims=True)
        out[okq, col] = (w * (Yref[okr][:, i] > 0.5).astype(np.float64)[idx]).sum(1)
    return out


def main(probe_path, out_path):
    d = {k: v for k, v in np.load(probe_path, allow_pickle=True).items()}
    ids = [str(u) for u in d["oof_ids"]]
    Y, P, A = d["oof_truth"], d["P"], d["A"]
    ch = d["changed"].astype(bool)
    print(f"{len(ids)} fold-0 queries | {int(ch.sum())} ({100 * ch.mean():.1f}%) have an "
          f"alternate acquisition | {int((~ch).sum())} are identical\n")

    a_p, a_a = aucs(Y, P), aucs(Y, A)
    print(f"{'':<20} {'all 870':>18}   {'the 379 that CHANGED':>22}")
    print(f"{'':<20} {'P':>8} {'A':>9}   {'P':>10} {'A':>11}")
    p_ch, a_ch = aucs(Y, P, ch), aucs(Y, A, ch)
    for j, nm in enumerate(LAB):
        print(f"{nm:<20} {a_p[j]:>8.4f} {a_a[j]:>9.4f}   {p_ch[j]:>10.4f} {a_ch[j]:>11.4f}")
    print(f"{'MACRO':<20} {np.nanmean(a_p):>8.4f} {np.nanmean(a_a):>9.4f}   "
          f"{np.nanmean(p_ch):>10.4f} {np.nanmean(a_ch):>11.4f}")

    # the decorrelation measure -- the number the concept lives or dies on
    rho_all = np.array([spearmanr(P[:, j], A[:, j]).statistic for j in range(12)])
    rho_ch = np.array([spearmanr(P[ch, j], A[ch, j]).statistic for j in range(12)])
    print(f"\n{'finding':<20} {'rho(P,A) all':>13} {'rho(P,A) changed':>18}")
    for j, nm in enumerate(LAB):
        print(f"{nm:<20} {rho_all[j]:>13.3f} {rho_ch[j]:>18.3f}")
    print(f"{'MEAN':<20} {rho_all.mean():>13.3f} {rho_ch.mean():>18.3f}")
    print("  the left column is inflated by the 491 studies that cannot differ; the right one "
          "is the mechanistic number")

    # rank-mean of the two views
    two = rank_blend([{"pred": P, "columns": list(range(12)), "weight": 1.0},
                      {"pred": A, "columns": list(range(12)), "weight": 1.0}])
    a_two = aucs(Y, two)
    print(f"\nrankmean(P, A): {np.nanmean(a_two):.4f}  vs P {np.nanmean(a_p):.4f} "
          f"({np.nanmean(a_two) - np.nanmean(a_p):+.4f}), A {np.nanmean(a_a):.4f}")

    # ---- the deployment question: does A add to the strongest ensemble we have? ----
    members = []
    for name, path in GLOBALS:
        q = np.load(path, allow_pickle=True)
        assert [str(u) for u in q["oof_ids"]] == ids, f"{name} is a different fold/order"
        members.append({"name": name, "pred": q["oof_pred"], "columns": list(range(12)),
                        "weight": 1.0})
    q = np.load(CROP, allow_pickle=True)
    cp = np.full((len(ids), 12), np.nan)
    for i, lab in enumerate(CROP_LAB):
        col = q["oof_pred"][:, i].astype(float)
        col[~q["oof_valid"][:, i].astype(bool)] = np.nan
        cp[:, LAB.index(lab)] = col
    crop_cols = [LAB.index(x) for x in CROP_LAB]
    members.append({"name": "crop", "pred": cp, "columns": crop_cols, "weight": 1.0})

    load = lambda p: {k: v for k, v in np.load(p, allow_pickle=True).items()}
    retr = []
    for tag, pth, cols, v in (
            ("R20", "kaggle/arm27/output/arm27_embeddings.npz", list(range(12)), None),
            ("R23", "kaggle/arm29b/output/arm29b_embeddings.npz", list(range(12)), None),
            ("R21", "kaggle/arm29a/output/arm29a_crop_embeddings.npz", crop_cols, "valid")):
        e = load(pth)
        o = {str(u): i for i, u in enumerate(e["ids"])}
        qq = [o[u] for u in ids]
        isr = e["is_reference"] == 1
        assert not isr[qq].any()
        retr.append({"name": tag, "columns": cols, "weight": 1.0,
                     "pred": knn(e["pool"][qq], e["pool"][isr], e["y"][isr], cols, 32, 0.07,
                                 valid_q=e[v][qq] if v else None,
                                 valid_r=e[v][isr] if v else None)})

    rng = np.random.default_rng(0)
    res = {}
    print(f"\n{'baseline':<22} {'macro':>8} {'+A @0.5':>9} {'+A @1.0':>9}   "
          f"{'CI @1.0 (all 870)':>22} {'up':>5}")
    for tag, base in (("7 members", members), ("7 + 3 retrievals", members + retr)):
        b0 = aucs(Y, rank_blend(base))
        row = {"macro": float(np.nanmean(b0))}
        cells = []
        for w in (0.5, 1.0):
            m = {"name": "A", "pred": A, "columns": list(range(12)), "weight": w}
            b1 = aucs(Y, rank_blend(base + [m]))
            row[f"delta_{w}"] = float(np.nanmean(b1) - np.nanmean(b0))
            cells.append(row[f"delta_{w}"])
            if w == 1.0:
                boot = []
                for _ in range(N_BOOT):
                    i = rng.integers(0, len(ids), len(ids))
                    bb = [{**x, "pred": x["pred"][i]} for x in base]
                    boot.append(np.nanmean(aucs(Y[i], rank_blend(bb + [{**m, "pred": A[i]}])))
                                - np.nanmean(aucs(Y[i], rank_blend(bb))))
                boot = np.array(boot)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                row.update(ci=[float(lo), float(hi)], p_pos=float((boot > 0).mean()),
                           n_up=int(np.nansum(b1 > b0)))
        print(f"{tag:<22} {row['macro']:>8.4f} {cells[0]:>+9.4f} {cells[1]:>+9.4f}   "
              f"{row['ci'][0]:>+9.4f}..{row['ci'][1]:<+9.4f} {row['n_up']:>3}/12")
        res[tag] = row

    strong = res["7 + 3 retrievals"]
    best = max(strong["delta_0.5"], strong["delta_1.0"])
    n_up = strong["n_up"]
    rho = float(rho_ch.mean())
    alive = rho < 0.90
    if best >= SERIOUS and n_up >= 6:
        v = (f"SUBMISSION CANDIDATE -- {best:+.4f} on the strongest ensemble from {n_up}/12 "
             f"findings, with rho {rho:.3f} on the changed studies")
    elif best >= INVESTIGATE:
        v = (f"INVESTIGATE / TRAIN arm31 -- {best:+.4f}, rho {rho:.3f}. The second acquisition "
             f"carries independent evidence; randomising it during TRAINING should exploit it "
             f"more than using it as a test-time view")
    elif alive:
        v = (f"CONCEPT ALIVE, TTA INSUFFICIENT -- rho {rho:.3f} on the changed studies says the "
             f"two acquisitions disagree, but simply averaging them adds only {best:+.4f}. "
             f"Training on randomised acquisitions is a different experiment and is NOT "
             f"answered by this")
    else:
        v = (f"CONCEPT DEAD -- rho {rho:.3f} on the studies that actually changed. Swapping to "
             f"an equally eligible series barely changes what the model sees, so there is no "
             f"independent evidence to exploit")
    print(f"\nVERDICT: {v}")
    json.dump({"n_query": len(ids), "n_changed": int(ch.sum()),
               "auc_primary": float(np.nanmean(a_p)), "auc_alternate": float(np.nanmean(a_a)),
               "auc_primary_changed": float(np.nanmean(p_ch)),
               "auc_alternate_changed": float(np.nanmean(a_ch)),
               "rho_all": rho_all.tolist(), "rho_changed": rho_ch.tolist(),
               "rho_all_mean": float(rho_all.mean()), "rho_changed_mean": rho,
               "rankmean_two_views": float(np.nanmean(a_two)),
               "per_finding": {LAB[j]: {"P": float(a_p[j]), "A": float(a_a[j]),
                                        "P_changed": float(p_ch[j]),
                                        "A_changed": float(a_ch[j]),
                                        "rho_changed": float(rho_ch[j])} for j in range(12)},
               "blend": res, "thresholds": {"kill": KILL, "investigate": INVESTIGATE,
                                            "serious": SERIOUS},
               "verdict": v}, io.open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["kaggle/arm31probe/output/arm31probe_preds.npz",
                            "kaggle/arm31probe/output/arm31probe_decision.json"]))
