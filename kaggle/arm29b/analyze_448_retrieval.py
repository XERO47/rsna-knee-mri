"""arm29b: a third retrieval geometry, judged against a blend that already has the other two.

    R20  arm20 pooled space, 336 px, 20 epochs   (adopted; +0.0038 replicated over five folds)
    R21  arm21 crop space, ROI crops             (kept; +0.0006 over B+R20, rho(R21,R20)=0.615)
    R23  arm23 pooled space, 448 px              (this arm)
    B    the 7-member fold-0 blend

PRIMARY, pre-registered:   delta = AUC(B + R20 + R21 + R23) - AUC(B + R20 + R21)
NOVELTY:                   rho(R23, R20) and rho(R23, R21), not rho against arm23's own head
ADOPTION:                  >= +0.0010 strong | +0.0005-0.0010 keep if consistent | < +0.0005 no

LOW PRIOR, recorded before the run: arm23's own experiment rejected 448 px at +0.0011 OOF and
the mechanism was falsified there -- the focal findings (Fracture, Contusion) got WORSE at
higher resolution. k=32 and T=0.07 are frozen from arm27 and were not re-tuned for this arm.

The interesting alternative outcome: R23 is a 336-vs-448 change of the SAME architecture on the
SAME corpus geometry, so if its retrieval is highly correlated with R20 that tells us the
decorrelation in arm29a came from arm21 being a genuinely different KIND of model (crops, not
whole-knee windows) rather than from retrieval geometries being diverse in general.
"""
import importlib.util
import io
import json
import sys

import numpy as np
from scipy.stats import spearmanr

spec = importlib.util.spec_from_file_location(
    "a29a", "D:/knee rsna/kaggle/arm29a/analyze_crop_retrieval.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

LAB, CROP_LAB, GLOBALS, CROP = a.LAB, a.CROP_LAB, a.GLOBALS, a.CROP
K, T, W = a.K, a.T, a.W
STRONG, KEEP = a.STRONG, a.KEEP
N_BOOT = 2000


def main(crop_path, emb20_path, emb23_path, out_path):
    load = lambda p: {k: v for k, v in np.load(p, allow_pickle=True).items()}
    c, e20, e23 = load(crop_path), load(emb20_path), load(emb23_path)

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
    crop_cols = [LAB.index(x) for x in CROP_LAB]
    members.append({"name": CROP[0], "pred": cp, "columns": crop_cols, "weight": 1.0})

    def rows_of(e):
        o = {str(u): i for i, u in enumerate(e["ids"])}
        q = [o[u] for u in ids]
        isr = e["is_reference"] == 1
        assert not isr[q].any(), "a query study is inside a reference database"
        return q, isr

    q20, r20 = rows_of(e20)
    R20, nn20 = a.knn(e20["pool"][q20], e20["pool"][r20], e20["y"][r20], list(range(12)), K, T)
    q21, r21 = rows_of(c)
    R21, nn21 = a.knn(c["pool"][q21], c["pool"][r21], c["y"][r21], crop_cols, K, T,
                      valid_q=c["valid"][q21], valid_r=c["valid"][r21])
    q23, r23 = rows_of(e23)
    R23, nn23 = a.knn(e23["pool"][q23], e23["pool"][r23], e23["y"][r23], list(range(12)), K, T)
    head23 = e23["prob"][q23]

    m20 = {"name": "R20", "pred": R20, "columns": list(range(12)), "weight": W}
    m21 = {"name": "R21", "pred": R21, "columns": crop_cols, "weight": W}
    m23 = {"name": "R23", "pred": R23, "columns": list(range(12)), "weight": W}

    a20, a23 = a.aucs(Y, R20), a.aucs(Y, R23)
    print(f"R23 solo: {np.nanmean(a23):.4f} macro  (R20 solo {np.nanmean(a20):.4f}, "
          f"arm23's own head {np.nanmean(a.aucs(Y, head23)):.4f})\n")

    ladder, combos = {}, [("B", members), ("B+R20", members + [m20]),
                          ("B+R20+R21", members + [m20, m21]),
                          ("B+R20+R21+R23", members + [m20, m21, m23]),
                          ("B+R20+R23", members + [m20, m23])]
    print(f"{'configuration':<18} {'macro(12)':>10}")
    for tag, mem in combos:
        au = a.aucs(Y, a.rank_blend(mem))
        ladder[tag] = {"macro12": float(np.nanmean(au)), "aucs": au}
        print(f"{tag:<18} {ladder[tag]['macro12']:>10.4f}")

    base, new = ladder["B+R20+R21"]["aucs"], ladder["B+R20+R21+R23"]["aucs"]
    d12 = float(np.nanmean(new) - np.nanmean(base))
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(ids), len(ids))
        bm = [{**m, "pred": m["pred"][i]} for m in members + [m20, m21]]
        boot.append(np.nanmean(a.aucs(Y[i], a.rank_blend(bm + [{**m23, "pred": R23[i]}])))
                    - np.nanmean(a.aucs(Y[i], a.rank_blend(bm))))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nPRIMARY  AUC(B+R20+R21+R23) - AUC(B+R20+R21) = {d12:+.4f}")
    print(f"         95% CI {lo:+.4f}..{hi:+.4f}, P(>0) {(boot > 0).mean():.2f}")
    print(f"secondary AUC(B+R23) - AUC(B) would read "
          f"{ladder['B+R20+R23']['macro12'] - ladder['B+R20']['macro12']:+.4f} against B+R20")

    print(f"\n{'finding':<18} {'R23':>7} {'R20':>7} {'d(12)':>8}  "
          f"{'rho(R23,head23)':>16} {'rho(R23,R20)':>13} {'rho(R23,R21)':>13}")
    per, rr20 = {}, []
    for j in range(12):
        ok = np.isfinite(R23[:, j])
        rh = float(spearmanr(R23[ok, j], head23[ok, j]).statistic)
        r0 = float(spearmanr(R23[ok, j], R20[ok, j]).statistic)
        ok2 = ok & np.isfinite(R21[:, j])
        r1 = float(spearmanr(R23[ok2, j], R21[ok2, j]).statistic) if ok2.sum() > 30 else float("nan")
        rr20.append(r0)
        per[LAB[j]] = {"r23_auc": float(a23[j]), "r20_auc": float(a20[j]),
                       "delta12": float(new[j] - base[j]), "rho_vs_head23": rh,
                       "rho_vs_R20": r0, "rho_vs_R21": r1}
        print(f"{LAB[j]:<18} {a23[j]:>7.4f} {a20[j]:>7.4f} {new[j] - base[j]:>+8.4f}  "
              f"{rh:>16.3f} {r0:>13.3f} " + (f"{r1:>13.3f}" if r1 == r1 else f"{'—':>13}"))
    mr20 = float(np.mean(rr20))
    print(f"\nmean rho(R23, R20) = {mr20:.3f}   vs   mean rho(R21, R20) = 0.615 (arm29a)")
    print("Both are retrieval in a pooled space; R23 differs from R20 only in input resolution,")
    print("R21 differs in what it looks at. Whichever correlation is lower says where the")
    print("decorrelation actually comes from.")

    npos = int(np.nansum(new > base))
    if d12 >= STRONG:
        v = "STRONG ADOPTION -- a third geometry still adds; deploy and keep extracting"
    elif d12 >= KEEP and npos >= 8:
        v = f"KEEP -- {d12:+.4f} with {npos}/12 findings positive"
    elif d12 >= KEEP:
        v = f"BORDERLINE AND INCONSISTENT -- {d12:+.4f}, only {npos}/12 positive"
    else:
        v = ("DO NOT DEPLOY -- below +0.0005 once R20 and R21 are already in the blend. "
             "The retrieval axis is saturating, as the low prior expected")
    print(f"\nVERDICT: {v}")
    json.dump({"k": K, "temperature": T,
               "ladder": {k: v2["macro12"] for k, v2 in ladder.items()},
               "primary_delta": d12, "primary_ci": [float(lo), float(hi)],
               "primary_p_positive": float((boot > 0).mean()),
               "r23_solo_macro": float(np.nanmean(a23)),
               "mean_rho_R23_R20": mr20, "per_finding": per,
               "findings_positive": npos, "verdict": v},
              io.open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["kaggle/arm29a/output/arm29a_crop_embeddings.npz",
                            "kaggle/arm27/output/arm27_embeddings.npz",
                            "kaggle/arm29b/output/arm29b_embeddings.npz",
                            "kaggle/arm29b/output/arm29b_retrieval.json"]))
