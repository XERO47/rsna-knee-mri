# -*- coding: utf-8 -*-
"""Render the 58 gold studies to viewable PNG montages and zip one archive per study.

Purpose: a blind reliability test of a vision-LLM against the OFFICIAL image-derived labels.
For each gold study, render per-plane montages (fluid-sensitive preferred, plus one anatomical
sagittal) of evenly-sampled, intensity-normalised slices. The zips contain IMAGES ONLY -- the gold
labels are NOT included, so the read is blind; we compare to gold_test.csv afterwards.

Output in /kaggle/working:
  gold_zips/study_<NN>_<shortuid>.zip   (per study: sag_fs.png, cor_fs.png, ax_fs.png, sag_anat.png)
  labeling_template.csv                 (StudyInstanceUID + 12 blank finding columns)
  RUBRIC.txt                            (the 12 RSNA definitions to paste into the VLM prompt)
  index.csv                             (zip -> StudyInstanceUID map)
"""
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WORK = Path("/kaggle/working")
N_TILES = 12          # slices sampled per series
TILE = 320            # px per tile
GRID = (3, 4)         # rows x cols for the montage

RUBRIC = """Read the knee MRI IMAGES only (no report). For each of the 12 findings output 0 or 1,
using the RSNA competition definitions. Borderline/ambiguous -> 0 (negative).
  ACL: high-grade partial (>50% fibers) or complete tear
  MCL: high-grade partial or complete acute tear
  Medial Meniscus / Lateral Meniscus: definite surface-reaching tear on >=2 images
  Medial OA / Lateral OA / PF OA: ~>=1 cm area of >50% cartilage loss
  Effusion: moderate/large effusion
  Synovitis: definite synovial inflammation/thickening
  Baker's: moderate/large Baker cyst
  Contusion: impact-related marrow edema without a fracture line
  Fracture: acute cortical break / fracture line
Output exactly: StudyInstanceUID then 12 values in the order ACL,MCL,Medial Meniscus,
Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture."""


def find(name):
    for p in Path("/kaggle/input").rglob(name):
        return p
    raise FileNotFoundError(name)


def read_norm(path):
    d = pydicom.dcmread(str(path))
    arr = apply_modality_lut(d.pixel_array.astype(np.float32), d)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def montage(series_dir):
    import cv2
    files = []
    for f in Path(series_dir).glob("*.dcm"):
        try:
            inst = int(getattr(pydicom.dcmread(str(f), stop_before_pixels=True), "InstanceNumber", 0))
        except Exception:
            inst = 0
        files.append((inst, f))
    files.sort(key=lambda x: x[0])
    files = [f for _, f in files]
    if not files:
        return None
    idx = np.linspace(0, len(files) - 1, min(N_TILES, len(files))).round().astype(int)
    tiles = []
    for i in idx:
        try:
            img = cv2.resize(read_norm(files[i]), (TILE, TILE), interpolation=cv2.INTER_AREA)
            tiles.append(img)
        except Exception:
            continue
    if not tiles:
        return None
    while len(tiles) < GRID[0] * GRID[1]:
        tiles.append(np.zeros((TILE, TILE), np.uint8))
    rows = [np.hstack(tiles[r * GRID[1]:(r + 1) * GRID[1]]) for r in range(GRID[0])]
    grid = np.vstack(rows)
    ok, buf = cv2.imencode(".png", grid)
    return buf.tobytes() if ok else None


def pick(rows, plane, fluid):
    cand = [r for r in rows if r["Anatomical_Plane"] == plane
            and int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
    cand = cand or [r for r in rows if r["Anatomical_Plane"] == plane]
    return sorted(cand, key=lambda r: str(r["SeriesInstanceUID"]))[0] if cand else None


def main():
    root = find("train_series.csv").parent
    gold = pd.read_csv(find("gold_test.csv")); gold["StudyInstanceUID"] = gold.StudyInstanceUID.astype(str)
    series = pd.read_csv(root / "train_series.csv"); series["StudyInstanceUID"] = series.StudyInstanceUID.astype(str)
    sr = root / "train_series"
    outdir = WORK / "gold_zips"; outdir.mkdir(parents=True, exist_ok=True)
    (WORK / "RUBRIC.txt").write_text(RUBRIC, encoding="utf-8")

    wanted = [("sag_fs", "Sagittal", 1), ("cor_fs", "Coronal", 1),
              ("ax_fs", "Axial", 1), ("sag_anat", "Sagittal", 0)]
    index = []
    for i, uid in enumerate(gold.StudyInstanceUID):
        rows = series[series.StudyInstanceUID == uid].to_dict("records")
        pngs = {}
        for tag, plane, fluid in wanted:
            s = pick(rows, plane, fluid)
            if s is None:
                continue
            png = montage(sr / uid / str(s["SeriesInstanceUID"]))
            if png:
                pngs[f"{tag}.png"] = png
        zpath = outdir / f"study_{i:02d}_{uid[-8:]}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in pngs.items():
                z.writestr(name, data)
        index.append({"zip": zpath.name, "StudyInstanceUID": uid, "n_png": len(pngs)})
        if (i + 1) % 10 == 0:
            print(f"{i+1}/58 rendered", flush=True)

    pd.DataFrame(index).to_csv(WORK / "index.csv", index=False)
    tmpl = pd.DataFrame({"StudyInstanceUID": gold.StudyInstanceUID})
    for c in LAB:
        tmpl[c] = ""
    tmpl.to_csv(WORK / "labeling_template.csv", index=False)
    print(f"done: {len(index)} zips, mean png/study {np.mean([r['n_png'] for r in index]):.1f}", flush=True)
    print("GOLD_RENDER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
