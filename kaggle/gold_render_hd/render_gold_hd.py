# -*- coding: utf-8 -*-
"""Max-fidelity static re-render of 5 gold studies for the definitive VLM image test.

Every slice, 512 px/tile (3x3 -> 1536 px wide, matching GPT-4V's 512 px vision patch so each slice
fills one patch), labels in a black header strip off the anatomy, laterality from DICOM when present
(never guessed). Same 5 studies [06,08,11,12,18] that contain GPT's focal misses, so the before/after
is clean. If GPT still can't recover the misses at this fidelity with full coverage, it's the model's
ceiling, not the images.
"""
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

STUDY_IDX = list(range(58))   # all 58 gold studies for the full comparison
TILE = 512
HEADER = 46
ROWS, COLS = 3, 3
WORK = Path("/kaggle/working")


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


def laterality(series_dir):
    """Read DICOM Laterality / ImageLaterality from the first slice; '' if absent (never guess)."""
    for f in Path(series_dir).glob("*.dcm"):
        try:
            h = pydicom.dcmread(str(f), stop_before_pixels=True)
            v = str(getattr(h, "Laterality", "") or getattr(h, "ImageLaterality", "") or "").upper()
            return {"L": "LEFT", "R": "RIGHT"}.get(v, v)
        except Exception:
            return ""
    return ""


def ordered_files(series_dir):
    tagged = []
    for f in Path(series_dir).glob("*.dcm"):
        try:
            inst = int(getattr(pydicom.dcmread(str(f), stop_before_pixels=True), "InstanceNumber", 0))
        except Exception:
            inst = 0
        tagged.append((inst, f))
    tagged.sort(key=lambda x: x[0])
    return [f for _, f in tagged]


def tile(img_u8, label):
    img = cv2.cvtColor(cv2.resize(img_u8, (TILE, TILE), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_GRAY2BGR)
    head = np.zeros((HEADER, TILE, 3), np.uint8)
    cv2.putText(head, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([head, img])


def sheets(series_dir, tag, lat):
    files = ordered_files(series_dir)
    if not files:
        return []
    out = []
    blank = np.zeros((TILE + HEADER, TILE, 3), np.uint8)
    for s in range(0, len(files), ROWS * COLS):
        chunk = files[s:s + ROWS * COLS]
        tiles = []
        for k, f in enumerate(chunk):
            try:
                im = read_norm(f)
            except Exception:
                im = np.zeros((TILE, TILE), np.uint8)
            lbl = f"{tag.upper()} #{s + k}" + (f" | {lat}" if lat else "")
            tiles.append(tile(im, lbl))
        while len(tiles) < ROWS * COLS:
            tiles.append(blank)
        grid = np.vstack([np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)])
        ok, buf = cv2.imencode(".png", grid)
        if ok:
            out.append((f"sheet_{s // (ROWS * COLS):02d}_slices_{s}-{s + len(chunk) - 1}.png", buf.tobytes()))
    return out


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
    outdir = WORK / "gold_hd_zips"; outdir.mkdir(parents=True, exist_ok=True)
    all_uids = list(gold.StudyInstanceUID)
    pd.DataFrame({"study_idx": STUDY_IDX,
                  "key": [f"study_{i:02d}" for i in STUDY_IDX],
                  "StudyInstanceUID": [all_uids[i] for i in STUDY_IDX]}
                 ).to_csv(WORK / "index.csv", index=False)
    for i in STUDY_IDX:
        uid = all_uids[i]
        # EVERY series: all planes, fluid-sensitive AND anatomical, duplicates included
        rows = sorted(series[series.StudyInstanceUID == uid].to_dict("records"),
                      key=lambda r: (str(r["Anatomical_Plane"]),
                                     -int(r.get("Fluid_Sensitive", 0) or 0),
                                     str(r["SeriesInstanceUID"])))
        zpath = outdir / f"study_{i:02d}_{uid[-8:]}_HD512.zip"
        n = 0; seen = {}
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for r in rows:
                plane = str(r["Anatomical_Plane"])[:3].lower()
                kind = "fs" if int(r.get("Fluid_Sensitive", 0) or 0) == 1 else "anat"
                base = f"{plane}_{kind}"
                seen[base] = seen.get(base, 0) + 1
                tag = base if seen[base] == 1 else f"{base}{seen[base]}"
                sd = sr / uid / str(r["SeriesInstanceUID"])
                for name, data in sheets(sd, tag.upper(), laterality(sd)):
                    z.writestr(f"{tag}/{name}", data); n += 1
        print(f"study_{i:02d}: {len(rows)} series, {n} sheets @512px", flush=True)
    print("GOLD_HD512_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
