#!/usr/bin/env python3
"""E9: build the teacher-filtered train set for train_jersey.py --train_keep.

Keeps (train split only, valid untouched):
  positives  numbered-track crops where the MMOCR teacher read exactly the ground-truth number
             (so the digits are legible in that crop)
  negatives  every crop of a track without a number
Drops the numbered crops MMOCR could not read: mostly backs turned away, blur or occlusion. On those
the only way to get the label right is to recognise the player, which is the shortcut D1-D3 learned.

    python tools/make_e9_keep.py   # after: d0_mmocr.py --split train --out runs/jerseyD/E9_teacher
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jersey_lib import load_labels  # noqa: E402

pack, teach = "data/packs/D", Path("runs/jerseyD/E9_teacher")
df = load_labels(pack, "train")
fr = pd.read_csv(teach / "train_frames.csv", dtype={"mmocr_number": str})
df = df.merge(fr[["file", "mmocr_number", "mmocr_conf"]], on="file", how="left")
read = pd.to_numeric(df.mmocr_number, errors="coerce")
pos = (df.has == 1) & (read == pd.to_numeric(df.jersey, errors="coerce"))
neg = df.has == 0
keep = df[pos | neg]
keep[["file", "track", "jersey", "mmocr_conf"]].to_csv(teach / "train_keep.csv", index=False)
print(f"positives {pos.sum()} crops from {df[pos].track.nunique()} tracks "
      f"({df[pos].jersey.nunique()} distinct numbers); negatives {neg.sum()} crops; "
      f"dropped {len(df) - len(keep)} numbered crops MMOCR could not read -> {teach}/train_keep.csv")
