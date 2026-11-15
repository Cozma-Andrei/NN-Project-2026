#!/usr/bin/env python3
"""Task D baseline D0: today's pipeline jersey reader (MMOCR DBNet + SAR on the whole box crop)
plus today's track vote (tracklab select_highest_voted_att), on the same pack-D valid crops and
ground-truth tracks as D1-D3. Writes runs/jerseyD/D0_mmocr/valid_tracks.csv, then score it with
    python tools/eval_jersey.py --tracks_csv runs/jerseyD/D0_mmocr/valid_tracks.csv
"""
import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jersey_lib import load_labels, track_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/packs/D")
    ap.add_argument("--out", default="runs/jerseyD/D0_mmocr")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="valid", choices=["valid", "train"],
                    help="train = E9 teacher pass: MMOCR on train crops of numbered tracks only (no scoring)")
    ap.add_argument("--max_tracks", type=int, default=0, help="debug: only the first N valid tracks")
    ap.add_argument("--num_shards", type=int, default=1, help="split tracks into N shards run as N processes")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="merge valid_frames_shard*.csv and score tracks")
    ap.add_argument("--gpu_mem_frac", type=float, default=0.0,
                    help="cap GPU memory of this process (MMOCR otherwise grows to fill the card)")
    a = ap.parse_args()
    if a.gpu_mem_frac > 0 and a.device.startswith("cuda"):
        import torch
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac, torch.device(a.device).index or 0)

    from tracklab.utils.attribute_voting import select_highest_voted_att

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df_tr, df = load_labels(a.pack, "train"), load_labels(a.pack, a.split)
    if a.split == "train":
        df = df[df.has == 1].reset_index(drop=True)  # negatives come from no-number tracks, not from MMOCR
    if a.max_tracks:
        df = df[df.track.isin(df.track.drop_duplicates().iloc[:a.max_tracks])].reset_index(drop=True)
    if a.merge:
        parts = sorted(out.glob(f"{a.split}_frames_shard*.csv"))
        fr = pd.concat([pd.read_csv(p, dtype={"mmocr_number": str}) for p in parts])
        assert len(fr) == len(df) and set(fr.file) == set(df.file), f"shards incomplete: {len(fr)}/{len(df)} crops"
        fr["mmocr_number"] = fr.mmocr_number.where(fr.mmocr_number.notna(), None)
        df = df.merge(fr[["file", "mmocr_number", "mmocr_conf"]], on="file", how="left")
        df[["file", "track", "mmocr_number", "mmocr_conf"]].to_csv(out / f"{a.split}_frames.csv", index=False)
        return score(df, df_tr, out, select_highest_voted_att)
    if a.num_shards > 1:
        tracks = df.track.drop_duplicates().sort_values().to_numpy()
        df = df[df.track.isin(tracks[a.shard::a.num_shards])].reset_index(drop=True)

    from sn_gamestate.jersey.mmocr_api import MMOCR
    ocr = MMOCR(batch_size=a.batch, device=a.device)

    import torch
    n_split = 0

    def infer(crops):
        """MMOCR on a batch; on CUDA OOM free the cache and retry in two halves (rare: MMOCR pads a
        batch to its largest crop, so a few tall crops can spike memory). Counted in n_split."""
        nonlocal n_split
        try:
            return ocr.run_mmocr_inference(crops)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(crops) == 1:
                raise
            n_split += 1
            h = len(crops) // 2
            return infer(crops[:h]) + infer(crops[h:])

    jn, conf = [], []
    for s in range(0, len(df), a.batch):
        # pack-D crops are the full-resolution box crops the pipeline module also reads (RGB arrays)
        crops = [np.asarray(Image.open(Path(a.pack) / f).convert("RGB")) for f in df.file.iloc[s:s + a.batch]]
        for pred in infer(crops):
            j, c = ocr.extract_jersey_numbers_from_ocr(pred)
            jn.append(j); conf.append(float(c))
        if (s // a.batch) % 5 == 0:
            print(f"{time.strftime('%H:%M:%S')} {s + len(crops)}/{len(df)} crops, OOM splits so far {n_split}", flush=True)
    df["mmocr_number"], df["mmocr_conf"] = jn, conf
    if a.num_shards > 1:
        df[["file", "track", "mmocr_number", "mmocr_conf"]].to_csv(out / f"{a.split}_frames_shard{a.shard}.csv", index=False)
        print(f"shard {a.shard} done: {len(df)} crops")
        return
    df[["file", "track", "mmocr_number", "mmocr_conf"]].to_csv(out / f"{a.split}_frames.csv", index=False)
    if a.split == "train":
        print(f"teacher pass done: {len(df)} crops -> {out}/train_frames.csv"); return
    score(df, df_tr, out, select_highest_voted_att)


def score(df, df_tr, out, select_highest_voted_att):
    preds = {}
    for tr, g in df.groupby("track"):
        v = select_highest_voted_att(list(g.mmocr_number), list(g.mmocr_conf))  # exactly today's vote
        preds[tr] = int(v) if v not in (None, "") and str(v).isdigit() else -1
    train_numbers = sorted({int(j) for j in df_tr.jersey if j})
    t = track_table(df, preds, train_numbers)
    t.to_csv(out / "valid_tracks.csv", index=False)
    print(f"D0 track accuracy: {t.correct.mean() * 100:.2f}%  -> {out}/valid_tracks.csv")


if __name__ == "__main__":
    main()
