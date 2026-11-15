#!/usr/bin/env python3
"""Task D: evaluate jersey predictions per ground-truth track on pack D 'valid', with the
error-analysis breakdowns required by the plan, and save failure strips.

    python tools/eval_jersey.py --ckpt runs/jerseyD/D1_r18_digits/best.pt
    python tools/eval_jersey.py --tracks_csv runs/jerseyD/D0_mmocr/valid_tracks.csv   # D0 or any saved table

Track rule (fixed, in tools/jersey_lib.py): keep frames with p(has number) > 0.5; none kept -> "none";
else the number with the largest summed probability over kept frames.
Outputs next to the checkpoint / csv: eval_metrics.json, eval_breakdowns.csv, failures/*.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jersey_lib import JerseyCrops, JerseyNet, aggregate_tracks, build_transforms, load_labels, track_metrics, track_table  # noqa: E402


def breakdowns(t):
    """Accuracy by the dataset properties named in the plan (higher is better)."""
    rows = []

    def add(group, name, sub):
        rows.append({"group": group, "bin": name, "n_tracks": len(sub),
                     "track_acc": round(float(sub.correct.mean() * 100), 2) if len(sub) else np.nan})
    add("all", "all", t)
    add("label", "has a number", t[t.true >= 0])
    add("label", "no number", t[t.true < 0])
    if "unseen_in_train" in t:
        add("label", "number unseen in train", t[t.unseen_in_train])
        add("label", "number seen in train", t[(t.true >= 0) & ~t.unseen_in_train])
    for lo, hi in ((1, 10), (11, 50), (51, 10**9)):
        add("crops per track (fragmentation)", f"{lo}-{hi if hi < 10**9 else 'max'}", t[(t.n_crops >= lo) & (t.n_crops <= hi)])
    for lo, hi in ((0, 80), (80, 120), (120, 10**9)):
        add("median box height px (1080p)", f"{lo}-{hi if hi < 10**9 else 'max'}", t[(t.median_h >= lo) & (t.median_h < hi)])
    for g, sub in t.groupby("game_id"):
        add("match (game_id)", str(g), sub)
    # error types, as shares of all tracks
    err = t[~t.correct]
    for name, m in (("number read as none", (err.true >= 0) & (err.pred < 0)),
                    ("none read as a number", (err.true < 0) & (err.pred >= 0)),
                    ("wrong number", (err.true >= 0) & (err.pred >= 0))):
        rows.append({"group": "error type (% of all tracks)", "bin": name, "n_tracks": int(m.sum()),
                     "track_acc": round(float(m.sum() / len(t) * 100), 2)})
    return pd.DataFrame(rows)


def save_failures(t, df, pack, out_dir, n=5, per_strip=8, seed=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fails = t[~t.correct]
    picks = []
    for kind in (fails[(fails.true >= 0) & (fails.pred >= 0)], fails[(fails.true >= 0) & (fails.pred < 0)],
                 fails[(fails.true < 0) & (fails.pred >= 0)]):
        if len(kind):
            picks += list(kind.sample(min(len(kind), max(1, n // 3 + 1)), random_state=int(rng.integers(1e9))).track)
    for tr in picks[:n]:
        rows = df[df.track == tr]
        rows = rows.iloc[np.linspace(0, len(rows) - 1, min(per_strip, len(rows))).astype(int)]
        ims = [Image.open(Path(pack) / f).convert("RGB") for f in rows.file]
        h = 160
        ims = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in ims]
        strip = Image.new("RGB", (sum(im.width for im in ims) + 4 * len(ims), h), "white")
        x = 0
        for im in ims:
            strip.paste(im, (x, 0)); x += im.width + 4
        r = t[t.track == tr].iloc[0]
        lab = lambda v: "none" if v < 0 else str(v)  # noqa: E731
        strip.save(out_dir / f"{tr}_true-{lab(r.true)}_pred-{lab(r.pred)}.jpg", quality=90)
    return picks[:n]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt")
    g.add_argument("--tracks_csv")
    ap.add_argument("--pack", default="data/packs/D")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    df_tr, df_va = load_labels(a.pack, "train"), load_labels(a.pack, "valid")
    train_numbers = sorted({int(j) for j in df_tr.jersey if j})
    if a.ckpt:
        out = Path(a.ckpt).parent
        ck = torch.load(a.ckpt, map_location="cpu")
        model = JerseyNet(ck["backbone"], ck["head"], pretrained=False, adv_classes=ck.get("adv_classes", 0))
        model.load_state_dict(ck["model"])
        device = torch.device(a.device if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        mean, std = model.data_mean_std()
        dl = DataLoader(JerseyCrops(a.pack, df_va, build_transforms(mean, std, False)), batch_size=256,
                        num_workers=a.workers, pin_memory=True)
        P_has, P_num = [], []
        with torch.no_grad():
            for x, _ in dl:
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    ph, pn = model.frame_probs(model(x.to(device)))
                P_has.append(ph.cpu()); P_num.append(pn.cpu())
        p_has, p_num = torch.cat(P_has).numpy(), torch.cat(P_num).numpy()
        t = track_table(df_va, aggregate_tracks(df_va, p_has, p_num), train_numbers)
        t.to_csv(out / "eval_valid_tracks.csv", index=False)
    else:
        out = Path(a.tracks_csv).parent
        t = pd.read_csv(a.tracks_csv)
        t["correct"] = t.true == t.pred
        if "unseen_in_train" not in t:
            t["unseen_in_train"] = (t.true >= 0) & ~t.true.isin(train_numbers)

    m = track_metrics(t)
    (out / "eval_metrics.json").write_text(json.dumps(m, indent=2))
    b = breakdowns(t)
    b.to_csv(out / "eval_breakdowns.csv", index=False)
    save_failures(t, df_va, a.pack, out / "failures")
    print(json.dumps(m, indent=2))
    print(b.to_string(index=False))


if __name__ == "__main__":
    main()
