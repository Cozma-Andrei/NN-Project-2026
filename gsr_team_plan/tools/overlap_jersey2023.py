#!/usr/bin/env python3
"""D9 overlap check: which SoccerNet Jersey-2023 tracklets show the same players/clips as our GSR splits?

Both datasets are cut from the SoccerNet-Tracking videos and share no IDs, so we match by content:
each tracklet / track = mean DINOv2 embedding of up to K crops (upper 60% of the box, as in training).
For each Jersey-2023 tracklet we take the most similar GSR track (pack D train + valid) and record its
cosine similarity and whether the jersey numbers agree. Same clip and same player gives near-identical
crops, so a very high similarity together with the same number means overlap.

    python tools/overlap_jersey2023.py      # -> runs/jerseyD/D9_overlap/{matches.csv,summary.json,sim_hist.png}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jersey_lib import UPPER_BODY, load_labels  # noqa: E402

K = 8
EXT = Path("data/external/jersey2023")
OUT = Path("runs/jerseyD/D9_overlap")


def crop_upper(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        return im.crop((0, 0, w, max(1, int(round(h * UPPER_BODY)))))


@torch.no_grad()
def embed_groups(groups, model, tf, device, bs=256):
    """groups: list of (key, [paths]) -> (keys, L2-normalised mean embeddings)."""
    keys, flat, owner = [], [], []
    for gi, (k, paths) in enumerate(groups):
        keys.append(k)
        idx = np.linspace(0, len(paths) - 1, min(K, len(paths))).astype(int)
        for i in idx:
            flat.append(paths[i]); owner.append(gi)
    feats = []
    for s in range(0, len(flat), bs):
        x = torch.stack([tf(crop_upper(p)) for p in flat[s:s + bs]]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            feats.append(torch.nn.functional.normalize(model(x).float(), dim=1).cpu())
        if (s // bs) % 20 == 0:
            print(f"  embedded {s + len(x)}/{len(flat)} crops", flush=True)
    f = torch.cat(feats)
    owner = torch.tensor(owner)
    m = torch.zeros(len(keys), f.shape[1]).index_add_(0, owner, f)
    return keys, torch.nn.functional.normalize(m, dim=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**{**cfg, "input_size": (3, 224, 224), "crop_pct": 1.0})

    # Jersey-2023 tracklets (labelled parts only)
    jt = []
    for part in ("train", "test"):
        gt = json.loads((EXT / part / part / f"{part}_gt.json").read_text())
        for tid, num in gt.items():
            paths = sorted((EXT / part / part / "images" / tid).glob("*.jpg"))
            if paths:
                jt.append((f"{part}/{tid}", paths, int(num)))
    print(f"Jersey-2023: {len(jt)} labelled tracklets")

    # our tracks (pack D: train + valid)
    gs = []
    for split in ("train", "valid"):
        df = load_labels("data/packs/D", split)
        for tr, g in df.groupby("track"):
            gs.append((tr, [Path("data/packs/D") / f for f in g.file], split, int(g.jersey.iat[0]) if g.jersey.iat[0] else -1,
                       int(g.game_id.iat[0])))
    print(f"GSR pack D: {len(gs)} tracks")

    jk, je = embed_groups([(k, p) for k, p, _ in jt], model, tf, device)
    gk, ge = embed_groups([(k, p) for k, p, *_ in gs], model, tf, device)
    sim = je @ ge.T
    best_sim, best = sim.max(1)
    rows = []
    for i, (k, _, num) in enumerate(jt):
        g = gs[int(best[i])]
        # best match among GSR tracks with the SAME number (tells us if a same-number twin exists)
        same = [j for j, t in enumerate(gs) if t[3] == num]
        s_same = float(sim[i, same].max()) if same else float("nan")
        g_same = gs[same[int(sim[i, same].argmax())]] if same else None
        rows.append({"jersey_tracklet": k, "jersey_number": num, "best_track": g[0], "best_split": g[2], "best_game": g[4],
                     "best_number": g[3], "best_sim": float(best_sim[i]), "same_number_sim": s_same,
                     "same_number_split": g_same[2] if g_same else "", "same_number_game": g_same[4] if g_same else -1})
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "matches.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    agree = m.best_number == m.jersey_number
    ax.hist(m.best_sim[agree], bins=60, alpha=.7, label="nearest track has the same number")
    ax.hist(m.best_sim[~agree], bins=60, alpha=.7, label="nearest track has another number")
    ax.set(xlabel="cosine similarity to nearest GSR track", ylabel="Jersey-2023 tracklets", title="D9 overlap check")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "sim_hist.png", dpi=120)

    summ = {"n_jersey_tracklets": len(m), "n_gsr_tracks": len(gs),
            "nearest_has_same_number_%": round(100 * agree.mean(), 1)}
    for thr in (0.8, 0.85, 0.9, 0.95):
        hit = m[(m.best_sim >= thr) & agree]
        summ[f"sim>={thr}_and_same_number"] = {"n": len(hit), **hit.best_split.value_counts().to_dict(),
                                              "games": hit.best_game.value_counts().to_dict()}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
