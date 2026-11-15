#!/usr/bin/env python3
"""Task D: train the jersey frame network on pack D (sprint runs D1-D3).

    python tools/train_jersey.py --name D1_r18_digits --backbone resnet18 --head digits
    python tools/train_jersey.py --name D2_vits_digits --backbone vit_small_patch16_224 --head digits
    python tools/train_jersey.py --name D3_r18_cls101 --backbone resnet18 --head cls101
    add --smoke for a 20-batch check (also on CPU with --device cpu)

Fixed recipe (Team rules: do not tune): AdamW lr 1e-4, weight decay 0.05, cosine decay per step,
8 epochs, batch 128, seed 0. Train on pack D 'train' only; 'valid' picks the best epoch by
track accuracy (all tracks). Outputs in runs/jerseyD/<name>/: history.csv, curves.png,
last.pt (every epoch, resumable with --resume), best.pt, best_metrics.json, config.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jersey_lib import JerseyCrops, JerseyNet, aggregate_tracks, build_transforms, load_labels, track_metrics, track_table  # noqa: E402


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, df_valid, train_numbers, device, amp):
    model.eval()
    tot, n, P_has, P_num = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x)
            loss = model.loss(out, y)
        tot += loss.item() * len(x); n += len(x)
        ph, pn = model.frame_probs(out)
        P_has.append(ph.cpu()); P_num.append(pn.cpu())
    p_has, p_num = torch.cat(P_has).numpy(), torch.cat(P_num).numpy()
    t = track_table(df_valid, aggregate_tracks(df_valid, p_has, p_num), train_numbers)
    return tot / n, track_metrics(t), t


def plot(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(e, [h["train_loss"] for h in hist], "o-", label="train")
    ax[0].plot(e, [h["valid_loss"] for h in hist], "o-", label="valid")
    ax[0].set(xlabel="epoch", ylabel="loss (lower is better)", title="Loss"); ax[0].legend()
    for k, lab in (("track_acc_all", "all tracks"), ("track_acc_number", "with number"), ("track_acc_none", "no number")):
        ax[1].plot(e, [h[k] for h in hist], "o-", label=lab)
    ax[1].set(xlabel="epoch", ylabel="track accuracy % (higher is better)", title="Valid track accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--head", choices=["digits", "cls101"], required=True)
    ap.add_argument("--pack", default="data/packs/D")
    ap.add_argument("--out", default="runs/jerseyD")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="20 train batches, valid on ~40 tracks")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--adv", action="store_true",
                    help="D4: gradient-reversal adversary that predicts clip x team side (appearance), lam ramped 0->1 (DANN schedule)")
    ap.add_argument("--train_keep", default="",
                    help="csv with a 'file' column: train only on these train crops (E9 teacher-filtered set)")
    ap.add_argument("--gpu_mem_frac", type=float, default=0.0,
                    help="cap this process's GPU memory (e.g. 0.6 on the GPU that drives the Windows display)")
    a = ap.parse_args()

    seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    if device.type == "cuda" and a.gpu_mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac, device.index or torch.cuda.current_device())
    amp = device.type == "cuda" and not a.no_amp
    out = Path(a.out) / (a.name + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)

    df_tr, df_va = load_labels(a.pack, "train"), load_labels(a.pack, "valid")
    assert set(df_tr.game_id) == {4, 6, 9} and set(df_va.game_id) == {2, 3, 5}, "pack D splits are mixed!"
    train_numbers = sorted({int(j) for j in df_tr.jersey if j})  # from the full train split, before any filtering
    if a.train_keep:
        keep_files = set(pd.read_csv(a.train_keep).file)
        df_tr = df_tr[df_tr.file.isin(keep_files)].reset_index(drop=True)
        assert len(df_tr) == len(keep_files), "train_keep lists files that are not train crops"
    if a.smoke:
        keep = df_va.track.drop_duplicates().iloc[:40]
        df_va = df_va[df_va.track.isin(keep)].reset_index(drop=True)

    if a.adv:
        df_tr["adv"] = pd.factorize(df_tr["clip"] + "_" + df_tr["team"].astype(str))[0]
    n_adv = int(df_tr.adv.max()) + 1 if a.adv else 0
    model = JerseyNet(a.backbone, a.head, adv_classes=n_adv).to(device)
    mean, std = model.data_mean_std()
    dl_tr = DataLoader(JerseyCrops(a.pack, df_tr, build_transforms(mean, std, True)), batch_size=a.batch, shuffle=True,
                       num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=a.workers > 0,
                       generator=torch.Generator().manual_seed(a.seed))
    dl_va = DataLoader(JerseyCrops(a.pack, df_va, build_transforms(mean, std, False)), batch_size=256, shuffle=False,
                       num_workers=a.workers, pin_memory=True)
    steps_per_epoch = 20 if a.smoke else len(dl_tr)
    epochs = 1 if a.smoke else a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    cfg = {**vars(a), "train_crops": len(df_tr), "valid_crops": len(df_va), "steps_per_epoch": steps_per_epoch,
           "train_numbers": train_numbers, "adv_classes": n_adv, "amp": amp, "mean": list(mean), "std": list(std)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    hist, start, best = [], 1, -1.0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); hist, best, start = ck["hist"], ck["best"], ck["epoch"] + 1
        print(f"resumed after epoch {ck['epoch']}")

    for ep in range(start, epochs + 1):
        model.train(); t0, tot, n = time.time(), 0.0, 0
        adv_tot, adv_hit = 0.0, 0
        for i, (x, y) in enumerate(dl_tr):
            if i >= steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if a.adv:  # DANN ramp: lam = 2 / (1 + exp(-10 p)) - 1, p = training progress 0 -> 1
                p = ((ep - 1) * steps_per_epoch + i) / (epochs * steps_per_epoch)
                model.lam = float(2 / (1 + np.exp(-10 * p)) - 1)
            with torch.autocast(device_type=device.type, enabled=amp):
                pred = model(x)
                loss = model.loss(pred, y)
                task_loss = loss.item()  # logged train loss = jersey loss only (comparable with D1-D3)
                if a.adv:
                    adv_loss = torch.nn.functional.cross_entropy(pred["adv"], y[:, 4])
                    adv_tot += adv_loss.item() * len(x); adv_hit += int((pred["adv"].argmax(1) == y[:, 4]).sum())
                    loss = loss + adv_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += task_loss * len(x); n += len(x)
            if a.smoke or i % 100 == 0:
                print(f"ep {ep} it {i}/{steps_per_epoch} loss {loss.item():.4f}", flush=True)
        train_loss, t_train = tot / n, time.time() - t0
        valid_loss, m, tracks = evaluate(model, dl_va, df_va, train_numbers, device, amp)
        rec = {"epoch": ep, "train_loss": train_loss, "valid_loss": valid_loss, **m,
               **({"adv_loss": adv_tot / n, "adv_acc": 100 * adv_hit / n, "adv_chance": 100 / n_adv, "lam_end": model.lam}
                  if a.adv else {}),
               "lr_end": sched.get_last_lr()[0], "train_minutes": round(t_train / 60, 2),
               "total_minutes": round((time.time() - t0) / 60, 2)}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if m["track_acc_all"] > best:
            best = m["track_acc_all"]
            torch.save({"model": model.state_dict(), "backbone": a.backbone, "head": a.head, "adv_classes": n_adv, "epoch": ep}, out / "best.pt")
            (out / "best_metrics.json").write_text(json.dumps(rec, indent=2))
            tracks.to_csv(out / "best_valid_tracks.csv", index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "hist": hist, "best": best, "epoch": ep}, out / "last.pt")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        plot(hist, out / "curves.png")
    print(f"done: best valid track accuracy {best:.2f}%  -> {out}")


if __name__ == "__main__":
    main()
