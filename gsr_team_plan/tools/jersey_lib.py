"""Shared code for Task D (jersey numbers): pack-D dataset, frame network, track aggregation, metrics.

Used by tools/train_jersey.py and tools/eval_jersey.py. See handoff/gsr_team_plan/plan/00_SPRINT.md,
"Task D sprint steps", for the specification this implements.

Targets per crop (weak labels: the track's number is given on every frame of the track):
  has    : 1 if the track has a number, else 0
  tens   : 0-9, or 10 = "blank" (one-digit number)        (digit heads)
  units  : 0-9                                             (digit heads)
  cls101 : the number 0-99, or 100 = "none"                (single 101-way head)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

BLANK = 10
NONE = 100
UPPER_BODY = 0.6  # keep the top 60% of each player crop


def load_labels(pack_root, split):
    df = pd.read_csv(Path(pack_root) / "labels.csv", dtype={"jersey": str, "frame": str})
    df["jersey"] = df["jersey"].fillna("")
    df = df[df["split"] == split].reset_index(drop=True)
    num = df.jersey.map(lambda s: int(s) if s else -1)
    df["has"] = (num >= 0).astype(int)
    df["tens"] = num.map(lambda n: BLANK if 0 <= n < 10 else (n // 10 if n >= 10 else 0))
    df["units"] = num.map(lambda n: n % 10 if n >= 0 else 0)
    df["cls101"] = num.map(lambda n: n if n >= 0 else NONE)
    df["track"] = df["clip"] + "_" + df["track_id"].astype(str)
    return df


def build_transforms(mean, std, train):
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if train:
        # No horizontal flip: it mirrors digits.
        return T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                          T.ColorJitter(brightness=0.2, contrast=0.2)] + norm)
    return T.Compose([T.Resize((224, 224))] + norm)


class JerseyCrops(Dataset):
    def __init__(self, pack_root, df, transform):
        self.root, self.df, self.tf = Path(pack_root), df, transform
        cols = ["has", "tens", "units", "cls101"] + (["adv"] if "adv" in df else [])
        self.targets = df[cols].to_numpy(dtype=np.int64)  # column 4 = adversary label (D4), if present

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        with Image.open(self.root / self.df.file.iat[i]) as im:
            im = im.convert("RGB")
            w, h = im.size
            im = im.crop((0, 0, w, max(1, int(round(h * UPPER_BODY)))))
        return self.tf(im), torch.from_numpy(self.targets[i])


class GradReverse(torch.autograd.Function):
    """Identity forward; multiplies the gradient by -lam backward (Ganin & Lempitsky 2015, DANN)."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class JerseyNet(nn.Module):
    """timm backbone (global features) + linear heads. head='digits' or 'cls101'.
    adv_classes > 0 adds an adversary MLP behind a gradient-reversal layer (D4): it learns to predict the
    appearance group (clip x team side) of a crop, while the reversed gradient pushes the backbone to make
    that impossible. Set self.lam from the training loop (0 = adversary trains but does not affect the backbone)."""

    def __init__(self, backbone, head, pretrained=True, adv_classes=0):
        super().__init__()
        self.head_type = head
        self.lam = 0.0
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        f = self.backbone.num_features
        if head == "digits":
            self.has, self.tens, self.units = nn.Linear(f, 2), nn.Linear(f, 11), nn.Linear(f, 10)
        elif head == "cls101":
            self.cls = nn.Linear(f, 101)
        else:
            raise ValueError(head)
        self.adv = nn.Sequential(nn.Linear(f, 256), nn.ReLU(), nn.Linear(256, adv_classes)) if adv_classes else None

    def data_mean_std(self):
        cfg = self.backbone.pretrained_cfg
        return cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225))

    def forward(self, x):
        z = self.backbone(x)
        if self.head_type == "digits":
            out = {"has": self.has(z), "tens": self.tens(z), "units": self.units(z)}
        else:
            out = {"cls": self.cls(z)}
        if self.adv is not None and self.training:
            out["adv"] = self.adv(GradReverse.apply(z, self.lam))
        return out

    def loss(self, out, y):
        if self.head_type == "digits":
            loss = F.cross_entropy(out["has"], y[:, 0])
            m = y[:, 0] == 1
            if m.any():  # digit losses only for crops whose track has a number
                loss = loss + F.cross_entropy(out["tens"][m], y[m, 1]) + F.cross_entropy(out["units"][m], y[m, 2])
            return loss
        return F.cross_entropy(out["cls"], y[:, 3])

    @torch.no_grad()
    def frame_probs(self, out):
        """-> p_has (N,), p_num (N, 100): probability of each number 0-99 for each crop."""
        if self.head_type == "digits":
            p_has = out["has"].float().softmax(1)[:, 1]
            pt, pu = out["tens"].float().softmax(1), out["units"].float().softmax(1)
            n = torch.arange(100, device=pt.device)
            tens_idx = torch.where(n < 10, torch.full_like(n, BLANK), n // 10)
            p_num = pt[:, tens_idx] * pu[:, n % 10]
            return p_has, p_num
        p = out["cls"].float().softmax(1)
        return 1 - p[:, NONE], p[:, :NONE]


def aggregate_tracks(df, p_has, p_num, threshold=0.5):
    """Track-level prediction (fixed rule, not tuned): keep frames with p_has > threshold;
    none kept -> 'none' (-1); else the number with the largest summed probability over kept frames."""
    tracks = df.track.to_numpy()
    out = {}
    order = np.argsort(tracks, kind="stable")
    bounds = np.flatnonzero(np.r_[True, tracks[order][1:] != tracks[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        keep = idx[p_has[idx] > threshold]
        out[tracks[idx[0]]] = -1 if len(keep) == 0 else int(p_num[keep].sum(0).argmax())
    return out


def track_table(df, preds, train_numbers=None):
    t = df.groupby("track").agg(clip=("clip", "first"), game_id=("game_id", "first"), jersey=("jersey", "first"),
                                n_crops=("file", "size"), median_h=("box_h", "median")).reset_index()
    t["true"] = t.jersey.map(lambda s: int(s) if s else -1)
    t["pred"] = t.track.map(preds)
    t["correct"] = t.true == t.pred
    if train_numbers is not None:
        t["unseen_in_train"] = (t.true >= 0) & ~t.true.isin(train_numbers)
    return t


def track_metrics(t):
    def acc(x):
        return float(x.correct.mean() * 100) if len(x) else float("nan")
    m = {"track_acc_all": acc(t), "track_acc_number": acc(t[t.true >= 0]), "track_acc_none": acc(t[t.true < 0]),
         "n_tracks": int(len(t)), "n_number": int((t.true >= 0).sum()), "n_none": int((t.true < 0).sum())}
    if "unseen_in_train" in t:
        u = t[t.unseen_in_train]
        m.update({"track_acc_unseen_number": acc(u), "n_unseen_number": int(len(u))})
    return m
