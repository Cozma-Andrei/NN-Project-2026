#!/usr/bin/env python3
"""Build small, training-ready packs of SoccerNet-GSR for one task (A, B or D).

Run on the PC that has the full dataset, then copy the pack to fep:
    python tools/pack_gsr.py --task A            # -> data/packs/A   (YOLO format, 1280x720)
    python tools/pack_gsr.py --task B            # -> data/packs/B   (NBJW calibration format, 960x540)
    python tools/pack_gsr.py --task D            # -> data/packs/D   (full-resolution player crops + labels.csv)
    rsync -av data/packs/A/ <user>@fep:<path>/sn-gamestate/data/packs/A/

Rules this tool enforces (see the Team rules tab):
  * only 'train' and 'valid' are packed, each into its own folder; 'test' and 'challenge' are refused;
  * packs are for TRAINING and COMPONENT METRICS only. Pipeline / GS-HOTA runs always use the full,
    unresized dataset on the A4500 PC (resized or subsampled frames change every module's result);
  * frames are subsampled by a stride on the frame index, per clip (never mixing clips or splits).

Task C is not packed: PRTReID trains through TrackLab, which reads the full dataset layout
(use tools/get_gsr.py on fep for train + valid).
"""
import argparse
import csv
import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

SPLITS = ("train", "valid")
W, H = 1920, 1080
PERSON_ROLES = {"player": 0, "goalkeeper": 1, "referee": 2, "other": 3}  # class ids for 4-class mode
JERSEY_ROLES = {"player", "goalkeeper"}
DEFAULT_STRIDE = {"A": (10, 10), "B": (10, 10), "D": (5, 10)}  # (train, valid)


def load_clip(clip_dir):
    with open(clip_dir / "Labels-GameState.json") as f:
        data = json.load(f)
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    images = [im for im in data["images"] if im.get("is_labeled", True)]
    return data["info"], images, anns_by_img


def sampled(images, stride):
    return images[::stride]


def yolo_line(cls, b):
    """bbox_image (pixels, 1920x1080) -> YOLO line, clipped to the image. Resize-invariant."""
    x1, y1 = max(0.0, b["x"]), max(0.0, b["y"])
    x2, y2 = min(W, b["x"] + b["w"]), min(H, b["y"] + b["h"])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def save_resized(src, dst, size, quality):
    with Image.open(src) as im:
        im.convert("RGB").resize(size, Image.BILINEAR).save(dst, quality=quality)


def pack_A(clip_dir, split, stride, out):
    info, images, anns = load_clip(clip_dir)
    n = 0
    for im in sampled(images, stride):
        stem = f"{clip_dir.name}_{Path(im['file_name']).stem}"
        lines1, lines4 = [], []
        for a in anns.get(im["image_id"], []):
            role = a.get("attributes", {}).get("role") if a.get("supercategory") == "object" else None
            if role not in PERSON_ROLES:
                continue  # skips balls and non-object annotations
            l1, l4 = yolo_line(0, a["bbox_image"]), yolo_line(PERSON_ROLES[role], a["bbox_image"])
            if l1:
                lines1.append(l1)
                lines4.append(l4)
        save_resized(clip_dir / "img1" / im["file_name"], out / "images" / split / f"{stem}.jpg", (1280, 720), 90)
        (out / "labels1" / split / f"{stem}.txt").write_text("\n".join(lines1))
        (out / "labels4" / split / f"{stem}.txt").write_text("\n".join(lines4))
        n += 1
    return n


def pack_B(clip_dir, split, stride, out):
    info, images, anns = load_clip(clip_dir)
    athletes, n = {}, 0
    for im in sampled(images, stride):
        if not im.get("has_labeled_pitch", False):
            continue
        img_anns = anns.get(im["image_id"], [])
        pitch = [a for a in img_anns if a.get("supercategory") == "pitch"]
        if not pitch:
            continue
        stem = f"{clip_dir.name}_{Path(im['file_name']).stem}"
        save_resized(clip_dir / "img1" / im["file_name"], out / split / f"{stem}.jpg", (960, 540), 90)
        with open(out / split / f"{stem}.json", "w") as f:
            json.dump(pitch[0]["lines"], f)  # normalised [0,1] coordinates: unaffected by the resize
        # foot points in ORIGINAL 1920x1080 pixels + true pitch position (m), for the position-error metric
        athletes[stem] = [
            {"foot_px": [a["bbox_image"]["x_center"], a["bbox_image"]["y"] + a["bbox_image"]["h"]],
             "pitch_m": [a["bbox_pitch"]["x_bottom_middle"], a["bbox_pitch"]["y_bottom_middle"]],
             "n_lines": len(pitch[0]["lines"])}
            for a in img_anns
            if a.get("supercategory") == "object" and a.get("attributes", {}).get("role") in PERSON_ROLES
            and a.get("bbox_pitch")
        ]
        n += 1
    with open(out / "athletes" / f"{split}_{clip_dir.name}.json", "w") as f:
        json.dump(athletes, f)
    return n


def pack_D(clip_dir, split, stride, out, min_h=20):
    info, images, anns = load_clip(clip_dir)
    rows = []
    for im in sampled(images, stride):
        crop_anns = [a for a in anns.get(im["image_id"], [])
                     if a.get("supercategory") == "object"
                     and a.get("attributes", {}).get("role") in JERSEY_ROLES
                     and a["bbox_image"]["h"] >= min_h]
        if not crop_anns:
            continue
        with Image.open(clip_dir / "img1" / im["file_name"]) as frame:
            frame = frame.convert("RGB")  # full 1080p: never crop from a downscaled frame
            for a in crop_anns:
                b = a["bbox_image"]
                box = (max(0, int(b["x"])), max(0, int(b["y"])),
                       min(W, int(b["x"] + b["w"])), min(H, int(b["y"] + b["h"])))
                if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                    continue
                name = f"{clip_dir.name}_t{a['track_id']:03d}_{Path(im['file_name']).stem}.jpg"
                frame.crop(box).save(out / split / name, quality=95)
                attrs = a["attributes"]
                rows.append({"file": f"{split}/{name}", "split": split, "clip": clip_dir.name,
                             "game_id": info.get("game_id"), "track_id": a["track_id"],
                             "frame": Path(im["file_name"]).stem, "role": attrs.get("role"),
                             "jersey": attrs.get("jersey") or "", "team": attrs.get("team") or "",
                             "box_h": round(b["h"], 1)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["A", "B", "D"], required=True)
    p.add_argument("--dataset_root", default="data/SoccerNetGS")
    p.add_argument("--out_root", default="data/packs")
    p.add_argument("--train_stride", type=int)
    p.add_argument("--valid_stride", type=int)
    p.add_argument("--max_clips", type=int, default=0, help="debug: only the first N clips per split")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    strides = dict(zip(SPLITS, DEFAULT_STRIDE[args.task]))
    if args.train_stride:
        strides["train"] = args.train_stride
    if args.valid_stride:
        strides["valid"] = args.valid_stride
    out = Path(args.out_root) / args.task
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; delete it first so packs never mix runs with different settings.")

    for split in SPLITS:
        if args.task == "A":
            for d in ("images", "labels1", "labels4"):
                (out / d / split).mkdir(parents=True, exist_ok=True)
        else:
            (out / split).mkdir(parents=True, exist_ok=True)
        if args.task == "B":
            (out / "athletes").mkdir(parents=True, exist_ok=True)

    counts, all_rows = {}, []
    for split in SPLITS:
        clips = sorted(Path(args.dataset_root, split).glob("SNGS-*"))
        if args.max_clips:
            clips = clips[: args.max_clips]
        fn = {"A": pack_A, "B": pack_B, "D": pack_D}[args.task]
        with Pool(args.workers) as pool:
            res = pool.map(partial(fn, split=split, stride=strides[split], out=out), clips)
        if args.task == "D":
            rows = [r for rs in res for r in rs]
            all_rows += rows
            counts[split] = {"clips": len(clips), "crops": len(rows),
                             "tracks": len({(r["clip"], r["track_id"]) for r in rows})}
        else:
            counts[split] = {"clips": len(clips), "frames": sum(res)}
        print(f"[{args.task}] {split}: {counts[split]}", flush=True)

    if args.task == "A":
        # YOLO finds a label by replacing /images/ with /labels/ in the image path and resolves
        # directory symlinks, so each mode gets REAL folders: images/<split>/ holds per-file relative
        # symlinks to the shared images, labels/<split>/ holds that mode's label files.
        # Relative links survive `rsync -a` to fep.
        for mode, labels in (("mode1", "labels1"), ("mode4", "labels4")):
            for split in SPLITS:
                img_dir, lbl_dir = out / mode / "images" / split, out / mode / "labels"
                img_dir.mkdir(parents=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
                for img in sorted((out / "images" / split).iterdir()):
                    os.symlink(os.path.join("..", "..", "..", "images", split, img.name), img_dir / img.name)
                os.rename(out / labels / split, lbl_dir / split)
            (out / labels).rmdir()
        names1 = "{0: person}"
        names4 = "{0: player, 1: goalkeeper, 2: referee, 3: other}"
        for mode, names in (("mode1", names1), ("mode4", names4)):
            (out / f"gsr{mode[-1]}.yaml").write_text(
                f"# absolute path on the machine that built the pack: change it after copying to fep\n"
                f"path: {(out / mode).resolve()}\ntrain: images/train\nval: images/valid\nnames: {names}\n")
    if args.task == "D":
        with open(out / "labels.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    manifest = {"task": args.task, "strides": strides, "counts": counts,
                "source": str(Path(args.dataset_root).resolve()), "max_clips": args.max_clips}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"done: {out}  {manifest['counts']}")


if __name__ == "__main__":
    main()
