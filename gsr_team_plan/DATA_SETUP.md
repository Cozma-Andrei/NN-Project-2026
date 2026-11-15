# Getting the dataset (SoccerNet-GSR) and the training packs

You do not need a copy from a teammate. The dataset is public on Hugging Face (`SoccerNet/SN-GSR-2025`, licence GPL-3.0, no login needed), and our packs are rebuilt the same way on every machine. A script checks your copy against the team's numbers at the end, so we all train on identical data.

## What you need

| | Size | Needed for |
|---|---|---|
| Full dataset, `train` + `valid` | ~21 GB download (train 9.8 GB, valid 11.2 GB zips), ~23 GB unpacked | everything: the packs are built from it; Task C trains on it directly |
| Pack A (YOLO, 1280×720) | 1.8 GB | Task A |
| Pack B (calibration, 960×540) | 1.1 GB | Task B |
| Pack D (player crops + `labels.csv`) | 0.9 GB | Task D |

Never download `test` or `challenge` (team rules: `test` is used once, at the end, by the integration owner, and is never copied to fep). The script below doesn't download them.

## Steps

1. Clone the code and set up the environment (once). Follow the repository README: `conda activate tracklab`, `pip install -e .`. If installation fails on calibration, run `pip install -e plugins/calibration` first, then `pip install -e .` again.
2. Copy the `tools/` folder from this zip into `sn-gamestate/tools/`.
3. From the `sn-gamestate` folder, run the setup for your task:

   ```bash
   conda activate tracklab
   bash tools/setup_data.sh D        # Task D: full dataset + pack D
   bash tools/setup_data.sh A        # Task A
   bash tools/setup_data.sh B        # Task B
   bash tools/setup_data.sh none     # Task C / pipeline runs: full dataset only
   ```

   It downloads only what is missing, so it is safe to run again after an interruption. On fep, run it inside a SLURM job or tmux, because the download takes a while.
4. The last line must say `ALL CHECKS PASSED: your data matches the team's.` If any line says `FAIL`, don't train. Post the output in the team chat.

Expected result:

```
data/SoccerNetGS/train/SNGS-xxx/   57 clips
data/SoccerNetGS/valid/SNGS-xxx/   58 clips
data/packs/D/   train 125,677 crops (1,224 tracks), valid 64,518 crops (1,220 tracks), labels.csv md5 c317ff3e...
data/packs/A/   train 4,275 frames, valid 4,350 frames   (same frames for pack B)
```

## If you can't download 21 GB (slow connection, small disk)

Ask the integration owner for your pack alone, as a single archive (pack D is `packD.tar`, 0.9 GB). Then, from the `sn-gamestate` folder:

```bash
mkdir -p data/packs && tar -xf packD.tar -C data/packs      # -> data/packs/D/
PACK_ONLY=1 bash tools/setup_data.sh D                      # skips the full dataset, checks the pack
```

This is enough for Task D training. You still need the full dataset for anything that runs the TrackLab pipeline. Pack A uses symlinked folders (`mode1/`, `mode4/`), so rebuild pack A instead of copying it.

## Pack D: the player crops

The full dataset has no crops, only whole 1920×1080 frames (`data/SoccerNetGS/<split>/SNGS-xxx/img1/000001.jpg` … `000750.jpg`, 30 s at 25 fps) and one `Labels-GameState.json` per clip with a box, role, team and jersey number per person per frame. `tools/pack_gsr.py --task D` cuts the crops out of those frames.

**What's in it**

| | Train | Valid |
|---|---|---|
| Matches (game_id) | 4, 6, 9 | 2, 3, 5 |
| Clips | 57 | 58 |
| Frames used | every 5th | every 10th |
| Crops | 125,677 | 64,518 |
| Tracks (one player in one clip) | 1,224 (933 with a number, 41 distinct numbers) | 1,220 (254 without a number) |

- Only players and goalkeepers are included (no referees, no "other" staff), because those are the only ones with jersey numbers.
- Each crop is the annotated box at **full resolution**: no resizing, no padding, no context around the box. Box height: median 98 px, 80% of crops between 64 and 141 px (at 1080p).

**Layout**

```
data/packs/D/
  MANIFEST.json                 strides and counts (checked by setup_data.sh)
  labels.csv                    one row per crop
  train/SNGS-060_t001_000001.jpg   = clip SNGS-060, track 1, frame 000001
  valid/...
```

**`labels.csv` columns**

| Column | Meaning |
|---|---|
| `file` | path relative to `data/packs/D/`, e.g. `train/SNGS-060_t001_000001.jpg` |
| `split` | `train` or `valid` |
| `clip` | clip name, e.g. `SNGS-060` |
| `game_id` | match id (train 4, 6, 9; valid 2, 3, 5) |
| `track_id` | track id inside the clip (the same player for the whole clip) |
| `frame` | frame number in the clip, as text (`000001`) |
| `role` | `player` or `goalkeeper` |
| `jersey` | the number, **empty = this player's number is never visible in the clip** |
| `team` | `left` or `right`: side of the pitch in **that clip**, not a fixed team id (the sides swap at half time) |
| `box_h` | box height in pixels at 1080p |

**Things to know before training on it**

- **The jersey label belongs to the track, not the crop.** Every crop of a track gets the track's number, even when the number isn't visible (side view, front view, blur). In a typical track only a minority of crops show the digits. If you train on every crop, the network learns to recognise the *player* instead of reading the number; that's what happened in our runs D1–D3 (see `plan/TASK_D_STATUS.md`). `data_lists/E9_train_keep.csv` in this zip is a filtered list of train crops: the ones where the MMOCR reader confirmed the number, plus all crops of tracks without a number. Use it with `tools/train_jersey.py --train_keep`. It misses many readable crops, though.
- **One unique key per player in a split:** `clip + "_" + track_id`. Group by it for track-level evaluation. Never treat `track_id` alone as a player id across clips.
- **Matches are the real unit.** Players and kits repeat across the clips of one match. Never split train/valid by crop or by clip; use the given `split` column.
- **What our Task D networks see** (`tools/jersey_lib.py`): the top 60% of the crop (upper body), resized to 224×224. Augmentation is a random resized crop (scale 0.8–1.0) plus brightness and contrast jitter. **Never flip horizontally**, because it mirrors the digits.
- **pandas pitfall:** use `df["clip"]`, not `df.clip`. `DataFrame.clip` is a pandas method, and the attribute silently returns that instead of the column.

**Load it**

```python
import pandas as pd
from PIL import Image

df = pd.read_csv("data/packs/D/labels.csv", dtype={"jersey": str, "frame": str})
df["jersey"] = df["jersey"].fillna("")             # "" = no visible number in this track
df["track"] = df["clip"] + "_" + df["track_id"].astype(str)
train = df[df["split"] == "train"]
row = train.iloc[0]
img = Image.open("data/packs/D/" + row.file).convert("RGB")
print(row.track, row.jersey or "none", img.size)
```

Or use `tools/jersey_lib.py`: `load_labels(pack, split)` adds `track`, `has` (0/1), the two digit targets and the 101-class target. `JerseyCrops` is a ready PyTorch `Dataset` with the upper-body cut.

**Other tasks' crops.** Task C (re-ID) doesn't use pack D. TrackLab cuts its own 256×128 crops from the full dataset (including referees) the first time PRTReID training runs; see `plan/TASK_C_reid_team_role.md`.

## Rules (from `plan/01_TEAM_RULES.md`)

- Train only on `train`. `valid` is only for choosing the epoch and the experiment.
- Never move, copy or re-split clips between splits, and never put frames from different splits in one folder.
- Packs are for training and component metrics only. GS-HOTA (pipeline) runs always use the full, unresized dataset on the A4500 PC.
- External data only from your task file, after its overlap check.

## Scripts used

| Script | Does |
|---|---|
| `tools/setup_data.sh` | Download if missing, build the packs, check everything against the team's numbers |
| `tools/get_gsr.py` | Download and unzip one split from Hugging Face (TrackLab's built-in download link is dead) |
| `tools/pack_gsr.py` | Build pack A, B or D from the full dataset |
