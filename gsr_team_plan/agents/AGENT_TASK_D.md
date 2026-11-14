# Agent brief — Task D: jersey number networks

You are helping one team member with their part of a university Neural Networks project: improving one
network of the SoccerNet Game State Reconstruction baseline (`sn-gamestate` + TrackLab), measured by ablations.
Work in the `sn-gamestate` repository. Environment: `conda activate tracklab` (Python 3.9, torch 1.13.1, CUDA 11.7).

## Read first, in this order

1. `handoff/gsr_team_plan/plan/00_SPRINT.md` — the current 3-day plan; it overrides the task file where they differ.
2. `handoff/gsr_team_plan/plan/01_TEAM_RULES.md` — rules for data, training, compute.
3. `handoff/gsr_team_plan/plan/TASK_D_jersey.md (full plan) and the 'Task D sprint steps' section of plan/00_SPRINT.md (what to do now)`.
4. `handoff/gsr_team_plan/tools/README.md` — helper scripts that already exist and were tested.

## Non-negotiable rules (from plan/01_TEAM_RULES.md)

1. Train only on the `train` split (matches 4, 6, 9). Use `valid` (matches 2, 3, 5) only for early stopping, choosing runs and metrics. Never touch `test`; never move, copy, merge or re-split clips or frames between splits.
2. One fixed training recipe per network (in the task file, sprint overrides in `plan/00_SPRINT.md`). Do not tune optimizer, learning rate, batch size or epochs. No normalization/activation swaps, no single-layer add/remove experiments.
3. Each experiment changes exactly one thing vs its named reference run. Seed 0.
4. Every run saves its config, train and validation loss per epoch, the validation metric per epoch, and the best checkpoint; plot train vs validation curves.
5. Run names: `<task><number>_<change>` (e.g. `D2_vits_digits`), used for the run folder, the checkpoint and the tracker row.
6. Training runs on the fep A100 queue as SLURM jobs from `jobs/template.sh`, each under 10 hours, checkpointing every epoch. Before submitting, smoke-test 20 batches locally (CPU is fine).
7. Never run the TrackLab pipeline on the downsized packs in `data/packs/`: they are for training and component metrics only. Pipeline (GS-HOTA) runs are done by the integration owner on the A4500 PC.
8. Do not edit other tasks' code, the shared configs in `sn_gamestate/configs/` or installed packages in site-packages.

## Stop and ask the human when

- a step's check fails, or a command errors and the fix is not obvious from the error;
- the plan does not cover the situation, or two plan files disagree (the sprint file wins; still mention it);
- anything would require changing the rules above, using another dataset, or touching `test`.

## Your sprint deliverables (3 days)

- Follow 'Task D sprint steps' in `plan/00_SPRINT.md` exactly.
- `tools/train_jersey.py`: timm backbone (`pretrained=True, num_classes=0`) + linear heads; input = top 60% of each pack-D crop resized to 224×224; augmentation random resized crop (scale 0.8–1.0) + colour jitter 0.2; **no horizontal flip**. Recipe: AdamW lr 1e-4, wd 0.05, cosine, 8 epochs, batch 128, seed 0. Keep the checkpoint with the best valid track accuracy.
- `tools/eval_jersey.py`: per valid track, keep frames with p(has_number) > 0.5 (fixed); none kept → "none"; else sum per-number probabilities (digit heads: p(tens)·p(units)) and take the max. Report track accuracy: all, with number, without number.
- Jobs, submitted together: D1 `resnet18` digit heads (has_number 2, tens 11 incl. blank, units 10), D2 `vit_small_patch16_224` digit heads, D3 `resnet18` single 101-way head.
- Results: track accuracies for D0 (MMOCR baseline, run by the integration owner) and D1–D3; curves; 5 failure tracks as crop strips.
- **Error analysis (required):** track accuracy on tracks with vs without a number (24% have none) and by track length, explained by weak per-track labels and fragmented tracks.
- Compare every run against your task's baseline row (A0/B0/C0/D0) computed **on the same pack / same valid data**, never against numbers from a different data setup. Never run the TrackLab pipeline on fewer frames or smaller images to make a 'downsized baseline': GS-HOTA needs all 750 frames at full resolution (see 'Which baseline to compare against' in `plan/00_SPRINT.md`).
- Results are single-seed and short-trained: label 1–2 point differences as preliminary.

## Files you create or change

`tools/train_jersey.py`, `tools/eval_jersey.py`, job files `jobs/D*.sh`, run outputs under `runs/jerseyD/`. Nothing else without asking.

## Task-specific facts already verified

- Pack D: `data/packs/D/labels.csv` (columns file, split, clip, game_id, track_id, frame, role, jersey, team, box_h; `jersey` empty = track never shows a number), crops in `data/packs/D/{train,valid}/`, cut from full 1080p frames. Train: 125,677 crops / 1224 tracks; valid: 64,518 crops / 1220 tracks.
- Labels are per track (weak): the number is given on frames where it is not visible. Do not try to 'clean' them by hand in the sprint.
- No TrackLab module in the sprint.

## How to work

- Go step by step through the task file (with the sprint overrides); run each step's check and show the output before moving on.
- Prefer existing code (the plan names the files) over rewriting it.
- After each finished run, give the human a tracker row: run name, change vs reference, reference run, component metric on valid, status.
