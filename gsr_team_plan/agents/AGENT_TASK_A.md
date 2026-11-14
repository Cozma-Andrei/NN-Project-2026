# Agent brief — Task A: person detector (YOLO11 / RT-DETR)

You are helping one team member with their part of a university Neural Networks project: improving one
network of the SoccerNet Game State Reconstruction baseline (`sn-gamestate` + TrackLab), measured by ablations.
Work in the `sn-gamestate` repository. Environment: `conda activate tracklab` (Python 3.9, torch 1.13.1, CUDA 11.7).

## Read first, in this order

1. `handoff/gsr_team_plan/plan/00_SPRINT.md` — the current 3-day plan; it overrides the task file where they differ.
2. `handoff/gsr_team_plan/plan/01_TEAM_RULES.md` — rules for data, training, compute.
3. `handoff/gsr_team_plan/plan/TASK_A_detector.md`.
4. `handoff/gsr_team_plan/tools/README.md` — helper scripts that already exist and were tested.

## Non-negotiable rules (from plan/01_TEAM_RULES.md)

1. Train only on the `train` split (matches 4, 6, 9). Use `valid` (matches 2, 3, 5) only for early stopping, choosing runs and metrics. Never touch `test`; never move, copy, merge or re-split clips or frames between splits.
2. One fixed training recipe per network (in the task file, sprint overrides in `plan/00_SPRINT.md`). Do not tune optimizer, learning rate, batch size or epochs. No normalization/activation swaps, no single-layer add/remove experiments.
3. Each experiment changes exactly one thing vs its named reference run. Seed 0.
4. Every run saves its config, train and validation loss per epoch, the validation metric per epoch, and the best checkpoint; plot train vs validation curves.
5. Run names: `<task><number>_<change>` (e.g. `A3_rtdetr_l`), used for the run folder, the checkpoint and the tracker row.
6. Training runs on the fep A100 queue as SLURM jobs from `jobs/template.sh`, each under 10 hours, checkpointing every epoch. Before submitting, smoke-test 20 batches locally (CPU is fine).
7. Never run the TrackLab pipeline on the downsized packs in `data/packs/`: they are for training and component metrics only. Pipeline (GS-HOTA) runs are done by the integration owner on the A4500 PC.
8. Do not edit other tasks' code, the shared configs in `sn_gamestate/configs/` or installed packages in site-packages.

## Stop and ask the human when

- a step's check fails, or a command errors and the fix is not obvious from the error;
- the plan does not cover the situation, or two plan files disagree (the sprint file wins; still mention it);
- anything would require changing the rules above, using another dataset, or touching `test`.

## Your sprint deliverables (3 days)

- A0: `yolo detect val` of the COCO `yolo11m.pt` on pack A valid (`classes=0`).
- `tools/recall_by_size.py` (recall for box heights <50, 50–100, >100 px in original 1080p pixels; pack A images are 1280×720, so multiply by 1.5).
- Training jobs, submitted together: A1 `yolo11m`, A2b `yolo11l`, A3 `rtdetr-l` (RT-DETR recipe: `optimizer=AdamW lr0=0.0001`), all with `data=data/packs/A/gsr1.yaml epochs=15 patience=5 imgsz=1280 batch=8 seed=0 deterministic=True` (YOLO: `optimizer=SGD lr0=0.01`). A7: same YOLO recipe with `model=yolo11m.pt data=data/packs/A/gsr4.yaml` (4 role classes, the auxiliary-task run); report per-class AP from `yolo val`. No pipeline change in the sprint.
- Results: mAP50-95, recall, recall by size for A0, A1 and the best run; per-class AP for A7; `results.png` curves; 5 failure images with one line each.
- **Error analysis (required):** recall by box height (<50 / 50–100 / >100 px at 1080p) for A0, A1 and the best run, and A7's per-class AP, each explained by a dataset property (small distant players; rare goalkeepers and 'other').
- Compare every run against your task's baseline row (A0/B0/C0/D0) computed **on the same pack / same valid data**, never against numbers from a different data setup. Never run the TrackLab pipeline on fewer frames or smaller images to make a 'downsized baseline': GS-HOTA needs all 750 frames at full resolution (see 'Which baseline to compare against' in `plan/00_SPRINT.md`).
- Results are single-seed and short-trained: label 1–2 point differences as preliminary.

## Files you create or change

`tools/recall_by_size.py`, job files `jobs/A*.sh`, run outputs under `runs/detA/`. Nothing else without asking.

## Task-specific facts already verified

- Pack A is already built and checked: `data/packs/A/gsr1.yaml` (1 class) and `gsr4.yaml` (4 role classes). After copying to fep, set `path:` in both yaml files to the fep pack path.
- The pipeline keeps only detector class 0, so A7's pipeline check (GS-HOTA) is after the sprint and needs a wrapper change by the integration owner; A7's training and per-class AP are in the sprint.

## How to work

- Go step by step through the task file (with the sprint overrides); run each step's check and show the output before moving on.
- Prefer existing code (the plan names the files) over rewriting it.
- After each finished run, give the human a tracker row: run name, change vs reference, reference run, component metric on valid, status.
