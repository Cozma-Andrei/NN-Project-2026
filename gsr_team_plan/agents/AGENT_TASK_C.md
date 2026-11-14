# Agent brief — Task C: re-ID, team and role network (PRTReID)

You are helping one team member with their part of a university Neural Networks project: improving one
network of the SoccerNet Game State Reconstruction baseline (`sn-gamestate` + TrackLab), measured by ablations.
Work in the `sn-gamestate` repository. Environment: `conda activate tracklab` (Python 3.9, torch 1.13.1, CUDA 11.7).

## Read first, in this order

1. `handoff/gsr_team_plan/plan/00_SPRINT.md` — the current 3-day plan; it overrides the task file where they differ.
2. `handoff/gsr_team_plan/plan/01_TEAM_RULES.md` — rules for data, training, compute.
3. `handoff/gsr_team_plan/plan/TASK_C_reid_team_role.md`.
4. `handoff/gsr_team_plan/tools/README.md` — helper scripts that already exist and were tested.

## Non-negotiable rules (from plan/01_TEAM_RULES.md)

1. Train only on the `train` split (matches 4, 6, 9). Use `valid` (matches 2, 3, 5) only for early stopping, choosing runs and metrics. Never touch `test`; never move, copy, merge or re-split clips or frames between splits.
2. One fixed training recipe per network (in the task file, sprint overrides in `plan/00_SPRINT.md`). Do not tune optimizer, learning rate, batch size or epochs. No normalization/activation swaps, no single-layer add/remove experiments.
3. Each experiment changes exactly one thing vs its named reference run. Seed 0.
4. Every run saves its config, train and validation loss per epoch, the validation metric per epoch, and the best checkpoint; plot train vs validation curves.
5. Run names: `<task><number>_<change>` (e.g. `C5b_focal_weighted`), used for the run folder, the checkpoint and the tracker row.
6. Training runs on the fep A100 queue as SLURM jobs from `jobs/template.sh`, each under 10 hours, checkpointing every epoch. Before submitting, smoke-test 20 batches locally (CPU is fine).
7. Never run the TrackLab pipeline on the downsized packs in `data/packs/`: they are for training and component metrics only. Pipeline (GS-HOTA) runs are done by the integration owner on the A4500 PC.
8. Do not edit other tasks' code, the shared configs in `sn_gamestate/configs/` or installed packages in site-packages.

## Stop and ask the human when

- a step's check fails, or a command errors and the fix is not obvious from the error;
- the plan does not cover the situation, or two plan files disagree (the sprint file wins; still mention it);
- anything would require changing the rules above, using another dataset, or touching `test`.

## Your sprint deliverables (3 days)

- Editable PRTReID in `~/prtreid` (`pip install -e ~/prtreid --no-deps`) with configurable loss: `cfg.loss.team_weight` (0.1), `cfg.loss.role_weight` (1.5), `cfg.loss.role_loss` in {focal, ce, focal_weighted, ce_smooth}; defaults must reproduce today's loss exactly. Commit in `~/prtreid`.
- A 1-epoch smoke job first. Every TrackLab training command includes: `dataset.nvid=-1 test_tracking=False eval_tracking=False modules.reid.training_enabled=True modules.reid.cfg.test.evaluate=False modules.reid.cfg.model.load_config=False modules.reid.cfg.train.eval_freq=1 modules.reid.cfg.test.start_eval=0 modules.reid.cfg.train.seed=0 data_dir=<fep data folder>`.
- After the smoke job passes, jobs submitted together (all fine-tuned from the provided checkpoint, `modules.reid.cfg.train.max_epoch=5`): C-ref (focal), C5b `loss.role_loss=focal_weighted`, C5a `loss.role_loss=ce`, C4a `loss.role_weight=0`.
- `tools/eval_reid_attrs.py`: role accuracy + per-class F1 (goalkeeper, referee, other) and team accuracy (k-means with 2 clusters on mean ground-truth player-track embeddings per clip, as `sn_gamestate/team/tracklet_team_clustering_api.py`, best of 2 label swaps), on valid ground-truth boxes of every 10th frame.
- Results: C0 (provided checkpoint) and each run: re-ID mAP/rank-1, role F1 per class, team accuracy, side by side; curves; 5 failure cases.
- **Error analysis (required):** role F1 per class (player 605k vs referee 61.5k vs goalkeeper 24.1k vs other 1.4k boxes) and team accuracy per clip, explained by class imbalance and per-match kits.
- Compare every run against your task's baseline row (A0/B0/C0/D0) computed **on the same pack / same valid data**, never against numbers from a different data setup. Never run the TrackLab pipeline on fewer frames or smaller images to make a 'downsized baseline': GS-HOTA needs all 750 frames at full resolution (see 'Which baseline to compare against' in `plan/00_SPRINT.md`).
- Results are single-seed and short-trained: label 1–2 point differences as preliminary.

## Files you create or change

`~/prtreid` (your fork of PRTReID), `tools/eval_reid_attrs.py`, job files `jobs/C*.sh`. Nothing else without asking.

## Task-specific facts already verified

- C needs the full train + valid dataset on fep: `python tools/get_gsr.py <data folder> train valid` (TrackLab's own download link is dead).
- The TrackLab PRTReID training path has not been run by the team yet: if the smoke job errors, stop and report the error.

## How to work

- Go step by step through the task file (with the sprint overrides); run each step's check and show the output before moving on.
- Prefer existing code (the plan names the files) over rewriting it.
- After each finished run, give the human a tracker row: run name, change vs reference, reference run, component metric on valid, status.
