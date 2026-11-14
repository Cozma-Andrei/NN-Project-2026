# Agent brief — Task B: pitch keypoint network for calibration (NBJW HRNet)

You are helping one team member with their part of a university Neural Networks project: improving one
network of the SoccerNet Game State Reconstruction baseline (`sn-gamestate` + TrackLab), measured by ablations.
Work in the `sn-gamestate` repository. Environment: `conda activate tracklab` (Python 3.9, torch 1.13.1, CUDA 11.7).

## Read first, in this order

1. `handoff/gsr_team_plan/plan/00_SPRINT.md` — the current 3-day plan; it overrides the task file where they differ.
2. `handoff/gsr_team_plan/plan/01_TEAM_RULES.md` — rules for data, training, compute.
3. `handoff/gsr_team_plan/plan/TASK_B_calibration.md`.
4. `handoff/gsr_team_plan/tools/README.md` — helper scripts that already exist and were tested.

## Non-negotiable rules (from plan/01_TEAM_RULES.md)

1. Train only on the `train` split (matches 4, 6, 9). Use `valid` (matches 2, 3, 5) only for early stopping, choosing runs and metrics. Never touch `test`; never move, copy, merge or re-split clips or frames between splits.
2. One fixed training recipe per network (in the task file, sprint overrides in `plan/00_SPRINT.md`). Do not tune optimizer, learning rate, batch size or epochs. No normalization/activation swaps, no single-layer add/remove experiments.
3. Each experiment changes exactly one thing vs its named reference run. Seed 0.
4. Every run saves its config, train and validation loss per epoch, the validation metric per epoch, and the best checkpoint; plot train vs validation curves.
5. Run names: `<task><number>_<change>` (e.g. `B2b_combmseaw`), used for the run folder, the checkpoint and the tracker row.
6. Training runs on the fep A100 queue as SLURM jobs from `jobs/template.sh`, each under 10 hours, checkpointing every epoch. Before submitting, smoke-test 20 batches locally (CPU is fine).
7. Never run the TrackLab pipeline on the downsized packs in `data/packs/`: they are for training and component metrics only. Pipeline (GS-HOTA) runs are done by the integration owner on the A4500 PC.
8. Do not edit other tasks' code, the shared configs in `sn_gamestate/configs/` or installed packages in site-packages.

## Stop and ask the human when

- a step's check fails, or a command errors and the fix is not obvious from the error;
- the plan does not cover the situation, or two plan files disagree (the sprint file wins; still mention it);
- anything would require changing the rules above, using another dataset, or touching `test`.

## Your sprint deliverables (3 days)

- `tools/train_kp.py`: NBJW keypoint HRNet (built like `NBJW_Calib_Keypoints.__init__` in `sn_gamestate/calibration/nbjw_calib.py`, starting from `pretrained_models/calibration/SV_kp`), data from pack B through `SoccerNetCalibrationDataset(root_dir='<abs>/data/packs/B/', split=..., transform=..., main_cam_only=False)` followed by `ds.files = ds.get_image_files(rate=1)`. Recipe: AdamW lr 1e-4, cosine, 6 epochs, batch 4, early stopping after 3, seed 0.
- `tools/eval_calib.py`: keypoints + lines → camera exactly as the pipeline's `NBJW_Calib_Keypoints.process` / `NBJW_Calib.process`; metrics: athlete position error (m), share within 5 m, share of frames with a camera, split wide vs close-up (close-up = fewer than 4 labelled lines), using `data/packs/B/athletes/`.
- Jobs, submitted together: B1 `MSELoss`, B2b `CombMSEAW`, B4a close-up zoom augmentation (random 1.5–3× zoom crop that moves the line points the same way and drops points leaving the image).
- Results: the metrics for B0 (pretrained), B1 and the best run; curves; 5 failure frames with predicted keypoints drawn.
- **Error analysis (required):** position error and within-5 m share split into close-ups (fewer than 4 labelled lines) vs wide shots for B0, B1 and the best run, explained by how many pitch lines are visible.
- Compare every run against your task's baseline row (A0/B0/C0/D0) computed **on the same pack / same valid data**, never against numbers from a different data setup. Never run the TrackLab pipeline on fewer frames or smaller images to make a 'downsized baseline': GS-HOTA needs all 750 frames at full resolution (see 'Which baseline to compare against' in `plan/00_SPRINT.md`).
- Results are single-seed and short-trained: label 1–2 point differences as preliminary.

## Files you create or change

`tools/train_kp.py`, `tools/eval_calib.py`, job files `jobs/B*.sh`, run outputs under `runs/calibB/`. Nothing else without asking.

## Task-specific facts already verified

- Always `import tools.nbjw_compat` before importing `nbjw_calib.model.transforms` (torchvision 0.14 has no `transforms.v2`; the shim fixes it, verified).
- Use the presets `transforms` (train) and `no_transforms` (valid) from `nbjw_calib.model.transforms`; one loaded sample is image (3, 540, 960), target (58, 270, 480).
- The dataset path must contain no dot except file extensions (the loader splits on the first `.`).
- Do not modify files in `plugins/calibration/`; subclass or wrap instead.

## How to work

- Go step by step through the task file (with the sprint overrides); run each step's check and show the output before moving on.
- Prefer existing code (the plan names the files) over rewriting it.
- After each finished run, give the human a tracker row: run name, change vs reference, reference run, component metric on valid, status.
