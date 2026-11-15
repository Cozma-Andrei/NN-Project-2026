# GSR team plan: SoccerNet Game State Reconstruction (NN project)

Everything a team member, or their coding agent, needs to start on their task.
The project improves four neural networks of the SoccerNet-GSR baseline
(`sn-gamestate` + TrackLab), one per member, measured by ablations.

## What is in this folder

| Path | What it is |
| --- | --- |
| `plan/00_SPRINT.md` | **Start here.** The 3-day plan for the first batch of results: compute, fep setup, data packs, scope per task, day-by-day. It overrides the task files where they differ. |
| `DATA_SETUP.md` | **Getting the data.** Download SoccerNet-GSR (train + valid) and build your training pack with one command, checked against the team's numbers. Also explains the pack D player crops: layout, `labels.csv` columns, pitfalls, loading code. |
| `plan/TASK_D_STATUS.md` | Task D results so far (D0–D6), what worked, and how past SoccerNet winners solved jersey numbers. |
| `plan/01_TEAM_RULES.md` | Rules for everyone: data splits (no mixing!), training rules, GPUs, pipeline runs, results tracker. |
| `plan/TASK_A_detector.md` | Task A: person detector (YOLO11 / RT-DETR). |
| `plan/TASK_B_calibration.md` | Task B: pitch keypoint network for camera calibration (NBJW HRNet). |
| `plan/TASK_C_reid_team_role.md` | Task C: re-ID, team and role network (PRTReID). |
| `plan/TASK_D_jersey.md` | Task D: jersey number networks (full plan for after the sprint; sprint steps are in `00_SPRINT.md`). |
| `agents/AGENT_TASK_*.md` | One brief per task to give your coding agent (Claude Code, Cursor, ...). |
| `agents/AGENT_INTEGRATION.md` | Brief for the integration owner (baseline, packs, pipeline runs, test). |
| `tools/` | Helper scripts already tested on the team PC (see `tools/README.md`). Copy them into `sn-gamestate/tools/`. |
| `jobs/template.sh` | SLURM job template for the fep A100 queue. |

## How to use it with your agent

1. Clone the team's `sn-gamestate` repository and copy `tools/` and `jobs/` from this folder into it.
2. Put this whole folder next to the repo (or inside it as `handoff/`).
3. Start your agent in the repo and give it this as the first message (replace X with your task letter):

```
Read handoff/gsr_team_plan/agents/AGENT_TASK_X.md and follow it. It points to the plan files
you must read first. Work step by step, run each step's check before moving on, and stop and
tell me when a step fails or the plan does not cover the situation. Do not change the data
splits, the fixed training recipe or another task's code.
```

If your agent supports a project instruction file (`CLAUDE.md`, `AGENTS.md`, `.cursorrules`),
you can copy your `AGENT_TASK_X.md` into it instead.

## Facts checked on the team PC (Nov 2026)

- Environment: Python 3.9, torch 1.13.1 + CUDA 11.7, mmcv 2.0.1, tracklab 1.3.24, ultralytics 8.4.211, timm 0.9.16.
- Dataset: 57 train / 58 valid / 49 test / 36 challenge clips (750 frames each, 1920x1080); 3 matches per labelled split.
- TrackLab's built-in dataset download points at a removed link; use `tools/get_gsr.py` (Hugging Face).
- Baseline on one validation clip (SNGS-021): GS-HOTA 32.3, GS-DetA 20.3, GS-AssA 51.4; ~10 minutes per clip.
