# Team rules

Each member trains and improves one network of the pipeline. These rules are the same for everyone; change them only by team decision. Read this file before your first training run.

## Who does what

| Task | Network you train | Plan | Owner |
| --- | --- | --- | --- |
| A | Person detector (YOLO11, RT-DETR) | `TASK_A_detector.md` | to assign |
| B | Pitch keypoint network for camera calibration (NBJW HRNet) | `TASK_B_calibration.md` | to assign |
| C | Re-ID, team and role network (PRTReID) | `TASK_C_reid_team_role.md` | to assign |
| D | Jersey frame network and track aggregator | `TASK_D_jersey.md` (sprint steps in `00_SPRINT.md`) | to assign |
| Integration | No network: baseline runs, data packs, saved tracker states, the final test run | this file | to assign |

Each owner follows their file step by step, in order. If a step fails or is unclear, stop and ask in the team chat before improvising.

## Data rules

All labelled clips come from only 9 matches, 3 per split. Clips of one match share kits, stadium, camera and players, so any mixing between splits inflates our scores.

| Split | Folder | Matches (game_id) | Clips | Use |
| --- | --- | --- | --- | --- |
| train | `data/SoccerNetGS/train` | 4, 6, 9 | 57 | training only |
| valid | `data/SoccerNetGS/valid` | 2, 3, 5 | 58 | choosing the best epoch and the best experiment; all ablations |
| test | `data/SoccerNetGS/test` | 7, 8, 11 | 49 | once, at the end, run by the integration owner |
| challenge | `data/SoccerNetGS/challenge` | hidden | 36 | not used: no labels |

1. Train only on `train`. No exceptions, not even "a few" validation frames.
2. Never move, copy, rename or re-split clips, and never put frames from different splits in one folder. Converted datasets (YOLO, calibration, crops) keep a separate `train` and `valid` folder built from the matching split. `tools/pack_gsr.py` already does this.
3. Need more data? Ask in the team chat. The answer is an external dataset named in your task file, never `valid` or `test`.
4. `valid` is for decisions: best epoch, best experiment, early stopping. It is never used for training.
5. `test` is used once, at the end, by the integration owner, for every task's final model at the same time. The one exception is the unchanged baseline, run once on `test` during the sprint to compare with the paper's 22.26 GS-HOTA; it chooses nothing. Nobody else runs on `test` before the end, and `test` is never copied to fep.
6. We do not retrain on `train` + `valid` at the end. Every reported model is trained on `train` only.
7. External datasets: only the ones named in your task file, and only after its overlap check step is done and written in the tracker.
8. Fixed ablation subset for every pipeline run, never changed: SNGS-021, SNGS-026, SNGS-031, SNGS-036 (match 2), SNGS-039, SNGS-046, SNGS-053 (match 3), SNGS-078, SNGS-085, SNGS-092 (match 5).

## Training rules

1. Each network has one fixed recipe (optimizer, learning rate and schedule, batch size, epoch budget), written in its task file (sprint overrides in `00_SPRINT.md`). Do not tune it. If a run diverges or runs out of memory, tell the team before changing anything.
2. One experiment changes exactly one thing compared with a named reference run. Never two changes at once.
3. Not allowed by the course: swapping normalization layers, swapping activation functions, adding or removing a single layer, comparing optimizers or learning rates.
4. Seed 0 for every run. At the end (after the sprint), repeat your best run and its reference with seeds 1 and 2, and report mean ± standard deviation.
5. Choose the epoch by the validation metric (early stopping). Never look at `test` to choose anything.
6. Every run saves: its config, training loss and validation loss per epoch, the validation metric per epoch, and the best checkpoint. Plot training vs validation curves; the report needs them for every network.
7. Name runs `<task><number>_<change>`, for example `A3_rtdetr_l`. Use that name for the run folder, the checkpoint and the tracker row.
8. Write every run in the results tracker on the day it ends, failed runs included, with the reason.

## GPUs and pipeline runs

Training runs on the shared fep A100 queue (SLURM, 12-hour job limit, long waits; setup in `00_SPRINT.md`). The 2 RTX A4500 GPUs are on the integration owner's PC, used only by them for pipeline runs. The environment on both is `conda activate tracklab`.

1. On the A4500 PC, start every command with `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1`. On fep, every run is a SLURM job with a time limit under 12 hours.
2. Run long jobs on the PC inside tmux: `tmux new -s <run name>`, start the command, detach with Ctrl+B then D, reattach with `tmux attach -t <run name>`.
3. Keep the PC from sleeping (Windows power settings) during runs.
4. A full pipeline run takes about 10 minutes per clip, so ~1h45 for the 10 ablation clips. Run the pipeline only for the best model of each experiment group, not for every training run.
5. Keep it to a few GPU processes per card (2–3). On Nov 14, 5–6 processes per card made the NVIDIA driver fail (FECS exceptions) and Windows crashed (bugcheck 0x113, `VIDEO_DXGKRNL_FATAL_ERROR`). GPU 0 also drives the Windows display: cap jobs there so Windows keeps some GPU memory (e.g. `--gpu_mem_frac 0.6` in `tools/train_jersey.py`, or `torch.cuda.set_per_process_memory_fraction`).

**Pipeline evaluation.** On day 1 the integration owner runs the baseline once on the 10 ablation clips and shares the saved state file (`outputs/.../states/sn-gamestate.pklz`). Reruns load it and run only the changed module and everything after it in the `pipeline:` list of `sn_gamestate/configs/soccernet.yaml`:

| Task | Pipeline rerun | Taken from the saved state |
| --- | --- | --- |
| A | the whole pipeline | nothing |
| B | `pitch` to the end | detector, re-ID, tracking |
| C | `reid` to the end | detector |
| D | `jersey_number_detect` to the end | everything before it |

Command shape (the integration owner adds the exact command here after the baseline run):

```
CUDA_VISIBLE_DEVICES=0 tracklab -cn soccernet dataset.nvid=-1 \
  'dataset.vids_dict.valid=[SNGS-021,SNGS-026,SNGS-031,SNGS-036,SNGS-039,SNGS-046,SNGS-053,SNGS-078,SNGS-085,SNGS-092]' \
  state.load_file=<baseline .pklz> 'pipeline=[<your modules>]' \
  visualization.cfg.save_videos=False
```

Baseline so far: GS-HOTA 32.3 on one clip (SNGS-021). The 10-clip baseline is the integration owner's first job.

**Known environment quirk (WSL PC only):** if cuDNN fails with `libcuda.so: cannot open shared object file`, the env needs `LD_LIBRARY_PATH=/usr/lib/wsl/lib` (already set in the PC's `tracklab` env).

## Results tracker

One row per run, added the day it ends. Higher is better for every metric here unless the column says otherwise. Keep it in a shared sheet; the starting rows are:

| Run | Task | Change vs reference | Reference run | Component metric (valid) | GS-HOTA, 10 clips | GPU | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BASE | Integration | baseline pipeline, all modules as shipped | none | none | | | Planned |
| A0 | A | COCO `yolo11m.pt`, no fine-tuning | none | | | | Planned |
| B0 | B | NBJW pretrained keypoint weights | none | | | | Planned |
| C0 | C | provided PRTReID checkpoint | none | | | | Planned |
| D0 | D | MMOCR and majority vote | none | | | | Planned |

Status is one of: Planned, Running, Done, Failed.
