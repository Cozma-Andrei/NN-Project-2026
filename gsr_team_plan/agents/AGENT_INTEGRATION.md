# Agent brief — Integration owner

You help the integration owner: the only person with the 2× RTX A4500 PC that holds the full dataset
(`~/sn-gamestate/data/SoccerNetGS`, all splits) and runs every TrackLab pipeline (GS-HOTA) evaluation.
No network is trained in this role. Environment: `conda activate tracklab` (on WSL, `LD_LIBRARY_PATH=/usr/lib/wsl/lib` is set in the env).

## Read first

1. `handoff/gsr_team_plan/plan/00_SPRINT.md` (Integration row of the day-by-day table)
2. `handoff/gsr_team_plan/plan/01_TEAM_RULES.md` (GPUs and pipeline runs; the results tracker)
3. `handoff/gsr_team_plan/tools/README.md`

## Jobs

**Day 1**
- Commit `tools/` and `jobs/` to the team repo and push it.
- Packs are built in `data/packs/{A,B,D}` (check `MANIFEST.json`); rsync each to fep, plus `pretrained_models/`. Never copy `test`.
- Baseline on the 10 ablation clips, on GPU 0 inside tmux, with state saving on (default `state.save_file`):
  `CUDA_VISIBLE_DEVICES=0 tracklab -cn soccernet dataset.nvid=-1 'dataset.vids_dict.valid=[SNGS-021,SNGS-026,SNGS-031,SNGS-036,SNGS-039,SNGS-046,SNGS-053,SNGS-078,SNGS-085,SNGS-092]' visualization.cfg.save_videos=False`
  Record GS-HOTA / DetA / AssA in the tracker (row BASE) and the `.pklz` path in `01_TEAM_RULES.md`. Verify the exact `state.load_file` + `pipeline=[...]` rerun command once and write it into the Team rules.

**Day 2**
- D0: run `sn_gamestate/jersey/mmocr_api.py`'s `MMOCR` on pack D valid crops, aggregate per track with `tracklab.utils.attribute_voting.select_highest_voted_att`, score with Task D's `tools/eval_jersey.py` metric.
- B0: help Task B run `tools/eval_calib.py` on the pretrained weights if fep is busy.
- Baseline on test, once (the only test run before the end; it chooses nothing): `CUDA_VISIBLE_DEVICES=1 tracklab -cn soccernet dataset.eval_set=test dataset.nvid=-1 visualization.cfg.save_videos=False`, all 49 clips in ONE process overnight (~8.5 h; GS-HOTA must be computed over all clips together). Compare with the paper's 22.26.
- Attribute breakdown of the baseline (10 ablation clips and test): re-score the saved state with `state.load_file=<.pklz> 'pipeline=[]'` plus, under `dataset.track_eval.`: row 1 `EVAL_SPACE=image USE_ROLES=False USE_TEAMS=False USE_JERSEY_NUMBERS=False`; row 2 `USE_ROLES=False USE_TEAMS=False USE_JERSEY_NUMBERS=False`; row 3 `USE_TEAMS=False USE_JERSEY_NUMBERS=False`; row 4 `USE_JERSEY_NUMBERS=False`; row 5 none. The drop between rows shows what each part costs (table in `00_SPRINT.md`, 'Error analysis').
- Run any training job still stuck in the fep queue on GPU 1 overnight.

**Day 3**
- Pipeline checks, from the saved baseline state: best A (full pipeline, `modules.bbox_detector.cfg.path_to_checkpoint=...`), best B (from `pitch`; find the keypoint checkpoint key in `sn_gamestate/configs/modules/pitch/nbjw_calib.yaml`), best C (from `reid`, `modules.reid.cfg.model.load_weights=...`).
- Complete the results tracker.

## Rules

- Pipeline runs always use the full-resolution dataset, never `data/packs/`.
- `test` is run once for the unchanged baseline (day 2) and once at the very end for every task's final model together; never for choosing anything.
- Never make a 'downsized' GS-HOTA baseline (fewer frames or smaller images): to save time, use fewer clips, never fewer frames or pixels.
- Disk: the full dataset is 37 GB; full pipeline videos are ~100 MB per clip, so keep `save_videos=False` except for presentation clips.
