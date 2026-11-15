# Helper scripts (tested on the team PC, Nov 2026)

Copy these into `sn-gamestate/tools/` and run them from the repository root (`conda activate tracklab`).

| Script | What it does | Tested |
| --- | --- | --- |
| `setup_data.sh` | One command for the data: downloads `train` + `valid` if missing, builds the packs you name (`A`, `B`, `D`, or `none`), and checks clip, frame and crop counts (and pack D's `labels.csv` checksum) against the team's copy. `PACK_ONLY=1` checks a pack received as an archive. See `../DATA_SETUP.md`. | all modes run on the team PC |
| `get_gsr.py` | Downloads SoccerNet-GSR splits from Hugging Face (`SoccerNet/SN-GSR-2025`) and unzips them into TrackLab's layout. Use it instead of TrackLab's built-in download, whose link is dead. `python tools/get_gsr.py <data_dir> train valid` | full download of all 4 splits (57/58/49/36 clips) |
| `pack_gsr.py` | Builds small training packs from the full dataset: `--task A` (YOLO, 1280×720, 1-class and 4-class labels), `--task B` (NBJW calibration format, 960×540, plus athlete foot points/pitch positions), `--task D` (full-resolution player/goalkeeper crops + `labels.csv`). Only `train` and `valid`, each in its own folder; refuses to overwrite a non-empty pack. Options: `--train_stride`, `--valid_stride`, `--max_clips` (debug), `--workers`. | all three packs built; pack A trained 1 YOLO epoch (1-class and 4-class); pack B loaded with NBJW's own loader; labels drawn and checked visually |
| `nbjw_compat.py` | `import tools.nbjw_compat` before importing `nbjw_calib.model.transforms`: NBJW imports `torchvision.transforms.v2`, missing in torchvision 0.14; this maps it to `torchvision.transforms` (only `Compose` is used). | NBJW train/valid presets run on pack B |

Packs are for **training and component metrics only**. Never run the TrackLab pipeline (GS-HOTA) on them.

Pack sizes with default strides: A 1.8 GB (4275 train / 4350 valid frames), B 1.1 GB (same frames), D 0.9 GB (125,677 / 64,518 crops).

## Task D scripts (smoke-tested on the team PC)

| Script | What it does |
| --- | --- |
| `jersey_lib.py` | Shared: pack-D dataset (top 60% of each crop, no flip), `JerseyNet` (timm backbone + digit heads or 101-way head), fixed track aggregation rule, track metrics. |
| `train_jersey.py` | Trains D1–D3 with the fixed recipe; saves `history.csv`, `curves.png`, `last.pt` every epoch (`--resume`), `best.pt`, per-track predictions. `--smoke` for a 20-batch check; `--gpu_mem_frac 0.6` on the GPU that drives the Windows display. `--train_keep <csv>` trains on a filtered train list (E9, runs D5/D6); `--adv` adds the gradient-reversal adversary (D4). |
| `eval_jersey.py` | Scores a checkpoint or a saved track table: accuracy all / with number / none / numbers unseen in train, breakdowns by track length, box height, match and error type, plus failure strips. |
| `make_e9_keep.py` | E9: builds the teacher-filtered train list (`train_keep.csv`) from the MMOCR teacher pass (`d0_mmocr.py --split train`). |
| `d0_mmocr.py` | Baseline D0: the pipeline's MMOCR reader + today's vote on the same crops. Slow (one image at a time): split with `--num_shards N --shard i`, at most 1–2 shards per GPU, then `--merge`. |
