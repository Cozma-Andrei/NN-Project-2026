# Helper scripts (tested on the team PC, Nov 2026)

Copy these into `sn-gamestate/tools/` and run them from the repository root (`conda activate tracklab`).

| Script | What it does | Tested |
| --- | --- | --- |
| `get_gsr.py` | Downloads SoccerNet-GSR splits from Hugging Face (`SoccerNet/SN-GSR-2025`) and unzips them into TrackLab's layout. Use it instead of TrackLab's built-in download, whose link is dead. `python tools/get_gsr.py <data_dir> train valid` | full download of all 4 splits (57/58/49/36 clips) |
| `pack_gsr.py` | Builds small training packs from the full dataset: `--task A` (YOLO, 1280×720, 1-class and 4-class labels), `--task B` (NBJW calibration format, 960×540, plus athlete foot points/pitch positions), `--task D` (full-resolution player/goalkeeper crops + `labels.csv`). Only `train` and `valid`, each in its own folder; refuses to overwrite a non-empty pack. Options: `--train_stride`, `--valid_stride`, `--max_clips` (debug), `--workers`. | all three packs built; pack A trained 1 YOLO epoch (1-class and 4-class); pack B loaded with NBJW's own loader; labels drawn and checked visually |
| `nbjw_compat.py` | `import tools.nbjw_compat` before importing `nbjw_calib.model.transforms`: NBJW imports `torchvision.transforms.v2`, missing in torchvision 0.14; this maps it to `torchvision.transforms` (only `Compose` is used). | NBJW train/valid presets run on pack B |

Packs are for **training and component metrics only**. Never run the TrackLab pipeline (GS-HOTA) on them.

Pack sizes with default strides: A 1.8 GB (4275 train / 4350 valid frames), B 1.1 GB (same frames), D 0.9 GB (125,677 / 64,518 crops).
