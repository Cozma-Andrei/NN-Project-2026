# Task B: Pitch keypoints and calibration

Fine-tune the pitch keypoint network that calibrates the camera, then study heatmap losses, backbones, a shared multi-task network, temporal input and close-up augmentation.

> **Sprint override (`00_SPRINT.md`):** skip step 2 and use pack B as the dataset (`root_dir` = the pack folder with a trailing `/`: 960×540 images, one lines `.json` each, every 10th frame, already built and checked with the real loader). Recipe: 6 epochs, early stopping after 3. Runs: B1 `MSELoss`, B2b `CombMSEAW`, B4a close-up zoom. Training runs as fep SLURM jobs. Step 8 (pipeline check) is done by the integration owner.

## Starting point

- **Today's networks:** two HRNetV2-W48 heatmap networks from "No Bells, Just Whistles" (NBJW): a keypoint network (58 channels, `plugins/calibration/nbjw_calib/config/hrnetv2_w48.yaml`) and a line network (`hrnetv2_w48_l.yaml`). Their weights (`pretrained_models/calibration/SV_kp`, `SV_lines`, downloaded from Zenodo) were trained on SoccerNet Calibration, never on GSR. The camera is then solved from the keypoints with no learning (`NBJW_Calib` in `sn_gamestate/calibration/nbjw_calib.py`). The pipeline resizes every frame to 960×540 first (`tfms_resize`, line ~105).
- **Scope:** you train the keypoint network. The line network stays at its pretrained weights in every experiment except B5.
- **Code that exists** in `plugins/calibration/nbjw_calib/`: the model (`model/cls_hrnet.py`), a dataset loader (`model/dataloader.py`, class `SoccerNetCalibrationDataset`: one `.jpg` plus one `.json` of lines per image), transforms (`model/transforms.py`: ToTensor, Normalize, RandomHorizontalFlip, LRAmbiguityFix, ColorJitter, AddGaussianNoise, Resize, plus presets `transforms`, `no_transforms`), losses (`model/losses.py`: MSELoss, HeatmapWeightingMSELoss, CombMSEAW = MSE + adaptive wing, KLDivLoss) and metrics (`model/metrics.py`). **There is no training script**; you write one in step 5.
- **Known incompatibility (fixed):** `model/transforms.py` imports `torchvision.transforms.v2`, which torchvision 0.14 does not have. `import tools.nbjw_compat` before importing it; then the presets work (checked).
- **Labels:** every GSR frame has pitch-line labels (`supercategory: pitch`, field `lines`, coordinates from 0 to 1), the same format as SoccerNet Calibration. Every athlete also has its true pitch position in metres (`bbox_pitch`), which gives a direct metric. Pack B stores these per clip in `athletes/<split>_<clip>.json` (foot point in original 1920×1080 pixels, pitch position in metres, number of labelled lines).

| Metric (on valid) | Direction |
| --- | --- |
| Validation heatmap loss and keypoint accuracy (`calculate_metrics` in `model/metrics.py`) | loss lower, accuracy higher |
| Athlete position error: foot point projected with the predicted camera vs the true pitch position, in metres | lower is better |
| Share of athletes within 5 m (the GS-HOTA tolerance) | higher is better |
| Share of frames that get a camera at all | higher is better |
| GS-HOTA and GS-DetA on the 10 ablation clips | higher is better |

## Step by step

Do the steps in order. Each step ends with a check; do not start the next step until the check passes.

1. **Read `01_TEAM_RULES.md`.** Then `conda activate tracklab`, `cd ~/sn-gamestate`, and make sure `timm==0.9.16` is installed (needed for B3).
   - Check: `python -c "import tools.nbjw_compat, timm, kornia, nbjw_calib"` prints nothing.
2. **(Done by `tools/pack_gsr.py --task B` — skip in the sprint.)** For every 10th frame with `has_labeled_pitch: true`: image resized to 960×540 as `<split>/<clip>_<frame>.jpg` plus `<clip>_<frame>.json` with only that frame's `lines` dict.
   - Check: `MANIFEST.json` shows 4275 train and 4350 valid frames. The path must contain no dot except the file extension, because the loader cuts the path at the first `.` to find the `.json`.
3. **Load the data with the existing loader.** In Python: `import tools.nbjw_compat` first, then `ds = SoccerNetCalibrationDataset(root_dir='<abs path>/data/packs/B/', split='train', transform=transforms, main_cam_only=False)` and `ds.files = ds.get_image_files(rate=1)` (the loader would otherwise keep only every 3rd file). `main_cam_only` must be False: the other mode needs a `match_info.json` we do not have. Use the preset `transforms` for train and `no_transforms` for valid (from `nbjw_calib.model.transforms`).
   - Check: one sample gives image `(3, 540, 960)`, target `(58, 270, 480)` with peak 1.0 and a mask with ~56 visible keypoints (verified on the PC); plot 5 targets over their images — peaks sit on line intersections and pitch landmarks.
4. **B0, the pretrained network on valid.** Write `tools/eval_calib.py`. For every valid image: run the keypoint and line networks and solve the camera exactly as `NBJW_Calib_Keypoints.process` and `NBJW_Calib.process` do in `sn_gamestate/calibration/nbjw_calib.py` (copy those steps; do not rewrite them). Then compute the four component metrics from the Starting point table, using the foot points and pitch positions in `athletes/`. Foot points are in 1920×1080 pixels, so solve the camera for that image size.
   - Check: numbers for B0 in the tracker.
5. **Write `tools/train_kp.py`.** Build the keypoint model like `NBJW_Calib_Keypoints.__init__` does (config + `get_cls_net` + the `SV_kp` weights), the loaders from step 3, and the loss. **Fixed recipe:** AdamW, learning rate 1e-4, cosine decay to 0, 20 epochs (sprint: 6), batch 4, early stopping after 5 epochs (sprint: 3) without a better validation loss, seed 0, reference loss `MSELoss`. Log training loss, validation loss and validation keypoint accuracy every epoch; save the best checkpoint to `runs/calibB/<run>/best.pt` and a checkpoint every epoch.
   - Check: run 50 batches; the training loss goes down.
6. **Time one epoch** and write it in the tracker. If the run would exceed 10 hours (fep limit), use fewer epochs and note it.
7. **B1, the first fine-tuning run** (fep job): start from `SV_kp`, reference recipe.
   - Check: curves saved; `tools/eval_calib.py` with B1 weights gives the four metrics; tracker filled.
8. **Pipeline check of B1** (integration owner). Point the pipeline's keypoint checkpoint to your `best.pt` (the integration owner finds the config key in `sn_gamestate/configs/modules/pitch/nbjw_calib.yaml`) and rerun from `pitch` on the 10 ablation clips using the saved baseline state.
   - Check: GS-HOTA and GS-DetA in the tracker.
9. **Run the experiments** in the table below, in order. Each repeats step 7 with one change; step 8 only for the best run of each group.
10. **Finish:** best run and B1 repeated with seeds 1 and 2; report mean ± std, curves, and position error split into wide shots and close-ups.

## Experiments

Run them top to bottom. The reference is B1 unless the row says otherwise; change only what the row names.

| Run | Group | Change | Failure it targets |
| --- | --- | --- | --- |
| B2a–B2c | Loss | `HeatmapWeightingMSELoss`; `CombMSEAW`; `KLDivLoss` (all in `model/losses.py`) | keypoints a few pixels off become metres of error far from the camera |
| B3a, B3b | Convolution vs attention | three timm backbones (`features_only=True`, ImageNet weights) with the same small upsampling head and 58 outputs, input resized to 544×960: B3-ref `hrnet_w48`, B3a `resnet50`, B3b `swin_tiny_patch4_window7_224` (create it with `img_size=(544, 960)`; its features come channels-last, so permute them). All three were checked to run at this size. Reference is B3-ref, not B1 | missed far-side keypoints |
| B4a, B4b | Augmentation | add a close-up transform: random zoom of 1.5–3× that crops the image and moves the line points the same way, dropping points that leave the image; then add motion blur on top | close-ups with few visible lines |
| B5 | Multi-task | one shared backbone with two heads (keypoints and lines) trained together, instead of two separate networks; loss = keypoint loss + line loss | auxiliary task: lines help keypoints |
| B6 | Temporal input | 3 consecutive frames stacked as 9 input channels; the first convolution gets 9 channels, initialised with the pretrained weights divided by 3 for each frame (needs consecutive frames: build a new pack with `--train_stride 1`, or read the full dataset) | camera jumps between frames |
| B7 (only if time is left) | Loss | add an athlete reprojection loss: soft-argmax keypoints → homography with `kornia.geometry.homography.find_homography_dlt` (differentiable) → project the frame's ground-truth foot points → L1 distance in metres to the true pitch position | position error in metres |

**Definitions used in the report:** a close-up is a frame whose ground-truth `lines` dict has fewer than 4 lines (`n_lines` in `athletes/`); every other frame is a wide shot.

**Done when:**

- [ ] B0 to B6 are in the tracker with curves saved (sprint: B0, B1, B2b, B4a)
- [ ] Position error and within-5 m share for B0, B1 and the best run, split into wide shots and close-ups
- [ ] Pipeline GS-HOTA for the best run of each group
- [ ] Best run and B1 repeated with seeds 1 and 2 (after the sprint)
- [ ] Five failure frames from the best model saved with predicted keypoints drawn, with one line each on why they failed
