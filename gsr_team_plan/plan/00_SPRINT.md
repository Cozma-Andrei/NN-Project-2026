# Sprint: first results in 3 days

In 3 days every task delivers one trained reference model and 2–3 experiments, each with curves and its component metric on valid. GS-HOTA is measured only for the baseline and each task's best model. The task files stay the full plan for after the sprint; this file says which parts of them to do now, and **it overrides the task files where they differ**.

## Compute

Training goes to the fep A100 queue; the two A4500s run the pipeline, because the data, environment and saved states are already there.

| Resource | Who uses it | Used for |
| --- | --- | --- |
| fep A100 queue (SLURM, 12-hour job limit, long waits) | everyone | all training runs |
| 2× RTX A4500 on the integration owner's PC | integration owner only | baseline and all pipeline evaluation runs (~1h45 per run on the 10 ablation clips); fallback for a job stuck in the queue |

**Queue rules**

1. A failed job costs a full queue wait. Before submitting, run your script for 20 batches on your own machine (CPU is fine) and check that the loss goes down and a checkpoint is written.
2. Submit all runs of your task at once, as separate jobs. They do not depend on each other.
3. Every job must finish in under 10 hours, leaving margin under the 12-hour limit. Save a checkpoint every epoch so a killed job can be resumed.
4. The test split is never copied to fep (Team rules, data rule 5).

## fep setup (day 1, one person, once)

1. **Code.** Clone the team's copy of `sn-gamestate` (pushed by the integration owner, including the `tools/` folder) into your fep home.
2. **Environment**, the same as on the A4500 PC (Python 3.9, torch 1.13.1 with CUDA 11.7, which supports the A100):

```
conda create -n tracklab pip python=3.9 pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate tracklab
cd sn-gamestate
pip install -e plugins/calibration
pip install -e .
mim install mmcv==2.0.1
pip install timm==0.9.16
```

Check: `python -c "import torch, mmcv, tracklab, sn_gamestate, timm; print(torch.cuda.is_available())"` prints `True` inside a GPU job.

3. **Data.** fep only needs training data, because every pipeline run happens on the A4500 PC with the full dataset. The integration owner builds a small pack per task on the PC with `python tools/pack_gsr.py --task A|B|D` and copies it with `rsync -av data/packs/<task>/ <user>@fep:<path>/sn-gamestate/data/packs/<task>/` (`-a` keeps the links inside the A pack). After copying the A pack, change `path:` in `gsr1.yaml` and `gsr4.yaml` to the fep path. Task C trains through TrackLab, which needs the full layout: on fep run `python tools/get_gsr.py <data folder> train valid` (23 GB) and add `data_dir=<data folder>` to TrackLab commands. Check: each pack's `MANIFEST.json` counts match the table below; `get_gsr.py` prints 57 clips for train and 58 for valid.

| Pack | Content | Frames or crops |
| --- | --- | --- |
| A | train and valid every 10th frame, resized to 1280×720 (lossless for `imgsz=1280`), YOLO labels for 1 class (`gsr1.yaml`) and 4 role classes (`gsr4.yaml`) | 4275 train, 4350 valid frames; 1.8 GB |
| B | train and valid every 10th frame with pitch labels, resized to 960×540 (what NBJW does first anyway), one `.json` of lines per image, plus `athletes/` with foot points in original pixels and true pitch positions for the position-error metric | 4275 train, 4350 valid frames; 1.1 GB |
| D | player and goalkeeper crops cut from the full 1080p frames (never from downscaled frames, or numbers become unreadable), train every 5th frame, valid every 10th, boxes at least 20 px tall, plus `labels.csv` (clip, match, track, frame, role, jersey, team, box height) | 125,677 train crops (1224 tracks, 291 without a number), 64,518 valid crops (1220 tracks, 254 without); 0.9 GB |

**Packs are for training and component metrics only.** Never run the TrackLab pipeline on resized or subsampled frames: every module (detector, re-ID, OCR) would see different inputs and GS-HOTA would be wrong.

4. **Model weights.** Copy `pretrained_models/` (1.1 GB) from the PC with `rsync -av pretrained_models/ <user>@fep:<path>/sn-gamestate/pretrained_models/`.
5. **Job template.** `jobs/template.sh` (in this package): copy it per run, change the name and the command. Fill in the A100 partition name from the fep docs.

Submit with `sbatch jobs/<run>.sh`, watch with `squeue -u $USER`, read `logs/<run>_<id>.out`.

## Sprint scope per task

Each owner follows their task file's steps with the changes below, and submits the listed runs together. Everything else in the task files waits until after the sprint.

| Task | Steps to follow | Changes for the sprint | Runs to submit together | Delivered at the end |
| --- | --- | --- | --- | --- |
| A | TASK_A steps 1–9 | skip steps 2–4: use pack A (`gsr1.yaml` / `gsr4.yaml`); step 5: `epochs=15 patience=5` | A1 `yolo11m`; A2b `yolo11l`; A3 `rtdetr-l`, compared with A2b; A7 4 role classes (`gsr4.yaml`), the auxiliary-task run: its per-class AP comes from `yolo val`, no pipeline change needed | mAP, recall by size for A0, A1 and the best run; per-class AP for A7; curves |
| B | TASK_B steps 1–7 | skip step 2: use pack B as the dataset root (`root_dir` = the pack folder); step 5: 6 epochs, early stopping after 3 | B1 `MSELoss`; B2b `CombMSEAW`; B4a close-up zoom | position error and within-5 m share, wide vs close-up, for B0, B1 and the best run; curves |
| C | TASK_C steps 1–7 (skip step 8) | the reference is a fine-tune from the provided checkpoint (keep `load_weights`), `train.max_epoch=5`; call it C-ref | C-ref (focal); C5b `focal_weighted`; C5a `ce`; C4a `role_weight=0` | role F1 per class, team accuracy, re-ID mAP; curves |
| D | the Task D sprint steps below | uses pack D (full-resolution crops); no pipeline module yet | D1 ResNet18 with digit heads; D2 ViT-S with digit heads; D3 ResNet18 with one 101-way head | track accuracy (all, with number, none) for D0 and each run; curves |

The integration owner runs the pipeline check (Team rules) for the best run of A, B and C on day 3. D gets its pipeline module after the sprint.

## Task D sprint steps

The sprint trains the frame network on pack D and scores it on ground-truth tracks; the pipeline module and the learned aggregator come after the sprint.

1. **Read pack D.** Load `data/packs/D/labels.csv` with pandas. Each row is one crop; `jersey` is the track's number (empty = the track never shows a number). The number is given on every frame of the track, even frames where it cannot be seen: that is the weak label, and step 4 handles it.
   - Check: train and valid row counts match `MANIFEST.json`; valid contains only clips SNGS-021 to SNGS-096 (matches 2, 3, 5).
2. **Targets.** Digit-head runs (D1, D2): `has_number` (2 classes), `tens` (11 classes: 0–9, or "blank" for one-digit numbers), `units` (10 classes). D3: one head with 101 classes (numbers 0–99 plus "none").
3. **Write `tools/train_jersey.py`.** Input: the top 60% of each crop (upper body), resized to 224×224. Augmentation: random resized crop with scale 0.8–1.0 and colour jitter (brightness and contrast 0.2). **No horizontal flip**: it mirrors digits. Model: `timm.create_model(<name>, pretrained=True, num_classes=0)` plus one linear layer per head. Loss: cross-entropy on `has_number` for every crop, plus cross-entropy on `tens` and `units` for crops whose track has a number (D3: one cross-entropy over 101 classes). **Fixed recipe:** AdamW, learning rate 1e-4, weight decay 0.05, cosine decay, 8 epochs, batch 128, seed 0. Each epoch, log training loss, validation loss and the track accuracy from step 4; keep the checkpoint with the best track accuracy.
   - Check: 20 batches on your machine; the loss goes down and a checkpoint is written.
4. **Write `tools/eval_jersey.py`** (also called inside training). For every valid track: keep the frames with p(has_number) > 0.5. No frame kept → predict "none". Otherwise, for every number, add up its probability over the kept frames (digit heads: p(tens) × p(units)) and predict the largest. The 0.5 threshold is fixed, not tuned. Report track accuracy on all tracks, on tracks with a number, and on tracks without one.
5. **D0, the baseline on the same tracks**, run on the A4500 PC by the integration owner: the `MMOCR` class from `sn_gamestate/jersey/mmocr_api.py` on the valid crops of pack D, aggregated with `select_highest_voted_att` (today's majority vote, in `tracklab/utils/attribute_voting.py`), scored with step 4's metric.
6. **Submit D1** (`resnet18`, digit heads), **D2** (`vit_small_patch16_224`, digit heads) and **D3** (`resnet18`, 101-way head) together.
7. **Deliver:** the three track accuracies for D0–D3, curves, and five failure tracks saved as crop strips with one line each on why they failed.

## Error analysis (required for the presentation)

The course asks us to explain failures through the data, not only to show example images, so every result comes with a breakdown and a sentence on which dataset property causes the failure.

**Where the baseline loses GS-HOTA (integration owner, day 2).** Re-score the saved baseline state without rerunning any module: `state.load_file=<baseline .pklz> 'pipeline=[]'`, adding one row's overrides each time (all under `dataset.track_eval.`). Each row takes minutes. Do it for the 10 ablation clips and for test.

| Row | Overrides | What the gap to the next row shows |
| --- | --- | --- |
| 1 Detection and tracking only | `EVAL_SPACE=image USE_ROLES=False USE_TEAMS=False USE_JERSEY_NUMBERS=False` (standard HOTA) | detector and tracker (Tasks A, C) |
| 2 + positions on the pitch | `USE_ROLES=False USE_TEAMS=False USE_JERSEY_NUMBERS=False` | calibration (Task B) |
| 3 + role | `USE_TEAMS=False USE_JERSEY_NUMBERS=False` | role (Task C) |
| 4 + team | `USE_JERSEY_NUMBERS=False` | team (Task C) |
| 5 Full GS-HOTA | none | jersey numbers (Task D) |

**Per task**, each linked to a dataset property from Team rules:

- A: recall by box height (small, distant players); per-class AP from A7 (rare goalkeepers and "other").
- B: position error on close-ups vs wide shots (frames with few visible lines).
- C: role F1 per class (player 605k vs other 1.4k boxes); team accuracy per clip (kits differ per match).
- D: accuracy on tracks with vs without a number (24% have none); track accuracy by track length (fragmented tracks).

**Reporting the sprint results honestly:** one seed and short training, so differences of 1–2 points may be noise. Label them preliminary; the best run of each task gets seeds 1 and 2 before the final.

## Which baseline to compare against

| Measure | Downsized how | Baseline row |
| --- | --- | --- |
| Component metrics (mAP, position error, role F1, jersey track accuracy) | packs: every 5th/10th frame, resized only where lossless | A0, B0, C0, D0, computed on the **same packs** as the experiments |
| GS-HOTA during development | **fewer clips** (the 10 ablation clips), all frames, full resolution | BASE on those 10 clips |
| GS-HOTA for the paper | none | baseline on test, run once |

**Never run the pipeline on fewer frames or smaller images to save time.** GS-HOTA scores every one of the 750 frames (missing frames count as misses), StrongSORT's motion model assumes consecutive frames, and smaller images degrade every module, so the numbers would measure the downsizing and could not be compared with the paper or the state of the art. To make GS-HOTA cheaper, use fewer clips, never fewer frames or pixels.

## Day by day

Every training job is submitted by the end of day 1 (C by day 2 morning), so queue waits overlap with writing evaluation scripts.

| Who | Day 1 | Day 2 | Day 3 |
| --- | --- | --- | --- |
| Integration owner | push the repo with `tools/`; copy packs A, B, D to fep; set up the fep environment; baseline pipeline on the 10 ablation clips (GPU 0, ~1h45, state saved) | D0 (MMOCR on pack D) and B0 on the PC; baseline on test, once (all 49 clips on GPU 1 overnight, ~8.5 h), to compare with the paper's 22.26; attribute breakdown of the baseline (see Error analysis) | pipeline checks for the best A (full pipeline), B (from `pitch`) and C (from `reid`) runs; tracker complete |
| A | job files for A1, A2b, A3, A7 from the template; smoke test (20 batches, `yolo11n`, `imgsz=320`); submit all four | `tools/recall_by_size.py`; score finished runs; per-class AP for A7 | tables, curves, five failure images |
| B | `tools/train_kp.py` and smoke test; submit B1, B2b, B4a | `tools/eval_calib.py`; score finished runs; wide vs close-up split | tables, curves, five failure frames |
| C | steps 2–4; download train and valid on fep; submit the 1-epoch smoke job | if the smoke job passed, submit C-ref, C5b, C5a, C4a; write `tools/eval_reid_attrs.py` | score finished runs; tables, curves, five failure cases |
| D | steps 1–4 and smoke test; submit D1, D2, D3 | score finished runs against D0 | tables, curves, five failure tracks |

**If a job fails or waits too long:** fix and resubmit once. If it still has not started by day 2 evening, the integration owner runs it on GPU 1 of the PC overnight instead.
