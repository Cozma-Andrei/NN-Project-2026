# Task A: Person detector

Fine-tune the detector on football, then study model size, convolution vs attention, input resolution, augmentation, freezing and an auxiliary role head.

> **Sprint override (`00_SPRINT.md`):** skip steps 2–4 and use pack A (`data/packs/A/gsr1.yaml`, `gsr4.yaml`: every 10th frame, 1280×720, already built and checked). Recipe: `epochs=15 patience=5`. Runs: A1 `yolo11m`, A2b `yolo11l`, A3 `rtdetr-l` (vs A2b), A7 4 role classes (`gsr4.yaml`; per-class AP from `yolo val`, no code change). Training runs as fep SLURM jobs (`jobs/template.sh`), not in tmux on the PC. Step 10 (pipeline check) is done by the integration owner.

## Starting point

- **Today's detector:** COCO-pretrained `pretrained_models/yolo/yolo11m.pt`, never trained on football. The pipeline keeps only class 0 (person) with confidence ≥ 0.4 (`tracklab/wrappers/bbox_detector/yolo_ultralytics_api.py`).
- **Training data:** GSR train boxes: 605k player, 61.5k referee, 24.1k goalkeeper, 1.4k other (41k ball boxes are not used). Median box height is 98 px in 1920×1080 frames; 10% are under 63 px.
- **Tools:** ultralytics 8.4.211 is installed in `tracklab`. It trains YOLO11 and RT-DETR with the same `yolo` command.

| Metric (on valid) | Direction |
| --- | --- |
| mAP50-95 and recall (from `yolo val`) | higher is better |
| Recall by box height: under 50 px, 50–100 px, over 100 px | higher is better |
| GS-HOTA and GS-DetA on the 10 ablation clips | higher is better |

## Step by step

Do the steps in order. Each step ends with a check; do not start the next step until the check passes.

1. **Read `01_TEAM_RULES.md`.** Then `conda activate tracklab` and `cd ~/sn-gamestate`.
   - Check: `yolo version` prints 8.4.211.
2. **(Done by `tools/pack_gsr.py --task A` — skip in the sprint.)** Convert GSR to YOLO format for train and valid separately, class mode 1 (always class 0) and 4 (player 0, goalkeeper 1, referee 2, other 3), skipping balls; labels are `class x_center/1920 y_center/1080 w/1920 h/1080` from `bbox_image`.
   - Check: `data/packs/A/MANIFEST.json` shows 4275 train and 4350 valid frames.
3. **Look at the labels.** Draw the boxes of 5 random train images and open them.
   - Check: every box sits on a person; no boxes on the ball. (Done once on the PC; repeat on fep after copying if in doubt.)
4. **Dataset files.** `data/packs/A/gsr1.yaml` and `gsr4.yaml` exist; set `path:` to the absolute pack path on the machine you train on.
5. **Fixed recipe for every YOLO run** (Team rules, training rule 1): `epochs=50 patience=10 imgsz=1280 batch=8 optimizer=SGD lr0=0.01 seed=0 deterministic=True` (sprint: `epochs=15 patience=5`), default augmentation unless the experiment changes it. RT-DETR runs use the same recipe except `optimizer=AdamW lr0=0.0001`. If batch 8 runs out of memory on the largest model, use `batch=4` for every A run and note it in the tracker.
6. **A0, the baseline without training:** `yolo detect val model=pretrained_models/yolo/yolo11m.pt data=data/packs/A/gsr1.yaml imgsz=1280 classes=0 project=runs/detA name=A0_coco`.
   - Check: mAP50-95 and recall printed; write them in the tracker row A0.
7. **Write `tools/recall_by_size.py`.** Run a model on the valid images, match predictions to ground-truth boxes at IoU ≥ 0.5, and print recall for box heights under 50 px, 50–100 px and over 100 px **in original 1080p pixels** (pack A images are 1280×720: multiply heights by 1.5). Run it on A0.
   - Check: the three recalls are printed; the small-box recall should be the lowest.
8. **A1, the first training run:** `yolo detect train model=yolo11m.pt data=data/packs/A/gsr1.yaml <recipe from step 5> project=runs/detA name=A1_yolo11m_ft` (as a SLURM job on fep).
   - Check: `runs/detA/A1_yolo11m_ft/results.csv` has training and validation losses and mAP for every epoch; `results.png` shows the curves; `weights/best.pt` exists.
9. **Score A1** with `yolo detect val` (as step 6, `model=runs/detA/A1_yolo11m_ft/weights/best.pt`, no `classes`) and with `tools/recall_by_size.py`. Fill the tracker.
10. **Pipeline check of A1** (integration owner, on the PC, full-resolution data). Copy `best.pt` to `pretrained_models/yolo/A1_yolo11m_ft.pt`. Run the full pipeline on the 10 ablation clips (command shape in Team rules) with `modules.bbox_detector.cfg.path_to_checkpoint=${model_dir}/yolo/A1_yolo11m_ft.pt`. Keep `min_confidence` at 0.4 for every run.
    - Check: GS-HOTA and GS-DetA in the tracker next to the baseline.
11. **Run the experiments** in the table below, in order. Each one repeats steps 8–9 with one change. Step 10 only for the best run of each experiment group.
12. **Finish:** repeat your best run and A1 with `seed=1` and `seed=2`; report mean ± std, curves and the recall-by-size table.

## Experiments

Run them top to bottom. The reference is A1 unless the row says otherwise; change only what the row names.

| Run | Group | Change | Failure it targets |
| --- | --- | --- | --- |
| A2a, A2b | Model scaling | `model=yolo11s.pt`, `model=yolo11l.pt` (A1 is the m size) | small, distant players |
| A3 | Convolution vs attention | `model=rtdetr-l.pt` with the RT-DETR recipe; reference A2b (YOLO11l, similar size) | overlapping players in crowds |
| A4 | Input resolution | `imgsz=640` | small players |
| A5a, A5b | Augmentation | `mosaic=0`; then `scale=0.9` | small players, only 3 training matches |
| A6 | Training procedure | `freeze=10` (backbone frozen) | overfitting to 3 matches |
| A7 | Auxiliary task | `data=data/packs/A/gsr4.yaml` (4 role classes) | role errors downstream |

**A7 in the sprint:** its component metric (per-class AP from `yolo val` with `gsr4.yaml`) needs no code change. **Only its pipeline check needs one, done with the integration owner:** the pipeline keeps only class 0, so a 4-class model would drop goalkeepers, referees and others. Copy the TrackLab detector wrapper to `sn_gamestate/detector/yolo_roles_api.py`, keep every box of classes 0–3, and add a config for it. Compare person recall with A1, and the detector's role accuracy with the role from Task C. (Checked: a 1-epoch 4-class training on pack A runs and reports per-class metrics.)

**Done when:**

- [ ] A0 to A7 are in the tracker with curves saved (sprint: A0–A3 and A7)
- [ ] Recall by size table for A0, A1 and the best run
- [ ] Pipeline GS-HOTA for the best run of each group
- [ ] Best run and A1 repeated with seeds 1 and 2 (after the sprint)
- [ ] Five failure cases from the best model saved as images with one line each on why it failed
