# Task D: Jersey number recognition

Replace the baseline's general-purpose OCR with two trained networks, a multi-task frame network and a learned track aggregator, and study their architectures, losses, regularization and training procedures by ablation on SoccerNet-GSR.

> **Sprint override:** for the 3-day sprint, follow **"Task D sprint steps" in `00_SPRINT.md`** (pack D crops, digit-head vs 101-way head, ResNet18 vs ViT-S, rule-based track evaluation, no pipeline module yet). This file is the full plan for after the sprint.

## Why Task D

The baseline reads numbers with a general text reader and lets any single misread decide a whole track, so it almost never answers "no number".

**What runs today**

1. **Per frame** (`sn_gamestate/jersey/mmocr_api.py`): every detected box in every frame goes through MMOCR, a DBNet text detector followed by a SAR recognizer, on the whole-body crop. It keeps the digits of the highest-confidence reading and cuts them to 2 characters.
2. **Per track** (`MajorityVoteTracklet` → `select_highest_voted_att` in `tracklab/utils/attribute_voting.py`): it sums confidence per number and picks the largest sum. This is already a confidence-weighted vote, so weighted voting is not a new contribution.

**Where it fails**

- Frames with no reading vote `None` with confidence 0, so one stray misread anywhere in a track wins. Yet 291 of 1224 player and goalkeeper tracks in GSR train (24%) have no number.
- The reader is not trained on jerseys: small, blurred, folded or side-on numbers are common.
- OCR runs on every box in every frame, a large share of the ~10 minutes the pipeline needs per clip.
- Jersey number is one of the attributes GS-HOTA requires to count a match, so these errors lower detection accuracy (GS-DetA), not just a side metric.

Baseline reference: on one validation clip (SNGS-021) the full pipeline scored 32.3 GS-HOTA, 20.3 GS-DetA, 51.4 GS-AssA. The isolated jersey score from the GSR paper is 56.8.

## Data

The sprint uses only GSR crops (pack D). After the sprint, SoccerNet Jersey-2023 and its weak legibility labels can be added for the legibility head and extra recognizer data.

| Dataset | What it has | Size | Used for |
| --- | --- | --- | --- |
| SoccerNet-GSR train (games 4, 6, 9), pack D | 1224 player/goalkeeper tracks, 933 with a number, 41 distinct numbers; 125,677 crops (every 5th frame) | 0.9 GB with valid | frame network training; track aggregator training |
| SoccerNet-GSR valid (games 2, 3, 5), pack D | 1220 tracks, 254 without a number; 64,518 crops (every 10th frame) | | all ablations |
| SoccerNet-GSR test (games 7, 8, 11) | same format | | final numbers only, used once by the integration owner |
| [SoccerNet Jersey-2023](https://github.com/SoccerNet/sn-jersey) | 2853 player tracklets of low-resolution, blurred crops; one number per tracklet, `-1` = never visible; challenge split has hidden labels | train 1.25 GB, test 1.07 GB on [Hugging Face](https://huggingface.co/datasets/SoccerNet/SN-Jersey-2023) | after the sprint: recognizer and legibility training |
| [Koshkina & Elder weak labels](https://github.com/mkoshkina/jersey-number-pipeline) | per-image legibility labels and jersey-crop labels for SoccerNet, produced by models trained on hockey | Google Drive links in the repo | after the sprint: legibility head |

**Rules**

- Track labels are weak: every frame gets the number even when it cannot be seen.
- The weak legibility labels are machine-made and noisy. We treat the noise as a training problem (E4 label smoothing, E9 self-training) instead of relabelling by hand.
- Before training on Jersey-2023, check that no Jersey-2023 tracklet comes from a GSR valid or test match. Both datasets are built from SoccerNet broadcasts, so overlap would leak players and kits into evaluation.

## Evaluation

Every idea is first scored on ground-truth tracks in minutes; only the finalists go through the full pipeline, which takes ~1h45 for the 10-clip subset.

**Fast harness (built in the sprint, `tools/eval_jersey.py`).** Run the frame network on ground-truth crops from GSR valid (pack D), aggregate per track, and compare with the track label. No detection, tracking or calibration involved.

| Metric | Level | Direction |
| --- | --- | --- |
| Track accuracy, all tracks | track | higher is better |
| Accuracy on tracks with a number | track | higher is better |
| Accuracy on tracks with no number ("none" predicted correctly) | track | higher is better |
| Legibility precision / recall | frame, on Jersey-2023 test | higher is better |
| Recognizer accuracy on legible crops | frame | higher is better |
| Inference time per clip (speed) | clip | lower is better |

**Full pipeline.** Reload saved tracker states and rerun only `jersey_number_detect` and `tracklet_agg` on the 10 fixed validation clips. Report GS-HOTA, GS-DetA and GS-AssA (higher is better), plus the attribute breakdown: GS-HOTA with jersey numbers switched off in the evaluator, to show how much jersey errors cost.

**Reference points.** The MMOCR baseline (D0) in both setups, the 56.8 isolated jersey score from the GSR paper, and the Jersey-2023 challenge leaderboard (top entry 92.85% accuracy) for the recognizer on its own.

## Models and experiments

Two trainable networks carry the project: a multi-task frame network and a track aggregator. Almost all effort goes into architectures, losses, regularization and training procedures; hand engineering is limited to a fixed upper-body crop and one rule-based baseline.

**Network 1: frame network.** One shared backbone, three heads: legibility (is a number visible), tens digit, units digit. Legibility is an auxiliary task inside the network instead of a separate classifier. **Never use horizontal flip** (it mirrors digits).

**Network 2: track aggregator.** Takes the frame network's per-frame features along a track and outputs one of the numbers or "none". It trains on GSR track labels with multiple-instance learning, so the weak per-track labels are the right supervision rather than a problem.

| # | Group | Experiment | What it tests | Failure it targets |
| --- | --- | --- | --- | --- |
| E1 | Architecture | ResNet18 vs ResNet34 vs ResNet50 | model scaling | small, blurred numbers |
| E2 | Architecture | ResNet34 vs ViT-S vs Swin-T | convolution vs attention | numbers seen at many angles and scales |
| E3 | Output and loss | 100-way softmax vs two digit heads vs PARSeq sequence decoder | problem framing; only 41 numbers appear in GSR train, so a 100-way head cannot predict unseen numbers | numbers absent from training |
| E4 | Loss | cross-entropy vs focal vs label smoothing | modified loss terms | skewed number and legibility frequencies; noisy weak labels |
| E5 | Auxiliary task | with vs without the legibility head (loss weight 0, 0.5, 1) | auxiliary task | reading numbers in frames where none is visible |
| E6 | Regularization | none vs blur + downscaling + random erasing vs + mixup/CutMix vs + stochastic depth | augmentation and regularization; only 3 training matches | overfitting to a few kits |
| E7 | Pretraining | ImageNet vs scene-text pretrained (PARSeq encoder) vs from scratch | transfer | domain gap |
| E8 | Training procedure | small ResNet student distilled from a fine-tuned PARSeq teacher vs trained on labels alone | distillation | speed with accuracy kept |
| E9 | Training procedure | self-training: teacher pseudo-labels unlabelled GSR frames, student retrains | teacher-student, semi-supervised | weak and missing frame labels |
| E10 | Aggregator | majority vote vs attention pooling vs GRU vs Transformer encoder | architecture over time | one misread decides the track |
| E11 | Aggregator training | two-stage (frozen frame network) vs end-to-end fine-tuning through the aggregator | training procedure | frame errors the aggregator cannot correct |

The sprint runs D1 (ResNet18 digit heads), D2 (ViT-S digit heads), D3 (ResNet18 101-way): first points of E2/E3.

**Order after the sprint.** E1, E3 and E10 first: they set the backbone, the head and the aggregator. Then E4–E7, then E8, E9 and E11 as time allows. For every network report optimizer, batch size and composition (balanced sampling of legible and illegible crops), epochs, and training and validation curves.

**Kept out on purpose.** Following the course rules, no normalization or activation swaps and no optimizer tuning: one optimizer and schedule per network, fixed early. Regularization experiments are tied to the measured overfitting gap, not added blind.

## Integration (after the sprint)

Two new TrackLab modules and two configs replace MMOCR and the vote; nothing else in the pipeline changes.

1. `sn_gamestate/jersey/jersey_api.py`: a `DetectionLevelModule` like `mmocr_api.py`. It runs the frame network once per box on a fixed upper-body crop **cut from the full-resolution frame**. Outputs `jersey_number_detection`, `jersey_number_confidence`, plus new columns `jersey_legibility` and per-frame features for the aggregator.
2. `sn_gamestate/jersey/jersey_tracklet_agg.py`: a `VideoLevelModule` that replaces `MajorityVoteTracklet` for jersey numbers only (majority vote baseline or a trained aggregator from E10). Role voting stays as it is.
3. Configs under `sn_gamestate/configs/modules/jersey_number_detect/` and `.../tracklet_agg/`, switched in `soccernet.yaml`.
4. Ablation runs load a saved tracker state and set `pipeline: [jersey_number_detect, tracklet_agg, team, team_side]`.

One small network per box replaces MMOCR's text detector plus recognizer, so the pipeline should get faster; we report time per clip.

## Timeline (after the sprint, approximate)

| Phase | Weeks | Content | Gate to pass |
| --- | --- | --- | --- |
| 0 Eval harness | 1 | GT-track scorer, MMOCR baseline (done in the sprint) | baseline reproduced |
| 1 Data | 2–3 | Jersey-2023 + legibility labels, leakage check | no overlap found |
| 2 Models | 3–6 | frame network E1–E9, aggregator E10–E11, curves for every run | finalists chosen |
| 3 Integration | 6–7 | TrackLab modules, 10-clip GS-HOTA | final test run |

## Risks and open questions

| Risk | Effect | Fallback |
| --- | --- | --- |
| Jersey-2023 shares matches with GSR valid or test | inflated scores, leakage | drop the overlapping tracklets; train on GSR train crops only for that comparison |
| Weak legibility labels too noisy | filter drops readable frames | rely on E4 label smoothing and E9 self-training; hand-label a few hundred crops only if both fail |
| Fragmented predicted tracks (77 tracks for 23 people on SNGS-021) | too few frames per track to read a number | report accuracy on ground-truth vs predicted tracks to separate our error from tracking error |
| Learned aggregator overfits 1224 training tracks | no gain over majority vote | add dropout over frames and track-level augmentation (dropping and shuffling frames); otherwise report it as a documented negative result |
| Gain hidden in GS-HOTA by detection and calibration errors | small end-to-end change | lead with the harness metrics and the jersey-off GS-HOTA breakdown |

- [ ] Confirm the Jersey-2023 train/test tracklet counts and which matches they come from.
- [ ] Decide whether to use the Koshkina & Elder fine-tuned PARSeq weights as a reference or train our own only.

## Sources

- [SoccerNet sn-jersey repository](https://github.com/SoccerNet/sn-jersey): Jersey-2023 dataset, label format, leaderboard
- [SN-Jersey-2023 on Hugging Face](https://huggingface.co/datasets/SoccerNet/SN-Jersey-2023): file sizes
- [Koshkina & Elder, jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline): legibility classifier, PARSeq weights, weak SoccerNet labels
- Baseline code: `sn_gamestate/jersey/mmocr_api.py` and TrackLab `tracklab/utils/attribute_voting.py`
- Somers et al., SoccerNet Game State Reconstruction, CVPRW 2024: isolated jersey score 56.8
