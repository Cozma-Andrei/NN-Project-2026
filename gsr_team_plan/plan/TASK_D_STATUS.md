# Task D (jersey numbers): status update, 15 Nov

**Summary.** The baseline (D0, MMOCR plus a vote) is still our best Task D result, at 34.9% track accuracy on valid. Our first three trained networks (D1–D3) scored 12.7–18.4% because they memorise players instead of reading digits (shortcut learning). The first fix, E9 (training only on crops where the number is readable), nearly doubles their scores: **D6 (ViT-S) reaches 32.95%**, close to D0. The second fix, D4 (an adversary that removes player appearance), removes appearance information as intended but does not improve accuracy over 3 seeds (33.0 ± 2.0 vs 33.5 ± 0.8 without it). Our best trained model is D6 at 33.5 ± 0.8, still just below D0, and no model reads numbers never seen in train. The bottleneck is the number of readable training crops. Past SoccerNet winners point the same way: the key step is filtering training images down to ones with a legible number, then adding more varied data.

All numbers below are on pack D **valid** (matches 2, 3, 5), scored per ground-truth track. Higher is better.

---

## 1. Setup (done)

- **Pack D** (`tools/pack_gsr.py`): full-resolution player crops plus `labels.csv`. Train has 125,677 crops in 1,224 tracks (matches 4, 6, 9). Valid has 64,518 crops in 1,220 tracks. The splits are never mixed.
- **Training** (`tools/train_jersey.py`): fixed recipe of AdamW, learning rate 1e-4, weight decay 0.05, cosine decay, 8 epochs, batch 128, seed 0, mixed precision. Input is the top 60% of the crop (upper body), with no horizontal flip because flipping mirrors digits.
- **Track rule** (`tools/jersey_lib.py`, fixed and not tuned): keep frames with p(has number) > 0.5. If no frame is kept, the track is "none". Otherwise the track gets the number with the largest summed probability.
- **Evaluation** (`tools/eval_jersey.py`): accuracy for all tracks, tracks with a number, tracks with no number, and numbers never seen in train. It also breaks results down by box height, fragmentation, match and error type, and saves failure image strips.
- **Baseline reproduction** (`tools/d0_mmocr.py`): the pipeline's own MMOCR reader and `select_highest_voted_att` vote, run on the same crops.

## 2. Results so far

| Run | What | All | With number | No number | Unseen numbers |
|---|---|---|---|---|---|
| **D0** | MMOCR + majority vote (today's pipeline) | **34.92** | 18.53 | **97.24** | **17.02** |
| D1 | ResNet-18, 2 digit heads + "has number" head | 15.0 | 14.9 | 15.4 | 0 |
| D2 | ViT-S/16, same heads | 18.44 | **19.9** | 13.0 | 0 |
| D3 | ResNet-18, one 101-class head | 12.7 | 11.8 | 16.1 | 0 |

| D5 | D1 trained on the teacher-filtered set (E9) | 28.28 | 10.87 | 94.49 | 0 |
| D6 | D2 trained on the teacher-filtered set (E9) | 32.95 | 16.36 | 96.06 | 0 |
| D4 | D5 + gradient-reversal adversary | 29.43 | 11.49 | 97.64 | 0 |
| D4b | D6 + gradient-reversal adversary (seed 0; see 3-seed table below) | 35.33 | 19.25 | 96.46 | 0 |

D1 and D2 were resumed from a checkpoint after the PC crash on 14 Nov, so their data order differs from an uninterrupted run. Best epoch for D5 and D6: 7 of 8.

**E9 result.** Filtering the train set roughly doubles overall accuracy (D1 15.0 → D5 28.3; D2 18.4 → D6 32.95). The gain comes almost entirely from "no number", which goes from 13–15% up to 94–96%, the same level as D0. On tracks with a number, though, D5 and D6 do slightly *worse* than D1 and D2 (10.9 vs 14.9 and 16.4 vs 19.9), and unseen numbers are still at 0%. The filtered set is small: only 1,169 numbered crops from 205 tracks, covering 27 distinct numbers, against 23,358 crops without a number. The networks still fit train perfectly while valid loss rises, so they still memorise, just on fewer examples. Conclusion: MMOCR is too weak a teacher to give us enough readable crops. We need more varied data (D9) or a stronger pretrained reader (D8).

**D4 result (seeds 0, 1, 2, ViT-S, mean ± standard deviation).**

| Run | Best epoch, all | Last epoch, all | Last epoch, with number | Last epoch, none |
|---|---|---|---|---|
| D6, filtered set | **33.50 ± 0.81** | **33.44 ± 0.86** | **17.12 ± 1.01** | 95.54 ± 0.91 |
| D4b, D6 + adversary | 33.01 ± 2.02 | 31.31 ± 1.36 | 13.91 ± 1.92 | 97.51 ± 0.99 |

- **The adversary does not help.** The 35.33 from seed 0 was luck; seeds 1 and 2 give 32.05 and 31.64.
- **Information removal works:** the adversary falls to chance (0.2–0.7% vs 0.9%) in every seed.
- **But reading gets worse.** Every seed peaks at epoch 4, then falls as the reversal weight ramps to 1, mostly on tracks with a number.
- **Likely reason:** kit appearance also helps reading (digit colour and contrast against the shirt), so removing it costs more than removing the shortcut gains.
- **D4 (ResNet-18), seed 0 only:** 29.43 vs D5's 28.28, too small to call.
- **Conclusion:** the diagnosis (shortcut learning) holds, but the limit is the missing signal. 1,169 readable crops covering 27 numbers cannot teach digits in general; unseen numbers stay at 0% in all runs. Best trained model: D6, 33.5 ± 0.8, still about 1.4 points below D0.

## 3. What works and what doesn't

**What works**
- The data packs, training, evaluation and baseline tools all run end to end, with reproducible configs, curves and checkpoints.
- D0 is very reliable on "no number": it says "none" correctly 97% of the time. When MMOCR is confident (above 0.9), it reads the correct number 83% of the time.
- A stronger pretrained backbone helps a little: ViT-S (D2) beats ResNet-18 (D1) on tracks with a number, 19.9 vs 14.9.
- Digit heads beat one 101-class head (D1 15.0 vs D3 12.7), as expected for numbers that are rare or never seen.

**What doesn't work**
- **D1–D3 memorise player identity.** Train has only 3 matches, so each number is a handful of players with fixed kits, and the network learns "this kit and this player means number 7" instead of reading the 7. The evidence:
  - D1 gets 97.4% on train tracks with a number but 14.9% on valid.
  - Train loss falls to about 0 while valid loss rises after the first epochs.
  - Accuracy is 0% on the 94 valid tracks whose number never appears in train (2, 37, 42, 43, 47, 59, 99).
  - Accuracy barely changes with box height, so image size is not the problem.
  - 65% of errors are a wrong number, not a missed one.
- **D0 fails the opposite way.** Before fine-tuning, MMOCR reads the true number on only 1.4% of numbered crops, so 54% of numbered tracks come out as "none".
- **Running many processes at once crashed the PC.** On 14 Nov, 5–6 GPU processes per card made the NVIDIA driver fail and Windows crashed (bugcheck 0x113). We now run at most 2–3 processes per card and cap memory on GPU 0. The team rules have been updated.

## 4. What is running now

| Run | Change (one only) | Reference | Status |
|---|---|---|---|
| E9 teacher pass | MMOCR reads every numbered **train** crop | none | Done |
| D5_r18_teacherfilter | Train only on crops where MMOCR read the true number, plus all crops of tracks without a number | D1 | Done, 28.28 |
| D6_vits_teacherfilter | Same filtered train set | D2 | Done, 32.95 |
| D4_r18_grl | Adversary behind a gradient-reversal layer (DANN) | D5 | Done, 29.43 (seed 0 only) |
| D4b_vits_grl | Same adversary | D6 | Done, 33.01 ± 2.02 (3 seeds); D6 is 33.50 ± 0.81 |

**Why E9.** On a numbered crop where the digits can't be seen (back turned, blur, occlusion), the only way to get the label right is to recognise the player. Dropping those crops removes the shortcut from the training signal. Valid is not touched.

**Why D4.** An extra head tries to predict which clip and which team side a crop comes from (114 groups). The gradient-reversal layer trains the backbone to make that impossible.
- **The reasoning:** a player is (match, team) plus number. If the features can't tell which team or clip a crop is from, the only thing left to separate players is the number. Digits don't help the adversary, since the same numbers appear on both teams and in every clip.
- **The strength** of the reversal follows DANN's standard 0→1 ramp and is not tuned.
- **For the report:** each epoch logs the adversary's accuracy against chance (0.9%). If it sits near chance, the identity information really was removed.

## 5. How others solved it (for the report)

**SoccerNet Jersey Number challenge 2023** (one number per tracklet; best 92.85% accuracy, 15 teams):
- **1st, ZZPM (Xidian), 92.85%.**
  - Ran the DBNet++ text detector on the training set and **dropped every training image without a detection at confidence ≥ 0.9**.
  - Fine-tuned several text recognisers (SVTR, SATRN, NRTR, ASTER) with heavy augmentation, including blending several frames into one image.
  - Combined them by voting. The filtering step is the same idea as our E9.
- **2nd, UniBw Munich, 90.95%.**
  - Did not train on the challenge labels at all, which they found too noisy.
  - Used CLIP ViT-L/14 zero-shot to pseudo-label the larger and more varied SoccerNet re-ID dataset, then fine-tuned on that.
  - Voted only over frames with confidence above 70%.
- **3rd, Tencent, 88.1%:** an OCR detector, then a transformer over the features of a whole tracklet.
- **5th, Meituan, 81.7%:** a video transformer with two digit heads, plus extra losses for rare numbers.
- **The common recipe:** text detection, then fine-tuned OCR, then a majority vote per tracklet.

**Koshkina & Elder (2024), the reference academic method.**
- A ResNet-34 **legibility classifier** first keeps only frames where the number can be read.
- Then **ViTPose** keypoints give a torso crop (shoulders to hips).
- Then a fine-tuned **PARSeq** text recogniser reads it.
- They report 87.4% on SoccerNet tracklets, and fine-tuning PARSeq matters a lot.

**Game State Reconstruction challenges (our task, scored by GS-HOTA).**
- **2024:** the organisers say the baseline's MMOCR reader, not fine-tuned on soccer, was a main bottleneck. That matches our D0 analysis.
- **Constructor.Tech** won 2024 and was a close 2nd in 2025 (63.81 GS-HOTA).
  - Their jersey model is close to our D1: a ResNet-18 with two digit heads, focal loss, AdamW at 1e-4.
  - The difference is data: **~70k upper-body crops chosen because the number is visible**.
- **2025 winner KIST, 63.90 vs the 29.01 baseline.** They prompt the LLaMA-3.2-Vision language model for number, role and colour, then vote per tracklet. Another team used the Qwen2-VL language model. These are large models used as they are, not networks we train, so they are outside our course scope (they could only serve as a teacher for pseudo-labels).
- **2025 4th, Playbox:** the Koshkina pipeline (ViTPose torso crop plus PARSeq).
- Several teams hand-annotated 1–3k extra boxes. Our rules don't allow that, but an external dataset does the same job.

**What this means for us.** Nearly every strong entry (1) trains only on crops with a legible number and (2) uses more varied data or a model pretrained on text. Our diagnosis (shortcut learning) and our E9 run match the field. D4 (adversarial training) is not something any team used, which makes it an original experiment for the course.

## 5b. D9 overlap check: SoccerNet Jersey-2023 (15 Nov)

Script: `tools/overlap_jersey2023.py` (output in `runs/jerseyD/D9_overlap/`). Both datasets are cut from the SoccerNet-Tracking videos (the GSR paper: "our dataset expands upon SoccerNet-Tracking"; the sn-jersey README: "derived from the sn-gamestate and sn-tracking datasets") and share no IDs. So we matched by content: the mean DINOv2 embedding of each tracklet, compared with every pack D track (train + valid).

- **Jersey-2023 `train` part (1,427 tracklets) is mostly our own GSR train data.** 1,000 tracklets have a near-duplicate track in our train split with the same number (cosine similarity ≥ 0.9), all from matches 4, 6 and 9. It adds almost no new players or kits, so it can't fix the shortcut.
- **Jersey-2023 `test` part (1,211 tracklets) is not in our train or valid.** Only 70 near-duplicates, consistent with chance look-alikes. Its median similarity is lower (0.89 vs 0.94). Since SoccerNet-Tracking's test clips are GSR's test clips (both 49 clips), it most likely comes from **our test matches (7, 8, 11)**. Training on it could leak test.
- **No exact duplicates of valid** (0 at similarity ≥ 0.95). The high-similarity valid matches mostly have a *different* number, which means look-alike kits rather than the same player.

**Decision: Jersey-2023 is not usable as planned.** The train part is redundant with our data. The test part can only be used after an overlap check against GSR `test`, which means reading test images, a decision for the integration owner. D9 is on hold. External data with no overlap risk: pretrained text recognisers (D8), the Koshkina & Elder hockey jersey dataset (different sport), or synthetic rendered digits.

## 6. Plan

| When | Run | Change | Reference |
|---|---|---|---|
| Done | D5, D6 | Teacher-filtered train set | D1, D2 |
| Done | D4, D4b | Gradient-reversal adversary (clip × team side) | D5, D6 |
| After the sprint | D7 | Torso crop from ViTPose keypoints instead of the top 60% of the box | best so far |
| After the sprint | D8 | Fine-tune pretrained PARSeq instead of an ImageNet classifier | best so far |
| On hold | D9 | Add SoccerNet Jersey-2023 data: the overlap check found its train part duplicates our train split and its test part likely comes from our test matches (section 5b) | best so far |
| After the sprint | D10 | Small transformer over each tracklet instead of summed votes (learned aggregation) | best so far |
| End | | Seeds 1 and 2 for the best run and its reference; GS-HOTA rerun from `jersey_number_detect` on the 10 ablation clips | |

Each run changes one thing against a named reference run, as the team rules require. D7–D10 is the order of priority. D8 and D9 are where the literature sees the largest gains.

**Still open:**
- the 10-clip baseline state, needed for the GS-HOTA rerun;
- tracker rows for D0–D3;
- D8 (PARSeq) and D9 (more data) now look like the most important next steps, because E9 showed we are short of readable training crops.

## Sources

- SoccerNet 2023 Challenges Results: https://arxiv.org/abs/2309.06006
- SoccerNet 2024 Challenges Results: https://arxiv.org/abs/2409.10587
- SoccerNet 2025 Challenges Results: https://arxiv.org/abs/2508.19182
- Constructor.Tech, "From Broadcast to Minimap" (CVPR 2025 workshop): https://arxiv.org/abs/2504.06357
- Koshkina & Elder, "A General Framework for Jersey Number Recognition in Sports Video": https://arxiv.org/abs/2405.13896
- Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation" (gradient reversal, DANN): https://arxiv.org/abs/1409.7495
- SoccerNet jersey dataset and leaderboard: https://github.com/SoccerNet/sn-jersey
