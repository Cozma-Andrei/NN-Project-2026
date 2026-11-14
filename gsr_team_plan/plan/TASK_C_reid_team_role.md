# Task C: Re-ID, team and role

Retrain the multi-task re-identification network (identity, team, role), then study backbones, embedding size, multi-task loss weights, the role loss under heavy class imbalance, augmentation and initialisation.

> **Sprint override (`00_SPRINT.md`):** do steps 1–7, skip step 8. The reference is **C-ref: a fine-tune from the provided checkpoint** (keep `load_weights`) with `modules.reid.cfg.train.max_epoch=5`. Runs: C-ref (focal), C5b `focal_weighted`, C5a `ce`, C4a `role_weight=0`. Training runs as fep SLURM jobs; TrackLab needs the full dataset there (`tools/get_gsr.py <data folder> train valid`, then `data_dir=<data folder>`). Step 9 (pipeline check) is done by the integration owner.

## Starting point

- **Today's network:** PRTReID (built on BPBReID): HRNet-W32 backbone, 256×128 input crops, 256-dimensional embedding, plus a team head and a role head. Checkpoint: `pretrained_models/reid/prtreid-soccernet-baseline.pth.tar` (396 MB, md5 `9633825232bc89f23a94522c5561650e`). Config: `sn_gamestate/configs/modules/reid/prtreid.yaml` and `dataset/prtreid_dataset.yaml`.
- **The loss** is hard-coded in `prtreid/engine/image/part_based_engine.py`, function `combine_losses`: `loss = reid_loss + 0.1 * team_loss + 1.5 * role_loss`. The role loss is a focal loss (gamma 2, no class weights). The team loss is a triplet loss on team labels, applied to the first 24 crops of a batch (players).
- **How the pipeline uses it:** the embedding drives tracking; teams come from k-means with 2 clusters over the mean embedding of each player track, per clip (`sn_gamestate/team/tracklet_team_clustering_api.py`); the role is the role head's argmax, voted per track.
- **Training is built into TrackLab** (`modules.reid.training_enabled=True`). It cuts crops from ground-truth boxes of `train` (`dataset.train.set_name: train`) and evaluates re-ID on `valid`.
- **Class imbalance** (train boxes): player 605k, referee 61.5k, goalkeeper 24.1k, other 1.4k (0.2%).
- **PRTReID defaults** (`prtreid/scripts/default_config.py`): Adam, lr 0.00035, weight decay 5e-4, `train.eval_freq=-1` (evaluate only after training), `test.evaluate` = test-only mode. Transform codes: `rf` flip, `rc` crop, `re` erasing, `cj` colour jitter, `ro` occlusion.

| Metric (on valid) | Direction |
| --- | --- |
| Re-ID mAP and rank-1 (printed by the training engine) | higher is better |
| Role accuracy and per-class F1 for goalkeeper, referee, other | higher is better |
| Team accuracy: k-means on ground-truth player tracks per clip vs true team, best of the 2 label swaps | higher is better |
| GS-HOTA and GS-AssA on the 10 ablation clips | higher is better |

## Step by step

Do the steps in order. Each step ends with a check; do not start the next step until the check passes.

1. **Read `01_TEAM_RULES.md`.** Then `conda activate tracklab` and `cd ~/sn-gamestate`.
2. **Make PRTReID editable** so your code changes are tracked: `git clone https://github.com/VlSomers/prtreid ~/prtreid` and `pip install -e ~/prtreid --no-deps`.
   - Check: `python -c "import prtreid; print(prtreid.__file__)"` prints a path inside `~/prtreid`.
3. **Make the loss configurable** (the only code change before training). In `~/prtreid/prtreid/scripts/default_config.py`, under `cfg.loss`, add `cfg.loss.team_weight = 0.1`, `cfg.loss.role_weight = 1.5`, `cfg.loss.role_loss = 'focal'`. In `combine_losses` of `prtreid/engine/image/part_based_engine.py`, replace `0.1` and `1.5` by those two values, and pick the role loss from `role_loss`: `focal` (today's), `ce`, `focal_weighted` (focal with inverse-frequency class weights as `weight`), `ce_smooth` (cross-entropy with label smoothing 0.1). Commit in `~/prtreid` with a clear message.
   - Check: with the defaults, one training batch gives the same loss value as before the change.
4. **Settings every C run uses** (add them to every command):
   - `dataset.nvid=-1` (otherwise the crops come from a single video per split)
   - `test_tracking=False eval_tracking=False` (train only, no pipeline run)
   - `modules.reid.training_enabled=True modules.reid.cfg.test.evaluate=False` (`evaluate=True` means test only)
   - `modules.reid.cfg.model.load_config=False` (otherwise the checkpoint's config overwrites yours)
   - `modules.reid.cfg.train.eval_freq=1 modules.reid.cfg.test.start_eval=0` (evaluate on valid after every epoch)
   - on fep: `data_dir=<data folder>`
   - **Fixed recipe:** the PRTReID defaults: Adam, learning rate 0.00035, weight decay 5e-4, batch 32 with 4 crops per identity, 20 epochs (sprint: 5), seed 0 (`modules.reid.cfg.train.seed=0`). Pick the best epoch by valid mAP.
5. **Smoke test** (a short fep job): the settings from step 4 plus `modules.reid.cfg.train.max_epoch=1`. The first run builds the crop dataset under `<data_dir>/reid` (slow once).
   - Check: the log shows the training loss for one epoch, then mAP and rank-1 on valid. If it stops with an error, paste the error in the team chat and wait; do not work around it. (This training path has not been run yet by the team.)
6. **Write `tools/eval_reid_attrs.py`.** For every valid clip, take ground-truth boxes of every 10th frame (skip balls), run the PRTReID model on the crops, and compute: role accuracy and per-class F1; team accuracy, by averaging the embeddings of each ground-truth player track, running k-means with 2 clusters exactly as `tracklet_team_clustering_api.py` does, and comparing with the true team (best of the 2 label swaps).
7. **C0, the provided checkpoint:** run step 6 on it, and its re-ID mAP with `modules.reid.cfg.test.evaluate=True` (test only, default weights).
   - Check: tracker row C0 filled.
8. **C1, retrain from ImageNet weights** (not in the sprint): settings from step 4 plus `modules.reid.cfg.model.load_weights=''`. If the log says an ImageNet HRNet file is missing in `pretrained_models/reid`, download the file it names from the BPBReID README and rerun.
   - Check: curves saved (training loss and valid mAP per epoch); step 6 metrics in the tracker. C1 should land near C0; if it is far below, tell the team before going on.
9. **Pipeline check** of the best run of each group (integration owner): point `modules.reid.cfg.model.load_weights` to its checkpoint and rerun from `reid` on the 10 ablation clips with the saved baseline state.
10. **Run the experiments** in the table below, in order; each repeats step 8 (sprint: the C-ref fine-tune) with one change.
11. **Finish:** best run and C1 repeated with seeds 1 and 2; report mean ± std, curves, and the per-class role F1 table.

## Experiments

Run them top to bottom. The reference is C1 (sprint: C-ref) unless the row says otherwise; change only what the row names (override keys start with `modules.reid.cfg.`).

| Run | Group | Change | Failure it targets |
| --- | --- | --- | --- |
| C2a, C2b | Convolution vs attention | `model.bpbreid.backbone=resnet50`; `model.bpbreid.backbone=fastreid_resnet_nl` (ResNet-50 with non-local attention blocks) | identity swaps in crowds |
| C3a, C3b | Embedding size | `model.bpbreid.dim_reduce_output=128`; `=512` | model scaling of the head |
| C4a–C4d | Multi-task weights | `loss.role_weight=0`; `loss.role_weight=3`; `loss.team_weight=0`; `loss.team_weight=0.5` | do team and role help or hurt re-ID? |
| C5a–C5c | Role loss | `loss.role_loss=ce`; `=focal_weighted`; `=ce_smooth` | rare roles: goalkeeper 3%, other 0.2% |
| C6a, C6b | Augmentation | `data.transforms=[rc,re,cj]` (adds colour jitter); `data.transforms=[rc,re,rf]` (adds horizontal flip) | expected: colour jitter hurts team accuracy, because team is the kit colour; we test it |
| C7 | Initialisation | fine-tune from the provided SoccerNet checkpoint instead of ImageNet | transfer |

For C4 and C5, report re-ID mAP, role F1 per class and team accuracy side by side: a change can help one task and hurt another, and that trade-off is the result.

**Done when:**

- [ ] C0 to C7 are in the tracker with curves saved (sprint: C0, C-ref, C5b, C5a, C4a)
- [ ] Per-class role F1 table for C0, the reference and the best run
- [ ] Pipeline GS-HOTA for the best run of each group
- [ ] Best run and its reference repeated with seeds 1 and 2 (after the sprint)
- [ ] Five failure cases (wrong role, wrong team, identity swap) saved as crops, with one line each on why they failed
