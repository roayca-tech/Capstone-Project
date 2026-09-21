# Deviations and paper open questions

Recorded per AGENT_SPEC §10. Do not treat these as results.

## Open questions (choices)

1. **Gating MLP hidden width.** Paper does not state it. Used **32**, inferred from Figure 1. WGSA learnable parameter count remains exactly 292.
2. **Augmentation.** Unspecified. Training uses **horizontal flip only** (plus ImageNet normalisation). Color jitter, blur, sharpen, JPEG, and random erasing are disabled so high-frequency evidence is not destroyed.
3. **Seeds.** Not reported. Default seed is `0`; in-domain runs should be repeated with a second seed and reported as mean ± std. Variance is unknown until those runs execute.
4. **Frame-level vs video-level.** Table 1 does not say. **Frame-level** metrics are used everywhere (`metrics.json` sets `"frame_level": true`).
5. **`eps` in Eq. 10.** Unspecified. Used **1e-6**.

## Major deviation: dataset substitution

**FF++ access was never granted.** The maintainer form returned no download script, and the DFDC portal at `dfdc.ai/sign-up` rejects valid 12-digit AWS account IDs ("Invalid aws account ID"), a documented and long-standing fault on their side.

**Training data is therefore DFDC, obtained from the Kaggle competition** ([deepfake-detection-challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge)) after accepting the competition rules. This is Meta's official competition release, not a third-party re-upload, so no unknown re-encoding is introduced.

Consequences, which must be stated in any write-up:

- **Tables 1, 3, and 6 cannot be reproduced.** They are FF++ in-domain and per-manipulation results. DFDC has no Deepfakes / Face2Face / FaceSwap / NeuralTextures breakdown.
- **The paper's DFDC figure (70.41% AUC) is not a comparison target.** That number is a model *trained on FF++ C23* and *tested* on DFDC. Training and testing on DFDC measures something different and should score higher.
- The defensible result is **in-domain DFDC: WGN versus the backbone-only baseline** under one identical pipeline. That isolates the WGSA contribution, which is the paper's core claim (C1 > C0, C1 > C2).

Protocol choices for DFDC, in `wgn/dfdc.py`:

- **Identity-safe splits.** Each fake's `original` field names its source real clip. A fake and its original share a face, so they are grouped and assigned to one split together. Split is 70/15/15 over *identity groups*, seeded and deterministic.
- **Class balancing by frame budget.** DFDC is roughly 1 real to 5 fakes. Fake clips give 10 frames each and real clips scale up (typically ~50) so frame counts per class match within 1.05. This generalises the paper's 10-per-fake / 40-per-real ratio (Sec. 4.2) instead of introducing loss weighting.
- Crops still come from the shared `wgn.preprocess.crop_face` at 256×256 with the 1.3× box expansion, so preprocessing is identical to the FF++ path.

## Implementation notes (not paper deviations)

- Official FF++ `train.json` / `val.json` / `test.json` were downloaded from [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits) (360 / 70 / 70 pairs → 720 / 140 / 140 clips).
- Ablation C2 is `wgn.baselines.CBAMAttention` (channel mean+max → 7×7 conv → sigmoid). It replaces WGSA only; fusion and the classifier are unchanged, so the model stays at ~5.76 M.
- Ablation C7 is `WGSA(bands="hh")`.
- Backbone-only rows use `BaselineNet` (GAP → Linear, no fusion conv), selected with `--no-wgsa`.
- Figure 2 column order, in the absence of the paper figure file: input, F (channel-mean), A, F⊙A (channel-mean), Grad-CAM on fused features, overlay.

## What was not done

No training run has executed yet. FF++, FaceShifter, Celeb-DF, and WildDeepfake data are not present on this machine; DFDC is being downloaded from Kaggle. See `REPORT.md` for the per-table status.
