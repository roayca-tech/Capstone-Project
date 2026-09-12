# Deviations and paper open questions

Recorded per AGENT_SPEC §10. Do not treat these as results.

## Open questions (choices)

1. **Gating MLP hidden width.** Paper does not state it. Used **32**, inferred from Figure 1. WGSA learnable parameter count remains exactly 292.
2. **Augmentation.** Unspecified. Training uses **horizontal flip only** (plus ImageNet normalisation). Color jitter, blur, sharpen, JPEG, and random erasing are disabled so high-frequency evidence is not destroyed.
3. **Seeds.** Not reported. Default seed is `0`; in-domain runs should be repeated with a second seed and reported as mean ± std. Variance is unknown until those runs execute.
4. **Frame-level vs video-level.** Table 1 does not say. **Frame-level** metrics are used everywhere (`metrics.json` sets `"frame_level": true`).
5. **`eps` in Eq. 10.** Unspecified. Used **1e-6**.

## Implementation notes (not paper deviations)

- Official FF++ `train.json` / `val.json` / `test.json` were downloaded from [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits) (360 / 70 / 70 pairs → 720 / 140 / 140 clips).
- Ablation C2 is `wgn.baselines.CBAMAttention` (channel mean+max → 7×7 conv → sigmoid). It replaces WGSA only; fusion and the classifier are unchanged, so the model stays at ~5.76 M.
- Ablation C7 is `WGSA(bands="hh")`.
- Backbone-only rows use `BaselineNet` (GAP → Linear, no fusion conv), selected with `--no-wgsa`.
- Figure 2 column order, in the absence of the paper figure file: input, F (channel-mean), A, F⊙A (channel-mean), Grad-CAM on fused features, overlay.

## What was not done

FF++ / FaceShifter / Celeb-DF / DFDC / WildDeepfake video data are not present on this machine. Preprocessing, training, and table numbers have **not executed**. See `REPORT.md`.
