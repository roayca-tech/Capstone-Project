# WGN reproduction report

**Paper:** Ghosh & Naskar, *WGN: Wavelet-Guided Network for Efficient and Generalised Deepfake Detection*, CVPRW 2026.

This report is honest about execution. Numbers that were not produced by a run are marked **not executed**. They are not filled in.

## Dataset substitution

FF++ access was never granted, and the DFDC AWS portal rejects valid account IDs. Training therefore uses **DFDC from the Kaggle competition** (official Meta release, not a re-upload). `runs/DEVIATIONS.md` records this in full.

This changes what can be claimed:

- Tables 1, 3, and 6 are FF++-specific and **cannot be reproduced**. DFDC has no per-manipulation breakdown.
- The paper's 70.41% DFDC AUC is **not** a target here. That is an FF++-trained model tested on DFDC; training in-domain on DFDC measures something else.
- The defensible result is **in-domain DFDC, WGN versus backbone-only**, under one identical pipeline. That isolates WGSA, which is the paper's core claim.

## Status

| Phase | Status |
|---|---|
| 0–2 Environment + WGSA + WGN | Code in place. 53 tests cover the paper invariants. |
| 3 Data pipeline | **Executed.** Parts 00–03, 6,236 clips, 643 identities, identity-safe 70/15/15 split, audit passed (`data/dfdc/verify_report.json`). FF++ path remains unused. |
| 4 In-domain training | **Executed.** WGN and backbone-only, seeds 0 and 1, 25 epochs, frame-level. See the in-domain table below. |
| 5 Generalisation | **Executed as extra tests of the DFDC checkpoints**, not as the paper's Table 2. Celeb-DF v2 official test list and HiDF. See the table below. |
| 6 Ablations | C0 (backbone-only) vs WGN is the in-domain table below. C2 (`CBAMAttention`) and C7 (`bands="hh"`) have not been trained. |
| 7 Visualisation / tables | Scripts exist. `scripts/make_tables.py` emits "not executed" for every row until `runs/*/metrics.json` appear. |

## What is implemented

- `wgn/` package: WGSA, WGN, BaselineNet, CBAMAttention, preprocess, DFDC planning, train.
- DFDC identity-safe splitting: each fake stays with the real clip named in its `original` field, so no face crosses a split boundary. Class balance comes from a per-class frame budget (10 frames per fake, real clips scaled up), generalising the paper's 10/40 ratio.
- Official FF++ `train.json` / `val.json` / `test.json` (360 / 70 / 70 pairs), unused for now.
- Failure logging in `wgn.preprocess` (`failures.json`, not silent drops).
- `scripts/prepare_dfdc.py`, `verify_data.py`, `contact_sheet.py`, `preprocess_external.py`, `gradcam.py`, `make_tables.py`.
- Every training run writes `runs/<tag>/config.json` (argv, git SHA, seed) and `runs/<tag>/metrics.json` (per-epoch + frame-level test).
- Recorded paper ambiguities and the dataset substitution: `runs/DEVIATIONS.md`.

## What did not execute

FF++ training and cross-manipulation evaluation were not run. Tables 1, 2, 3, and 6 below stay **not executed**. Table 2 is an FF++-trained model; the Celeb-DF and HiDF numbers later in this report are DFDC checkpoints tested on those sets, and they are not that table. WildDeepfake was not used. `runs/wgn_dfdc_s0_partial/` is an interrupted earlier attempt and is not a result.

## Table 1 (in-domain, frame-level)

| Run | ACC | AUC | Δ ACC | Δ AUC | source |
|---|---:|---:|---:|---:|---|
| WGN FF++ C23 | not executed | not executed | — | — | — |
| Backbone FF++ C23 | not executed | not executed | — | — | — |
| WGN FF++ C40 | not executed | not executed | — | — | — |
| WGN FaceShifter C23 | not executed | not executed | — | — | — |

Published targets: WGN C23 95.32 / 98.90; backbone C23 93.70 / 98.12; WGN C40 80.20 / 89.00; FaceShifter C23 98.00 / 99.92.

## Table 2 (cross-dataset AUC)

| Dataset | AUC | published | source |
|---|---:|---:|---|
| Celeb-DF | not executed | 77.62 | — |
| DFDC | not executed | 70.41 | — |
| WildDeepfake | not executed | 68.65 | — |

## Table 3 (cross-manipulation average AUC)

| Train | AVG AUC | published | source |
|---|---:|---:|---|
| Deepfakes | not executed | 72.40 | — |
| Face2Face | not executed | 76.47 | — |
| FaceSwap | not executed | 70.64 | — |
| NeuralTextures | not executed | 75.93 | — |

## Table 6 (ablations)

| ID | AVG | published | source |
|---|---:|---:|---|
| C0–C9 | not executed | see AGENT_SPEC | — |

WGN versus the backbone (the C1 versus C0 comparison) was run on DFDC and did not favor WGN. C1 versus C2 and C1 versus C3 have not been tested.

## Seeds and variance

Two seeds, identical split, frame-level test on 16,600 frames. Checkpoint chosen by best validation accuracy.

## Discrepancies

On this DFDC split the backbone-only model matches or beats WGN. Test accuracy is higher for the backbone on both seeds (about 0.1 point). Test AUC differs by less than 0.1 point and favors a different model on each seed. That does not support the claim that the wavelet gate improves detection here.

## In-domain DFDC (the actual capstone result)

| Run | Seed | Val ACC | Val AUC | Test ACC | Test AUC | source |
|---|---:|---:|---:|---:|---:|---|
| WGN | 0 | 96.46 | 99.46 | 96.99 | 99.51 | `runs/wgn_dfdc_s0` |
| Backbone-only | 0 | 96.54 | 99.48 | 97.09 | 99.48 | `runs/baseline_dfdc_s0` |
| WGN | 1 | 96.31 | 99.25 | 96.83 | 99.52 | `runs/wgn_dfdc_s1` |
| Backbone-only | 1 | 96.47 | 99.46 | 96.96 | 99.58 | `runs/baseline_dfdc_s1` |

Validation numbers are the saved best checkpoint. Test numbers are that checkpoint on the held-out frames. No published target exists for in-domain DFDC training.

## DFDC checkpoints on other datasets

Same four checkpoints, no retraining. Frame-level. Celeb-DF is the official test list (5,180 frames); 65.64% of those frames are fake, so accuracy at or below that is not better than always answering "fake". HiDF is 86,975 frames and nearly balanced (majority accuracy 50.05%).

| Test set | Model | Seed | Test ACC | Test AUC | source |
|---|---|---:|---:|---:|---|
| Celeb-DF v2 | WGN | 0 | 62.97 | 56.40 | `runs/wgn_dfdc_on_celebdf_s0` |
| Celeb-DF v2 | Backbone-only | 0 | 65.27 | 64.15 | `runs/baseline_dfdc_on_celebdf_s0` |
| Celeb-DF v2 | WGN | 1 | 62.16 | 56.09 | `runs/wgn_dfdc_on_celebdf_s1` |
| Celeb-DF v2 | Backbone-only | 1 | 64.71 | 59.99 | `runs/baseline_dfdc_on_celebdf_s1` |
| HiDF | WGN | 0 | 55.86 | 59.48 | `runs/wgn_dfdc_on_hidf_s0` |
| HiDF | Backbone-only | 0 | 55.86 | 58.81 | `runs/baseline_dfdc_on_hidf_s0` |
| HiDF | WGN | 1 | 56.72 | 61.40 | `runs/wgn_dfdc_on_hidf_s1` |
| HiDF | Backbone-only | 1 | 57.11 | 61.36 | `runs/baseline_dfdc_on_hidf_s1` |

On Celeb-DF the backbone leads on both seeds (AUC gaps −7.75 and −3.90). WGN is near chance. On HiDF both models sit just above chance; the AUC gap is +0.67 and +0.04, so the wavelet gate does not separate them. Neither result is the published Celeb-DF AUC of 77.62.

## Next step

C2 (CBAM) is still untrained, so the attention comparison in the paper's Table 6 is open. FF++ tables stay blocked until the maintainer grants access.
