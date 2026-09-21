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
| 3 Data pipeline | **DFDC path ready, not yet run.** `scripts/prepare_dfdc.py` plans identity-safe splits and balanced frame budgets; verified end to end on synthetic data. Awaiting the Kaggle download. FF++ path remains unused. |
| 4 In-domain training | **Not executed.** `--no-wgsa`, run logging, and seed flags are implemented. |
| 5 Generalisation | **Not executed**, and largely out of scope without FF++. Celeb-DF v2 requested via the authors' form. |
| 6 Ablations | Code ready (C2 `CBAMAttention`, C7 `bands="hh"`, C0 `BaselineNet`). No ablation has been trained. C1 vs C0 and C1 vs C2 are still runnable on DFDC. |
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

No training job was started. There is no `runs/<tag>/metrics.json` from a real fit, so no table below can be filled.

The DFDC planner and the data audit were exercised end to end on synthetic metadata only. No real video has been decoded.

FF++ remains gated on the [maintainer access form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform); no mirror was substituted for it. Celeb-DF and WildDeepfake are absent.

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

C1 > C0, C1 > C2, and C1 > C3 have **not been tested on real data**.

## Seeds and variance

No multi-seed run has executed. Planned tags: `*_s0` and `*_s1`.

## Discrepancies

None to report: no measured number exists yet.

## In-domain DFDC (the actual capstone result)

| Run | ACC | AUC | source |
|---|---:|---:|---|
| WGN DFDC | not executed | not executed | — |
| Backbone-only DFDC | not executed | not executed | — |

No published target exists for this comparison; the claim under test is that WGN beats the backbone under an identical pipeline.

## Next step

1. Accept the rules on the [Kaggle DFDC competition](https://www.kaggle.com/competitions/deepfake-detection-challenge) and download 3–4 training parts (~40 GB).
2. `python scripts/prepare_dfdc.py --root ~/data/dfdc --out data/dfdc --dry-run` and check the printed split and frame budget.
3. `python scripts/prepare_dfdc.py --root ~/data/dfdc --out data/dfdc`
4. `python scripts/verify_data.py --data data/dfdc --dataset dfdc`
5. Inspect `scripts/contact_sheet.py` output by eye; a wrong cropped face is a silent label error.
6. Train WGN and the backbone baseline on the same split, two seeds each.
