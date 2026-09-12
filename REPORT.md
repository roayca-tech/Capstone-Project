# WGN reproduction report

**Paper:** Ghosh & Naskar, *WGN: Wavelet-Guided Network for Efficient and Generalised Deepfake Detection*, CVPRW 2026.

This report is honest about execution. Numbers that were not produced by a run are marked **not executed**. They are not filled in.

## Status

| Phase | Status |
|---|---|
| 0–2 Environment + WGSA + WGN | Code in place. Unit tests cover the paper invariants. |
| 3 Data pipeline | **Blocked.** Official split JSONs are in `splits/`. FF++ / FaceShifter videos are not on this machine, so preprocessing has not run. |
| 4 In-domain training | **Not executed.** `--no-wgsa`, run logging, and seed flags are implemented. |
| 5 Generalisation | **Not executed.** `scripts/preprocess_external.py` is written and reuses `wgn.preprocess.crop_face`. |
| 6 Ablations | Code ready (C2 `CBAMAttention`, C7 `bands="hh"`, C0 `BaselineNet`). No ablation has been trained. |
| 7 Visualisation / tables | Scripts exist. `scripts/make_tables.py` emits "not executed" for every row until `runs/*/metrics.json` appear. |

## What is implemented

- `wgn/` package: WGSA, WGN, BaselineNet, CBAMAttention, preprocess, train.
- Official FF++ `train.json` / `val.json` / `test.json` (360 / 70 / 70 pairs).
- Failure logging in `wgn.preprocess` (`failures.json`, not silent drops).
- `scripts/verify_data.py`, `contact_sheet.py`, `preprocess_external.py`, `gradcam.py`, `make_tables.py`.
- Every training run writes `runs/<tag>/config.json` (argv, git SHA, seed) and `runs/<tag>/metrics.json` (per-epoch + frame-level test).
- Recorded paper ambiguities: `runs/DEVIATIONS.md`.

## What did not execute

No training job was started. There is no `runs/<tag>/metrics.json` from a real fit, so Tables 1–6 cannot be filled.

FF++ is gated on the [maintainer access form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform). A Kaggle mirror was **not** used.

Celeb-DF, DFDC, and WildDeepfake are also absent.

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

## Next step

1. Obtain FF++ `c23` / `c40` and FaceShifter through the official form.
2. `python -m wgn.preprocess --root $FFPP --out data/ffpp_c23 --split-json splits/ --compression c23`
3. `python scripts/verify_data.py --data data/ffpp_c23`
4. Visually inspect `scripts/contact_sheet.py` output, especially FaceSwap.
5. Only then start Phase 4 training.
