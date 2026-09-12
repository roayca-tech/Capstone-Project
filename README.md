# WGN Reproduction

PyTorch reproduction of Ghosh & Naskar, *WGN: Wavelet-Guided Network for
Efficient and Generalised Deepfake Detection* (CVPRW 2026).

## Start here

- **`files/AGENT_SPEC.md`** — implementation specification, phases, acceptance criteria.
- **`files/IMPLEMENTATION_PLAN.md`** — narrative plan, risks, effort.
- **`REPORT.md`** — what has and has not been run.
- **`runs/DEVIATIONS.md`** — paper open questions and recorded choices.

## Layout

```
wgn/wgsa.py              WGSA (Eq. 2-10) + bands= for C7
wgn/model.py             WGN (Eq. 1, 11-14) + BaselineNet
wgn/baselines.py         CBAMAttention (ablation C2)
wgn/preprocess.py        FF++ frames + shared crop_face()
wgn/train.py             train / eval, logs to runs/<tag>/
tests/test_wgn.py        original 27 acceptance tests + C7
scripts/verify_data.py
scripts/contact_sheet.py
scripts/preprocess_external.py   Celeb-DF / DFDC / WildDeepfake
scripts/gradcam.py
scripts/make_tables.py
splits/                  official FF++ train/val/test.json
```

## Setup

```bash
pip install -r requirements.txt
pytest tests/ -v          # must stay green
```

## Data

FF++ requires approval via the
[access form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform).
Do not substitute a Kaggle mirror. After the videos are on disk:

```bash
python -m wgn.preprocess --root $FFPP --out data/ffpp_c23 \
    --split-json splits/ --compression c23
python scripts/verify_data.py --data data/ffpp_c23
python scripts/contact_sheet.py --data data/ffpp_c23 --out contact_sheets/

python -m wgn.train --data data/ffpp_c23 --tag wgn_c23_s0 --seed 0 --epochs 25 --amp
python -m wgn.train --data data/ffpp_c23 --tag baseline_c23_s0 --seed 0 --no-wgsa --amp
```

Cross-manipulation, ablations, and Grad-CAM commands are in `files/AGENT_SPEC.md`.

## Verified invariants

| Property | Value |
|---|---|
| Haar perfect reconstruction error | < 1e-5 |
| WGSA learnable parameters | 292 |
| Parameters added over backbone | 0.821 M (paper: +0.82 M) |
