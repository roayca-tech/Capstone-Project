# WGN Reproduction

PyTorch reproduction of Ghosh & Naskar, *WGN: Wavelet-Guided Network for
Efficient and Generalised Deepfake Detection* (CVPRW 2026).

## Start here

- **`AGENT_SPEC.md`** — full implementation specification with phases and
  acceptance criteria. Read this first.
- **`IMPLEMENTATION_PLAN.md`** — narrative plan with effort estimates and risks.

## Layout

```
wgn/wgsa.py         WGSA module (Eq. 2-10)        verified
wgn/model.py        WGN assembly (Eq. 1, 11-14)   verified
wgn/preprocess.py   FF++ frame extraction + MTCNN
wgn/train.py        Training / evaluation loop
tests/test_wgn.py   27 acceptance tests
```

## Setup

```bash
pip install -r requirements.txt
pytest tests/ -v          # must be 27 passed
```

## Verified invariants

| Property | Value |
|---|---|
| Haar perfect reconstruction error | < 1e-5 |
| WGSA learnable parameters | 292 |
| Parameters added over backbone | 0.821 M (paper: +0.82 M) |

## Data

FF++ requires approval via the [access form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform).
Start this first — it gates every training phase.

```bash
python -m wgn.preprocess --root $FFPP --out data/ffpp_c23 \
    --split-json splits/ --compression c23
python -m wgn.train --data data/ffpp_c23 --epochs 25 --amp
```
