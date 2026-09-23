# WGN Reproduction

PyTorch reproduction of Ghosh & Naskar, *WGN: Wavelet-Guided Network for
Efficient and Generalised Deepfake Detection* (CVPRW 2026).

## Start here

- **`capstone_progress.ipynb`** — results a reviewer can read on GitHub. Each cell loads the logs under `runs/` and the saved output is what GitHub renders.
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
wgn/dfdc.py              DFDC identity-safe splits + class balancing
wgn/train.py             train / eval, logs to runs/<tag>/
tests/                   53 tests (27 original + ablations, splits, DFDC)
scripts/prepare_dfdc.py  DFDC download -> crop tree
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

FF++ access was not granted, so training uses **DFDC** from the Kaggle
competition. See `runs/DEVIATIONS.md` for what that changes about the claims.

### DFDC (current path)

Accept the rules on the
[Deepfake Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge)
page, then pull a few of the 50 training parts (~10 GB each):

```bash
pip install kaggle          # put your token at ~/.kaggle/kaggle.json
for i in 00 01 02 03; do
    kaggle competitions download -c deepfake-detection-challenge \
        -f dfdc_train_part_$i.zip -p ~/data/dfdc
done
unzip '~/data/dfdc/*.zip' -d ~/data/dfdc
```

Plan the identity-safe split first, then extract crops:

```bash
python scripts/prepare_dfdc.py --root ~/data/dfdc --out data/dfdc --dry-run
python scripts/prepare_dfdc.py --root ~/data/dfdc --out data/dfdc
python scripts/verify_data.py --data data/dfdc --dataset dfdc
python scripts/contact_sheet.py --data data/dfdc --out contact_sheets/

python -m wgn.train --data data/dfdc --tag wgn_dfdc_s0 --seed 0 --epochs 25 --amp
python -m wgn.train --data data/dfdc --tag baseline_dfdc_s0 --seed 0 --no-wgsa --amp
```

`prepare_dfdc.py` keeps each fake with its source real clip so no face appears in
two splits, and samples more frames per real clip to balance DFDC's 1:5 class
skew. The crop tree it writes matches the FF++ layout, so `wgn.train` needs no
new flags.

### FF++ (if access ever arrives)

```bash
python -m wgn.preprocess --root $FFPP --out data/ffpp_c23 \
    --split-json splits/ --compression c23
python scripts/verify_data.py --data data/ffpp_c23
python -m wgn.train --data data/ffpp_c23 --tag wgn_c23_s0 --seed 0 --epochs 25 --amp
```

Cross-manipulation, ablations, and Grad-CAM commands are in `files/AGENT_SPEC.md`.

## Verified invariants

| Property | Value |
|---|---|
| Haar perfect reconstruction error | < 1e-5 |
| WGSA learnable parameters | 292 |
| Parameters added over backbone | 0.821 M (paper: +0.82 M) |
