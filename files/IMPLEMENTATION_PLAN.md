# WGN Implementation Plan

Reproducing *WGN: Wavelet-Guided Network for Efficient and Generalised Deepfake Detection* (Ghosh & Naskar, CVPRW 2026).

---

## Summary of the target

| Item | Value from paper |
|---|---|
| Backbone | MobileViT-S, input 3×256×256 |
| Feature map | C×H×W (640×8×8 for MobileViT-S @ 256²) |
| Total params | 5.7 M (backbone-only 4.94 M, +0.82 M from WGN) |
| FLOPs | 3.7 G |
| Headline result | FF++ C23: 95.32% ACC / 98.90% AUC |
| Generalisation | Celeb-DF 77.62% AUC (trained on FF++ C23) |
| Training | Adam, lr 1e-4, cosine annealing, BCE, batch 32, 25 epochs |
| Hardware | Single RTX A2000 (12 GB) |

The +0.82 M figure is a useful sanity check. It comes almost entirely from the fusion 1×1 conv (2C→C = 640×1280 = 819,200 weights); WGSA itself contributes only **292 parameters**. If your parameter count is far off, the fusion layer or the backbone truncation is wrong.

---

## Phase 0 — Environment (0.5 day)

```bash
pip install torch torchvision timm opencv-python facenet-pytorch \
            scikit-learn tqdm pillow
```

Confirm `timm.create_model("mobilevit_s", features_only=True, out_indices=(-1,))` returns 640 channels at 8×8 for a 256² input. If your timm version names it differently, check `timm.list_models("*mobilevit*")`.

**Deliverable:** environment that imports cleanly and can run a forward pass.

---

## Phase 1 — WGSA module (0.5 day) ✅ *already written and verified*

The provided `wgn/wgsa.py` implements Eqs. 2–10. Three properties were verified numerically:

- **Perfect reconstruction:** `IDWT(DWT(z)) == z` to 4.8e-7. This confirms the orthonormal Haar kernels are correct — if this test fails, your normalisation constant is wrong.
- **Energy preservation:** input and subband energy match (Parseval), another orthonormality check.
- **Gradient flow:** gradients reach the input feature map, confirming the fixed-buffer DWT stays differentiable.
- **Parameter count:** WGSA = 292 params (the 4→32→4 MLP), fusion head = 0.821 M. Matches the paper.

**Gotchas:**
- Register Haar kernels as *buffers*, not `nn.Parameter`. If they train, you no longer have a wavelet transform and the ablation story collapses.
- Min–max normalisation in Eq. 10 is over the whole spatial map **per sample**, not per batch. Batch-wide normalisation leaks information across samples and will inflate your validation numbers.
- The module needs even H and W. Fine at 8×8, but a different backbone or input size can give you odd dimensions.

**Deliverable:** unit tests for reconstruction, shape, range ⊂ [0,1], and gradient flow.

---

## Phase 2 — Full model (0.5 day) ✅ *already written and verified*

`wgn/model.py` wires backbone → WGSA → Eq. 11 modulation → Eq. 12 concat fusion → GAP → linear logit. Backbone-agnostic: pass any `nn.Module` returning `(B,C,H,W)` plus `num_channels`.

All ablation switches from Tables 5 and 6 are exposed as constructor arguments (`pooling`, `norm`, `use_ll`, `gating`, `fusion`, `backbone`), so the ablation study needs no code changes — only config changes.

**Deliverable:** forward pass at batch 32, 256², under 12 GB with AMP enabled.

---

## Phase 3 — Data pipeline (2–3 days) ⚠️ *the real cost*

This is where the time actually goes, not the model.

1. **Request FF++ access.** Fill the form at the FaceForensics GitHub repo; approval takes days, sometimes longer. **Start this first** — it gates everything else. Download only `c23` and `c40` at first; skip `raw` (that's the bulk of the ~1.5 TB).
2. **Grab the official split JSONs** (`train.json` / `val.json` / `test.json`) from the same repo. These give the 720/140/140 clip partition. Do not invent your own split — identity leakage across splits will inflate results by several AUC points.
3. **Run `wgn/preprocess.py`.** MTCNN face detection, 1.3× box expansion, 256×256 crops.

**Sampling ratio matters.** The paper takes 10 frames per fake clip and 40 per real clip. That is not arbitrary — FF++ has 1,000 real and 4,000 fake clips, so 10/40 yields 28,800 frames on each side. Change the ratio and you need class weighting in the loss, or your ACC becomes meaningless.

**Gotchas:**
- Crop *tighter than the full frame but looser than the face box*. Blending seams sit just outside the tight bounding box; crop too tight and you delete the exact evidence WGSA exists to find.
- Some clips have multiple detected faces. Take the highest-confidence box, and spot-check a sample by eye — for face-swap manipulations the wrong face is a silent label error.
- Expect a handful of clips where detection fails entirely. Log and skip rather than crash.
- Budget ~150–250 GB for the c23 crops and several hours of GPU time for detection.

**Deliverable:** crop tree with per-split, per-manipulation frame counts logged, plus a contact sheet of ~50 random crops that you have actually looked at.

---

## Phase 4 — Training and in-domain reproduction (2–3 days)

```bash
python -m wgn.train --data /data/ffpp_crops --epochs 25 --amp
```

Target: FF++ C23 ≈ 95.3% ACC / 98.9% AUC. Also train the backbone-only baseline (98.12% AUC in Table 1) — without it you cannot show WGSA is doing anything.

**Gotchas:**
- Keep augmentation mild. Blur, heavy JPEG, and sharpening destroy high-frequency evidence. This architecture is more augmentation-sensitive than a standard RGB CNN.
- Frame-level vs video-level metrics differ by a few points. The paper reports frame-level; be consistent when comparing.
- Checkpoint selection is on **best validation accuracy**, per Sec. 5.1.

**Deliverable:** Table 1 rows for WGN and the backbone baseline, matched within ~1 point.

---

## Phase 5 — Generalisation experiments (3–4 days)

**Cross-manipulation (Table 3):** four training runs, each on one manipulation, each evaluated on all four. 16 numbers. `--train-manip Deepfakes` etc.

**Cross-dataset (Table 2):** train on FF++ C23, test on Celeb-DF, DFDC, WildDeepfake. Each needs its own download and the *same* MTCNN preprocessing — a preprocessing mismatch between train and test is the single most common cause of collapsed cross-dataset numbers.

Expect more variance here than in-domain. FaceSwap columns in particular swing widely (the paper's own numbers range from 27% to 65% AUC across methods); treat a few points of disagreement as noise, not as a bug.

**Deliverable:** Tables 2 and 3 reproduced, with variance across at least 2 seeds noted.

---

## Phase 6 — Ablations and visualisation (2 days)

Table 6 variants map directly onto constructor flags:

| ID | Variant | Config |
|---|---|---|
| C0 | Backbone only | baseline model, no WGSA |
| C1 | WGN (proposed) | defaults |
| C3 | No gating | `wgsa_kwargs={"gating": False}` |
| C4 | Mean pooling | `wgsa_kwargs={"pooling": "mean"}` |
| C5 | Sigmoid norm | `wgsa_kwargs={"norm": "sigmoid"}` |
| C6 | High-freq only | `wgsa_kwargs={"use_ll": False}` |
| C8 / C9 | Add / multiply fusion | `fusion="add"` / `"multiply"` |

C2 (CBAM-style attention, no DWT) needs a small extra module — it is the most important ablation, since it is what separates "wavelets help" from "any attention helps."

Ablations use AdamW, 10 epochs, DF-only training.

Add Grad-CAM on the fused features plus a panel of `F`, `A`, and `F⊙A` to reproduce Figure 2. `model(x, return_attention=True)` already returns all three.

---

## Timeline

| Phase | Effort |
|---|---|
| 0–2 Environment + model | 1 day ✅ done |
| 3 Data pipeline | 2–3 days (+ dataset access wait) |
| 4 In-domain training | 2–3 days |
| 5 Generalisation | 3–4 days |
| 6 Ablations + figures | 2 days |
| **Total** | **~2 weeks** of working time, plus access latency |

On a single 12 GB GPU, one 25-epoch run over ~58k frames is roughly 3–5 hours. Phase 5 alone is 4 training runs plus 3 dataset evaluations, so plan the GPU schedule rather than running interactively.

---

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| FF++ access delayed | High — blocks everything | Submit the form on day 1; prototype on Celeb-DF or DFDC-preview meanwhile |
| Preprocessing mismatch train vs test | High — silently kills cross-dataset numbers | One shared crop function, asserted identical across datasets |
| Identity leakage across splits | High — inflates results | Use official split JSONs only |
| Cross-manipulation variance | Medium | Multiple seeds; report mean ± std |
| MobileViT-S weight variation across timm versions | Low | Pin the timm version; record it |

---

## What is genuinely uncertain

The paper does not state the gating MLP hidden width (32 is inferred from the figure), the exact augmentation policy, or the seed. Cross-manipulation FaceSwap numbers are the most likely to disagree. If the in-domain numbers land but FS transfer does not, the discrepancy is almost certainly in preprocessing or augmentation rather than in the WGSA module.
