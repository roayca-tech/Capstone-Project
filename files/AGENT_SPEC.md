# WGN Reproduction — Agent Implementation Specification

**Target paper:** Ghosh & Naskar, *WGN: Wavelet-Guided Network for Efficient and Generalised Deepfake Detection*, CVPRW 2026.
**Goal:** reproduce Tables 1–6 within tolerance.
**Audience:** an autonomous coding agent. Read this file completely before writing code.

---

## 0. Operating rules

1. **Run `pytest tests/ -v` after every change to `wgn/`.** All 27 tests must stay green. A red test is a correctness bug — fix it before proceeding, do not modify the test to pass.
2. **Do not edit `tests/test_wgn.py`** except to add tests. The existing assertions encode the paper's invariants.
3. **Never fabricate results.** If a run did not execute, report that it did not execute. Do not fill tables with plausible numbers.
4. **Log every run** to `runs/<tag>/` with the full config, git SHA, seed, and per-epoch metrics as JSON. A result without a reproducible config does not count as done.
5. **Stop and ask** if: dataset access is unavailable, a phase's acceptance criterion cannot be met after two genuine attempts, or the spec appears to contradict the paper.
6. Work phase by phase. Do not begin Phase N+1 until Phase N's acceptance criteria are met.

---

## 1. Current state

**Complete and verified** (do not rewrite without cause):

| File | Status |
|---|---|
| `wgn/wgsa.py` | WGSA module, Eqs. 2–10. Verified. |
| `wgn/model.py` | WGN assembly, Eqs. 1, 11–14. Verified. |
| `wgn/preprocess.py` | FF++ frame extraction + MTCNN. **Untested against real data.** |
| `wgn/train.py` | Training/eval loop. **Untested against real data.** |
| `tests/test_wgn.py` | 27 acceptance tests, all passing. |

**Verified numerical invariants** — these are ground truth, treat any deviation as a bug:

- Haar kernels orthonormal; `IDWT(DWT(z)) == z` to < 1e-5
- Subband energy equals input energy (Parseval)
- WGSA learnable parameters = **exactly 292** (the 4→32→4 MLP)
- WGN adds **0.821 M** parameters over the backbone, matching the paper's "+0.82M" (Table 6). Nearly all of it is the 640×1280 fusion conv, not the attention module.
- DWT/IDWT weights are non-trainable buffers but remain differentiable

**Not started:** Phases 3–7 below.

---

## 2. Architecture contract

Do not deviate from these without recording an explicit justification in `runs/DEVIATIONS.md`.

```
x (B,3,256,256)
  └─ MobileViT-S backbone, features_only, out_indices=(-1,)   → F (B,640,8,8)   Eq. 1
       └─ WGSA(F)                                              → A (B,1,8,8)
            ├─ channel pool (mean + max)/2                     → F_g (B,1,8,8)  Eq. 2
            ├─ fixed Haar DWT, stride-2 conv                   → LL,LH,HL,HH    Eq. 3–5
            ├─ spatial means                                   → s (B,4)        Eq. 6
            ├─ MLP 4→32→4 + sigmoid                            → w (B,4)        Eq. 7
            ├─ subband reweighting                             →                Eq. 8
            ├─ IDWT, transposed conv                           → A_raw          Eq. 9
            └─ per-sample min–max normalisation                → A ∈ [0,1]      Eq. 10
       ├─ F_att = F ⊙ A                                                          Eq. 11
       ├─ fuse: 1×1 conv([F ‖ F_att]) → BN → ReLU              → Z (B,640,8,8)  Eq. 12
       ├─ GAP                                                  → z (B,640)      Eq. 13
       └─ Linear(640,1)                                        → logit          Eq. 14
```

### Hard constraints — violating any of these invalidates the reproduction

| Constraint | Why |
|---|---|
| Haar kernels are `register_buffer`, never `nn.Parameter` | If they train it is no longer a wavelet transform, and the C2 ablation loses meaning |
| Min–max normalisation is **per sample**, over the spatial map | Batch-wide normalisation leaks across samples and inflates validation scores |
| Official FF++ split JSONs only | Identity leakage across splits inflates AUC by several points |
| 10 frames/fake clip, 40 frames/real clip | Yields 28,800 per class. Changing it silently unbalances the classes |
| Identical preprocessing for train and all test datasets | The top cause of collapsed cross-dataset numbers |
| Checkpoint selected on **best validation accuracy** | Paper, Sec. 5.1 |
| Mild augmentation only | Blur/sharpen/heavy JPEG destroy the high-frequency evidence WGSA reads |

---

## 3. Phase 3 — Data pipeline

**Blocking dependency:** FF++ access via the Google Form (https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform). The maintainers email a download script on approval. If the data is absent, **stop and report** — do not substitute a Kaggle mirror without explicit approval, since mirrors have unknown re-encoding that corrupts frequency-domain evidence.

### Tasks

1. Download FF++ `c23` and `c40` (skip `raw`). Download FaceShifter at both levels. Budget ~500 GB.
2. Place the official `train.json` / `val.json` / `test.json` in `splits/`.
3. Run preprocessing for each compression level:
   ```bash
   python -m wgn.preprocess --root $FFPP --out data/ffpp_c23 \
       --split-json splits/ --compression c23
   ```
4. Write `scripts/verify_data.py` producing a report with: frame counts per split/manipulation, class balance, count of clips where detection failed, and mean/std of pixel intensities.
5. Write `scripts/contact_sheet.py` producing a PNG grid of 64 random crops per manipulation.

### Acceptance criteria

- [ ] Train split: real ≈ 28,800 frames; each manipulation ≈ 7,200 → fake total ≈ 28,800. Class ratio within 1.05.
- [ ] Detection failure rate < 2% of clips; failures logged, not silently dropped.
- [ ] All crops exactly 256×256 RGB PNG.
- [ ] Zero clip-ID overlap between train/val/test. Assert this programmatically — it is the highest-severity silent failure mode.
- [ ] Contact sheets generated and **visually inspected**. For face-swap manipulations, confirm the correct face was cropped; the wrong face is a silent label error.

---

## 4. Phase 4 — In-domain reproduction

### Runs

| Tag | Command |
|---|---|
| `baseline_c23` | backbone only, no WGSA, FF++ C23 |
| `wgn_c23` | `python -m wgn.train --data data/ffpp_c23 --epochs 25 --amp` |
| `wgn_c40` | same on `data/ffpp_c40` |
| `wgn_faceshifter_c23`, `wgn_faceshifter_c40` | FaceShifter |

Config: Adam, lr 1e-4, cosine annealing, BCE, batch 32, 25 epochs, 256².

> **Note:** `wgn/train.py` currently trains WGN only. Add a `--no-wgsa` flag or a `BaselineNet` for the backbone-only row. Reuse the same fusion-free head (GAP → Linear) so the comparison is clean.

### Targets (Table 1)

| Run | ACC | AUC |
|---|---|---|
| WGN FF++ C23 | 95.32% | 98.90% |
| Backbone FF++ C23 | 93.70% | 98.12% |
| WGN FF++ C40 | 80.20% | 89.00% |
| WGN FaceShifter C23 | 98.00% | 99.92% |

### Acceptance criteria

- [ ] `wgn_c23` within **±1.0 pt ACC and ±0.5 pt AUC** of target.
- [ ] WGN beats the backbone baseline on C23 AUC. **If it does not, stop.** The entire paper rests on this gap; a null result here means a pipeline bug, most likely in preprocessing.
- [ ] Two seeds per run; report mean ± std.
- [ ] Metrics are frame-level (state this explicitly in the report — video-level aggregation differs by several points).

---

## 5. Phase 5 — Generalisation

### 5a. Cross-manipulation (Table 3) — 4 runs, 16 numbers

```bash
for M in Deepfakes Face2Face FaceSwap NeuralTextures; do
  python -m wgn.train --data data/ffpp_c23 --train-manip $M --epochs 25 --amp
done
```
Each trains on one manipulation and evaluates on all four. Target average AUC: DF 72.40, F2F 76.47, FS 70.64, NT 75.93.

### 5b. Cross-dataset (Table 2)

Take the `wgn_c23` checkpoint (all four manipulations) and evaluate on Celeb-DF, DFDC, WildDeepfake. Each needs its own download and **the identical MTCNN pipeline**. Write `scripts/preprocess_external.py` reusing `crop_face()` from `wgn/preprocess.py` — do not reimplement cropping.

Targets: Celeb-DF 77.62%, DFDC 70.41%, WildDeepfake 68.65% AUC.

### Acceptance criteria

- [ ] Celeb-DF within ±3 pts. This is the paper's headline generalisation claim.
- [ ] Cross-manipulation averages within ±3 pts.
- [ ] FaceSwap transfer columns may deviate more — published FS numbers range from 27% to 65% AUC across methods, so treat several points as noise, not a bug.
- [ ] A single shared crop function is used across all datasets; assert this in code review.

---

## 6. Phase 6 — Ablations (Table 6)

Config differs from the main runs: **AdamW, 10 epochs, Deepfakes-only training**, evaluated on all four test sets.

| ID | Variant | Config | Target AVG |
|---|---|---|---|
| C0 | Backbone only | no WGSA | 64.96 |
| C1 | WGN proposed | defaults | 66.73 |
| C2 | CBAM-style attention, no DWT | **must be implemented** | 64.71 |
| C3 | No gating | `wgsa_kwargs={"gating": False}` | 64.16 |
| C4 | Mean pooling | `wgsa_kwargs={"pooling": "mean"}` | 65.51 |
| C5 | Sigmoid norm | `wgsa_kwargs={"norm": "sigmoid"}` | 65.76 |
| C6 | High-frequency only | `wgsa_kwargs={"use_ll": False}` | 66.23 |
| C7 | HH band only | **must be implemented** (add `bands` arg) | 64.88 |
| C8 | Additive fusion | `fusion="add"` | 65.49 |
| C9 | Multiply-only fusion | `fusion="multiply"` | 65.22 |

C3–C6, C8, C9 need **no code changes** — they are constructor flags already exposed and tested.

**C2 is the most important ablation.** It replaces the wavelet path with CBAM-style channel pooling plus a 2D conv, keeping the parameter count matched at 5.76 M. It is what separates "wavelets help" from "any attention helps." Implement it as `wgn/baselines.py::CBAMAttention` with the same `(B,C,H,W) → (B,1,H,W)` interface as `WGSA`, so it drops into `WGN` unchanged.

**C7** requires a `bands` argument on `WGSA` (e.g. `bands="hh"`). Add it, and add a test mirroring `test_use_ll_false_zeroes_approximation_band`.

### Acceptance criteria

- [ ] C1 > C0 (WGSA beats the raw backbone).
- [ ] C1 > C2 (wavelet structure beats generic attention). **This is the paper's core scientific claim** — if it fails, report it prominently rather than tuning until it passes. A genuine negative result here is a legitimate and publishable finding.
- [ ] C1 > C3 (gating matters).
- [ ] All ten variants trained under an identical pipeline; only the listed flag differs.

---

## 7. Phase 7 — Visualisation and reporting

1. `scripts/gradcam.py` — Grad-CAM on the fused features. `model(x, return_attention=True)` already returns `logit, A, F, F⊙A`. Reproduce Figure 2's six-column layout across one real and four manipulated inputs.
2. `scripts/make_tables.py` — read `runs/*/metrics.json`, emit markdown tables matching the paper's layout with a **delta column** against published numbers.
3. `REPORT.md` — every table, seeds and variance, all deviations, and an explicit list of what did **not** reproduce.

### Acceptance criteria

- [ ] Every reported number traces to a `runs/` directory with its config.
- [ ] Discrepancies > tolerance are listed with a hypothesis, not hidden.
- [ ] Figure 2 qualitatively reproduced: attention maps should be structured, not uniform or saturated.

---

## 8. Known failure modes

| Symptom | Most likely cause |
|---|---|
| Val AUC > 99.5% on C23, poor cross-dataset | Identity leakage — splits not from official JSONs |
| WGN ≈ backbone, no gain | WGSA output near-constant; check gate saturation and that Eq. 11 is applied |
| Attention map all 0 or all 1 | Min–max normalisation over the wrong axes, or `eps` too large |
| Parameter count ≠ 5.7 M | Backbone classifier head not stripped, or wrong `out_indices` |
| Cross-dataset AUC ≈ 50% | Preprocessing mismatch between train and test crops |
| Loss not decreasing | lr too high for a pretrained backbone; confirm 1e-4, and that pretrained weights loaded |
| CUDA OOM on 12 GB | Enable `--amp`; batch 32 at 256² should fit |
| `test_minmax_normalisation_is_per_sample` fails | Normalising across the batch — a silent scientific error, not a style issue |

---

## 9. Compute budget

One 25-epoch run over ~58k frames ≈ 3–5 h on a 12 GB GPU. Total across all phases: **~20 training runs**, roughly 80–120 GPU-hours. Schedule as batch jobs; do not run interactively.

---

## 10. Open questions in the paper

Record whatever you choose in `runs/DEVIATIONS.md`:

1. Gating MLP hidden width is not stated; 32 is inferred from Figure 1.
2. Augmentation policy is unspecified. Start minimal (flip only) and add only if in-domain targets are missed.
3. Seeds are not reported; variance across seeds is unknown.
4. Table 1 does not state frame-level vs video-level metrics. Assume frame-level.
5. `eps` in Eq. 10 is unspecified; 1e-6 is used.

If in-domain numbers reproduce but FaceSwap transfer does not, the discrepancy is almost certainly in preprocessing or augmentation rather than in the WGSA module — that module is verified correct against the paper's equations.
