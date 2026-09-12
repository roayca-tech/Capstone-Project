# Reproduction tables

Runs discovered: 0. Every number traces to a `runs/<tag>/` directory or is marked **not executed**.

### Table 1 — in-domain (frame-level)

| Run | ACC | AUC | Δ ACC | Δ AUC | source |
|---|---:|---:|---:|---:|---|
| WGN FF++ C23 | not executed | not executed | not executed | not executed | `—` |
| Backbone FF++ C23 | not executed | not executed | not executed | not executed | `—` |
| WGN FF++ C40 | not executed | not executed | not executed | not executed | `—` |
| WGN FaceShifter C23 | not executed | not executed | not executed | not executed | `—` |

Metrics are **frame-level**. Video-level aggregation is not reported.

### Table 2 — cross-dataset AUC (trained on FF++ C23, frame-level)

| Dataset | AUC | published | Δ | source |
|---|---:|---:|---:|---|
| celebdf | not executed | 77.62 | not executed | `—` |
| dfdc | not executed | 70.41 | not executed | `—` |
| wilddeepfake | not executed | 68.65 | not executed | `—` |

### Table 3 — cross-manipulation average AUC (frame-level)

| Train | AVG AUC | published | Δ | source |
|---|---:|---:|---:|---|
| Deepfakes | not executed | 72.40 | not executed | `—` |
| Face2Face | not executed | 76.47 | not executed | `—` |
| FaceSwap | not executed | 70.64 | not executed | `—` |
| NeuralTextures | not executed | 75.93 | not executed | `—` |

### Table 6 — ablations (AdamW, 10 epochs, Deepfakes-only, AVG)

| ID | AVG | published | Δ | source |
|---|---:|---:|---:|---|
| C0 | not executed | 64.96 | not executed | `—` |
| C1 | not executed | 66.73 | not executed | `—` |
| C2 | not executed | 64.71 | not executed | `—` |
| C3 | not executed | 64.16 | not executed | `—` |
| C4 | not executed | 65.51 | not executed | `—` |
| C5 | not executed | 65.76 | not executed | `—` |
| C6 | not executed | 66.23 | not executed | `—` |
| C7 | not executed | 64.88 | not executed | `—` |
| C8 | not executed | 65.49 | not executed | `—` |
| C9 | not executed | 65.22 | not executed | `—` |
