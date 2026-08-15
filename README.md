# DriftSense-AI
**I4C SEMICON India Hackathon 2026 — Problem Statement 2**
*Semiconductor Wafer Pattern Localization*

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` is optional. Without it, the FinFET Siamese CNN path will automatically
> fall back to the Phase Correlation heuristic. DRAM localization is unaffected.

---

### 2. Generate a Sample Dataset

```bash
# Generate 20 pairs (DRAM + FinFET mixed) in ./sample_dataset
python generate_dataset.py --arch BOTH --num 20 --out ./sample_dataset --seed 42

# Generate DRAM only
python generate_dataset.py --arch DRAM --num 30 --out ./dataset_dram

# Generate FinFET only
python generate_dataset.py --arch FINFET --num 30 --out ./dataset_finfet
```

Each pair is written to `<out>/<split>/pair_XXXXXX/`:
```
pair_000000/
  reference.png        ← degraded reference crop (64–128 px)
  search.png           ← 1000×1000 search image with pattern inside
  ground_truth.json    ← true center_x, center_y + metadata
```

---

### 3. Run Localization Inference

```bash
# Basic (competition-safe output: "x y")
python inference.py reference.png search.png

# JSON output
python inference.py reference.png search.png --json

# Verbose (debug info to stderr)
python inference.py reference.png search.png --verbose

# Override architecture
python inference.py reference.png search.png --arch FINFET
```

**Output format (stdout):**
```
632 417
```
or with `--json`:
```json
{"x": 632, "y": 417, "confidence": 0.912}
```

---

### 4. Train the Optional Siamese CNN (FinFET improvement)

```bash
# Step 1: Generate a FinFET training dataset
python generate_dataset.py --arch FINFET --num 85 --out ./dataset_finfet

# Step 2: Generate + Train in one command
python train_siamese.py --gen --dataset ./dataset_finfet --data siamese_train.npz --out finfet_siamese_cpu.pth

# Or separately:
python train_siamese.py --gen --dataset ./dataset_finfet --data siamese_train.npz  # generate data only
python train_siamese.py --data siamese_train.npz --out finfet_siamese_cpu.pth      # train only
```

The trained weights (`finfet_siamese_cpu.pth`) are automatically loaded by `inference.py` when they exist in the same directory.

---

## Repository Contents

| # | File | Description |
|---|------|-------------|
| 1 | `README.md` | This file — setup and usage instructions |
| 2 | `generate_dataset.py` | Standalone synthetic dataset generator (DRAM/FinFET) |
| 3 | `inference.py` | Standalone localization inference script |
| 4 | `finfet_siamese_cpu.pth` | Pretrained Siamese CNN weights (optional, FinFET only) |
| 5 | `train_siamese.py` | Siamese CNN training script |
| 6 | `requirements.txt` | Python dependencies |
| 7 | `references.md` | Citation documents and references |

---

## Algorithm Overview

### DRAM Path (v1.0)
1. Preprocessing: Gaussian denoise → CLAHE → Sobel gradients
2. Multi-scale NCC sweep (9 scales, 0.90–1.10×)
3. Global-max candidate per scale (ensures GT never NMS-suppressed)
4. High-confidence fast path: if NCC margin is clear (>0.001), return top candidate directly
5. Context NCC (3× expanded window) to disambiguate periodic aliases
6. FFT periodicity scoring + weighted ranking
7. Local pixel-level refinement (15px radius)

### FinFET Path (v1.1/v1.2)
1. Same preprocessing as DRAM
2. Edge-heavy candidate generation (w_edge=0.65) for gate-line discrimination
3. Denser scale sweep, smaller NMS radius, higher top-K
4. **Optional Siamese CNN scoring** (if `finfet_siamese_cpu.pth` exists)
5. Fallback to Phase Correlation if Siamese not available
6. FinFET-specific ranking weights (neighborhood-dominant)
7. Local refinement

> **FinFET Fundamental Limit:** Due to extreme spatial periodicity, classical NCC reaches a ~33% Acc@5px ceiling. The true location is information-theoretically indistinguishable from alias locations using only the small reference crop, as no macro-structural landmarks are visible in the field of view.

---

## Ground Truth Format

`ground_truth.json` in each pair directory:
```json
{
  "pair_id":      "pair_000000",
  "arch":         "DRAM",
  "center_x":     412,
  "center_y":     317,
  "search_width":  1000,
  "search_height": 1000,
  "ref_width":     96,
  "ref_height":    96,
  "rotation_deg":  1.23,
  "scale":         0.97,
  "noise_sigma":   8.4,
  "blur_type":     "motion",
  "seed":          42000
}
```

---

## Performance (45-pair held-out test set)

| Method | Acc@5px | DRAM Acc@5px | FinFET Acc@5px | Mean Error |
|--------|---------|--------------|----------------|------------|
| Baseline NCC | 48.9% | 87.0% | 9.1% | 158.6 px |
| Multi-scale NCC | 53.3% | 95.7% | 9.1% | 149.1 px |
| **DriftSense-AI v1.2** | **53.3%** | **91.3%** | **13.6%** | **163.9 px** |

---

## License

MIT License. See `LICENSE` for details.
