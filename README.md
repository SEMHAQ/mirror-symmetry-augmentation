# Mirror Symmetry as an Inductive Bias for Data-Efficient and Robust Image Classification

This repository contains the official experiment code accompanying the manuscript
*"Mirror Symmetry as an Inductive Bias for Data-Efficient and Robust Image Classification"*
submitted to the MDPI **Symmetry** Special Issue
*Symmetry in Computer Vision and Machine Learning*.

It provides a complete, reproducible pipeline for studying symmetry-group-based data
augmentation (mirror flip, vertical flip, rotations) for image classification under three
lenses: **standard accuracy**, **data efficiency**, and **robustness** (CIFAR-10-C).

---

## 1. Key results (ResNet-50, trained from scratch)

| Setting | None | hflip | standard |
|---|:---:|:---:|:---:|
| CIFAR-10 (full)        | 87.6 | 90.6 | 93.4 |
| CIFAR-100 (full)       | 60.0 | 68.0 | 73.3 |
| CIFAR-100 @ 25% data   | 34.0 | 41.8 | 51.9 |
| CIFAR-10-C mean corruption acc. | 71.2 | 72.0 | 80.0 |

Ablation (single geometric transform, CIFAR-100): `rot15` 68.3 ≈ `hflip` 67.2 ≈ `rot30` 67.7 > `rot90` 64.8 > `vflip` 61.1.

---

## 2. Environment

- Python ≥ 3.9
- PyTorch ≥ 2.0 with CUDA (tested on a single NVIDIA RTX 3090, 24 GB)
- `torchvision`, `timm`, `numpy`, `matplotlib`

```bash
pip install -r requirements.txt
```

---

## 3. Datasets

Datasets are **not** included (several GB). Place them under a `data/` directory:

| Dataset | Source | Expected path |
|---|---|---|
| CIFAR-10  | https://www.cs.toronto.edu/~kriz/cifar.html | `data/cifar-10-batches-py/` (auto-downloaded by torchvision) |
| CIFAR-100 | https://www.cs.toronto.edu/~kriz/cifar.html | `data/cifar-100-python/` |
| CIFAR-10-C | https://github.com/hendrycks/robustness | `data/CIFAR-10-C/*.npy` |

Set the data root by editing `DATA_DIR` / `PATHS["data"]` in `code/config.py` (default `/mnt/e/Project/MDPI/data`).

---

## 4. Repository structure

```
code/
├── config.py              # Experiment configuration & hyperparameters
├── augmentations.py       # Symmetry augmentation strategies (+ SymmetryAwareHFlip)
├── data.py                # Dataset / DataLoader (incl. data-fraction subsampling)
├── models.py              # CIFAR-adapted ResNet factory
├── trainer.py             # Training & evaluation loop (+ SCR consistency loss)
├── run_scr.py             # SCR experiments (verify / main / efficiency / ablation)
├── run_aug_all.sh         # One-shot full study: main + efficiency + robustness + ablation
├── exp4_robustness_scr.py # CIFAR-10-C robustness evaluation
├── make_figures.py        # Regenerate paper figures (efficiency curve, ablation bars)
└── prepare_data.py        # Extract Tiny-ImageNet / CIFAR-10-C archives
```

---

## 5. Reproducing the experiments

All experiments train ResNet-50 **from scratch** (no ImageNet pre-training) at native CIFAR
resolution (32×32), so each run is fast (~10–25 min on an RTX 3090).

```bash
cd code

# (A) Full study end-to-end (main + data-efficiency + robustness + ablation)
bash run_aug_all.sh

# (B) Or run individual phases
python -m run_scr --mode main        # baseline vs SCR, CIFAR-10/100 × R50
python -m run_scr --mode efficiency  # data-fraction sweep (the data-efficiency study)
python -m run_scr --mode ablation    # symmetry-direction ablation
python -m exp4_robustness_scr        # CIFAR-10-C robustness evaluation

# (C) Regenerate the paper figures from the reported numbers
python make_figures.py
```

Results are written as JSON to `results/`; checkpoints (for robustness eval) to `results/checkpoints/`.

---

## 6. Notes on reproducibility

- The default configuration uses a single random seed (`seed=42`). For publication-grade
  error bars, re-run each condition with multiple seeds and report mean ± std (see the
  accompanying peer-review for this recommendation).
- Mixed-precision (AMP) training is enabled by default.
- `run_aug_all.sh` saves best checkpoints only for the CIFAR-10 / ResNet-50 conditions
  needed by the robustness phase, to keep disk usage low.

---

## 7. License

The code is released under the MIT License for research purposes. The CIFAR and CIFAR-10-C
datasets retain their respective original licenses.
