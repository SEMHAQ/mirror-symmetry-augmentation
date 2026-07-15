#!/bin/bash
# Revision experiments responding to round-1 review.
# Addresses: C1 (standard-minus-flip control), M1/M5 (multi-seed error bars),
# robustness isolation. All from scratch, ResNet-50, native CIFAR.
set -e
cd /mnt/e/Project/MDPI/symmetry-paper-2

echo "################################################"
echo "PHASE 1: standard-minus-flip CONTROL (CIFAR-100) — isolates flip contribution"
echo "Started: $(date)"
echo "################################################"
python3 - <<'PYEOF'
from code.config import Config
from code.trainer import run_experiment, save_results
# Control: none / hflip / standard_noflip / standard at full, 25%, 50% data
for frac in (1.0, 0.25, 0.50):
    for aug in ("none", "hflip", "standard_noflip", "standard"):
        cfg = Config(experiment="control", dataset="cifar100", model_name="resnet50",
                     aug_strategy=aug, num_classes=100, epochs=50,
                     pretrained=False, data_fraction=frac, seed=42)
        r = run_experiment(cfg); save_results(r, cfg.output_dir)
print("[DONE] PHASE 1 control")
PYEOF

echo "################################################"
echo "PHASE 2: CIFAR-10 control + checkpoints for robustness"
echo "Started: $(date)"
echo "################################################"
python3 - <<'PYEOF'
from code.config import Config
from code.trainer import run_experiment, save_results
for aug in ("none", "hflip", "standard_noflip", "standard"):
    cfg = Config(experiment="control", dataset="cifar10", model_name="resnet50",
                 aug_strategy=aug, num_classes=10, epochs=50,
                 pretrained=False, save_checkpoints=True, seed=42)
    r = run_experiment(cfg); save_results(r, cfg.output_dir)
print("[DONE] PHASE 2 cifar10 control + ckpt")
PYEOF

echo "################################################"
echo "PHASE 3: MULTI-SEED on CIFAR-100 full (seeds 1, 2 add to seed 42 => 3 seeds)"
echo "Started: $(date)"
echo "################################################"
python3 - <<'PYEOF'
from code.config import Config
from code.trainer import run_experiment, save_results
for seed in (1, 2):
    for aug in ("none", "hflip", "standard_noflip", "standard"):
        cfg = Config(experiment="control_ms", dataset="cifar100", model_name="resnet50",
                     aug_strategy=aug, num_classes=100, epochs=50,
                     pretrained=False, seed=seed)
        r = run_experiment(cfg); save_results(r, cfg.output_dir)
print("[DONE] PHASE 3 multiseed")
PYEOF

echo "################################################"
echo "PHASE 4: ROBUSTNESS (CIFAR-10-C, none/hflip/standard_noflip/standard)"
echo "Started: $(date)"
echo "################################################"
python3 -m code.exp4_robustness_scr

echo "################################################"
echo "[ALL DONE] $(date)"
echo "################################################"
