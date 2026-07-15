#!/bin/bash
# Full experiment suite: Symmetry-based data augmentation systematic study.
# All from scratch (pretrained=False) — weak baseline = room for augmentation to help.
# SI keywords hit: symmetry (main) + data efficiency (phase 2) + robustness (phase 3).
set -e
cd /mnt/e/Project/MDPI/symmetry-paper-2

echo "################################################"
echo "PHASE 1/4: MAIN — none/hflip/standard x CIFAR-10/100 x R50 (from scratch)"
echo "Started: $(date)"
echo "################################################"
python3 - <<'PYEOF'
from code.config import Config
from code.trainer import run_experiment, save_results
NC = {"cifar10": 10, "cifar100": 100}
for ds in ("cifar10", "cifar100"):
    for aug in ("none", "hflip", "standard"):
        cfg = Config(
            experiment="main", dataset=ds, model_name="resnet50",
            aug_strategy=aug, num_classes=NC[ds], epochs=50,
            pretrained=False,
            save_checkpoints=(ds == "cifar10"),  # need ckpts for robustness
        )
        r = run_experiment(cfg)
        save_results(r, cfg.output_dir)
print("[DONE] MAIN")
PYEOF

echo "################################################"
echo "PHASE 2/4: DATA EFFICIENCY (core selling point)"
echo "Started: $(date)"
echo "################################################"
python3 - <<'PYEOF'
from code.config import Config
from code.trainer import run_experiment, save_results
for frac in (0.05, 0.10, 0.25, 0.50):
    for aug in ("none", "hflip", "standard"):
        cfg = Config(
            experiment="efficiency", dataset="cifar100", model_name="resnet50",
            aug_strategy=aug, num_classes=100, epochs=40,
            pretrained=False, data_fraction=frac,
        )
        r = run_experiment(cfg)
        save_results(r, cfg.output_dir)
print("[DONE] EFFICIENCY")
PYEOF

echo "################################################"
echo "PHASE 3/4: ROBUSTNESS (CIFAR-10-C, uses phase-1 checkpoints)"
echo "Started: $(date)"
echo "################################################"
python3 -m code.exp4_robustness_scr

echo "################################################"
echo "PHASE 4/4: ABLATION — why vertical-flip/rotation fail"
echo "Started: $(date)"
echo "################################################"
python3 - <<'PYEOF'
from code.config import Config
from code.trainer import run_experiment, save_results
for aug in ("hflip", "vflip", "rot90", "rot15", "rot30"):
    cfg = Config(
        experiment="ablation_dir", dataset="cifar100", model_name="resnet50",
        aug_strategy=aug, num_classes=100, epochs=40, pretrained=False,
    )
    r = run_experiment(cfg)
    save_results(r, cfg.output_dir)
print("[DONE] ABLATION")
PYEOF

echo "################################################"
echo "[ALL DONE] $(date)"
echo "################################################"
