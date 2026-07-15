#!/usr/bin/env python3
"""
Symmetry Consistency Regularization (SCR) experiments.

Modes:
  verify     — quick sanity check: baseline vs SCR, short epochs (proves the method works)
  main       — full main experiment across datasets/models
  efficiency — data-fraction sweep (core selling point: data efficiency)
  ablation   — lambda weight sweep
"""
import sys, os, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.config import Config
from code.trainer import run_experiment, save_results

NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200}


def _save(result, cfg):
    save_results(result, cfg.output_dir)


def mode_verify():
    """Quick: CIFAR-10, ResNet-50, baseline vs SCR, 20 epochs each."""
    print("\n##### VERIFY MODE: baseline vs SCR (CIFAR-10, R50, 20ep) #####")
    base = Config(experiment="verify", dataset="cifar10", model_name="resnet50",
                  aug_strategy="standard", num_classes=10, epochs=20, use_scr=False)
    scr = Config(experiment="verify", dataset="cifar10", model_name="resnet50",
                 aug_strategy="standard", num_classes=10, epochs=20,
                 use_scr=True, lambda_cons=0.5)

    results = []
    for cfg in (base, scr):
        r = run_experiment(cfg)
        _save(r, cfg)
        results.append(r)

    print("\n" + "=" * 50)
    print("VERIFY RESULT")
    print("=" * 50)
    for r in results:
        tag = "SCR " if "SCR" in r["run_name"] else "base"
        print(f"  {tag}: Top1={r['best_top1']:.2f}%  ({r['train_time_min']}m)")
    delta = results[1]["best_top1"] - results[0]["best_top1"]
    print(f"  >>> SCR - baseline = {delta:+.2f}%")
    if delta > 0:
        print("  >>> SCR helps. Proceeding to full experiments.")
    else:
        print("  >>> WARNING: SCR did not help — tune lambda before full runs.")


def mode_main(epochs=50):
    """Main: baseline vs SCR x {CIFAR-10, CIFAR-100} x {R50, R101}."""
    print("\n##### MAIN MODE #####")
    results = []
    for ds in ("cifar10", "cifar100"):
        for model in ("resnet50", "resnet101"):
            for use_scr in (False, True):
                cfg = Config(
                    experiment="main", dataset=ds, model_name=model,
                    aug_strategy="standard", num_classes=NUM_CLASSES[ds],
                    epochs=epochs, use_scr=use_scr, lambda_cons=0.5,
                )
                r = run_experiment(cfg)
                _save(r, cfg)
                results.append(r)
    _summary(results)


def mode_efficiency(epochs=50):
    """Data efficiency: baseline vs SCR at 5/10/25/50/100% data (CIFAR-100, R50)."""
    print("\n##### DATA EFFICIENCY MODE (core selling point) #####")
    results = []
    for frac in (0.05, 0.10, 0.25, 0.50, 1.0):
        for use_scr in (False, True):
            cfg = Config(
                experiment="efficiency", dataset="cifar100", model_name="resnet50",
                aug_strategy="standard", num_classes=100, epochs=epochs,
                use_scr=use_scr, lambda_cons=0.5, data_fraction=frac,
            )
            r = run_experiment(cfg)
            _save(r, cfg)
            results.append(r)
    _summary(results)


def mode_ablation(epochs=50):
    """Lambda sweep on CIFAR-100, R50."""
    print("\n##### ABLATION: lambda weight #####")
    results = []
    for lam in (0.1, 0.3, 0.5, 1.0, 2.0):
        cfg = Config(
            experiment="ablation_lambda", dataset="cifar100", model_name="resnet50",
            aug_strategy="standard", num_classes=100, epochs=epochs,
            use_scr=True, lambda_cons=lam,
        )
        r = run_experiment(cfg)
        _save(r, cfg)
        results.append(r)
    _summary(results)


def _summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x["run_name"]):
        print(f"  {r['run_name']:45s} Top1={r['best_top1']:5.1f}%  ({r['train_time_min']}m)")
    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="verify",
                    choices=["verify", "main", "efficiency", "ablation", "all"])
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    if args.mode == "verify":
        mode_verify()
    elif args.mode == "main":
        mode_main(args.epochs)
    elif args.mode == "efficiency":
        mode_efficiency(args.epochs)
    elif args.mode == "ablation":
        mode_ablation(args.epochs)
    elif args.mode == "all":
        mode_verify()
        mode_main(args.epochs)
        mode_efficiency(args.epochs)
        mode_ablation(args.epochs)
