#!/usr/bin/env python3
"""
Robustness evaluation on CIFAR-10-C.

Loads baseline and SCR checkpoints (trained on clean CIFAR-10 by the main experiment),
evaluates them on all 19 corruption types x 5 severity levels, and compares.
This directly addresses the SI keyword 'robustness'.
"""
import os, sys, json, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.models import build_model

CIFAR10C_DIR = "/mnt/e/Project/MDPI/data/CIFAR-10-C"
CKPT_DIR = "/mnt/e/Project/MDPI/symmetry-paper-2/results/checkpoints"
OUT = "/mnt/e/Project/MDPI/symmetry-paper-2/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "speckle_noise", "gaussian_blur", "saturate", "spatter",
]


def load_cifar10c(corruption, severity):
    data = np.load(f"{CIFAR10C_DIR}/{corruption}.npy")
    labels = np.load(f"{CIFAR10C_DIR}/labels.npy")
    s, e = (severity - 1) * 10000, severity * 10000
    x = torch.from_numpy(data[s:e]).float() / 255.0
    y = torch.from_numpy(labels[s:e]).long()
    x = x.permute(0, 3, 1, 2)
    x = (x - MEAN) / STD
    return x, y


def load_model(ckpt_path, model_name="resnet50", num_classes=10):
    model = build_model(model_name, num_classes, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model, x, y, batch_size=256):
    model.eval()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    correct, total = 0, 0
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        correct += (model(bx).argmax(1) == by).sum().item()
        total += by.size(0)
    return 100.0 * correct / total


def run():
    print(f"Device: {DEVICE}")
    # Compare robustness of models trained with different augmentations
    targets = [
        ("none",     "cifar10_resnet50_none.pth"),
        ("hflip",    "cifar10_resnet50_hflip.pth"),
        ("standard", "cifar10_resnet50_standard.pth"),
    ]
    models = {}
    for tag, fname in targets:
        path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(path):
            print(f"[SKIP] no checkpoint: {fname}")
            continue
        models[tag] = load_model(path)
        print(f"Loaded {tag}: {fname}")

    if not models:
        print("No checkpoints found. Run main experiment with save_checkpoints=True first.")
        return

    # Also evaluate clean CIFAR-10 test set
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN[0].tolist(), STD[0].tolist())])
    test = CIFAR10(root="/mnt/e/Project/MDPI/data", train=False, download=False, transform=tf)
    xt = torch.stack([d[0] for d in test])
    yt = torch.tensor([d[1] for d in test])

    results = {"corruptions": {}, "clean": {}}
    for tag, model in models.items():
        results["clean"][tag] = round(evaluate(model, xt, yt), 1)
        print(f"\n{tag} clean: {results['clean'][tag]}%")

    print("\nEvaluating on CIFAR-10-C (19 corruptions x 5 severities)...")
    for corr in CORRUPTIONS:
        results["corruptions"][corr] = {}
        for tag, model in models.items():
            sev_accs = []
            for sev in range(1, 6):
                x, y = load_cifar10c(corr, sev)
                sev_accs.append(round(evaluate(model, x, y), 1))
            avg = round(sum(sev_accs) / 5, 1)
            results["corruptions"][corr][tag] = {"per_severity": sev_accs, "avg": avg}
        b = results["corruptions"][corr].get("baseline", {}).get("avg", 0)
        s = results["corruptions"][corr].get("SCR", {}).get("avg", 0)
        print(f"  {corr:20s} baseline={b:5.1f}  SCR={s:5.1f}  delta={s-b:+.1f}")

    # Summary
    if "baseline" in models and "SCR" in models:
        b_all = np.mean([results["corruptions"][c]["baseline"]["avg"] for c in CORRUPTIONS])
        s_all = np.mean([results["corruptions"][c]["SCR"]["avg"] for c in CORRUPTIONS])
        results["mean_corruption_acc"] = {"baseline": round(b_all, 1), "SCR": round(s_all, 1)}
        print(f"\nMean corruption accuracy: baseline={b_all:.1f}%  SCR={s_all:.1f}%  delta={s_all-b_all:+.1f}")

    out_path = f"{OUT}/exp4_robustness.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
