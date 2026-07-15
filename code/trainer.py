"""Training and evaluation loops."""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from . import models as model_utils


class AverageMeter:
    """Running average counter."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def consistency_loss(logits, logits_flip, ctype="symmetric_kl"):
    """
    Mirror-symmetry consistency: prediction should be invariant to horizontal flip,
    since image-class labels are mirror-invariant (a flipped dog is still a dog).

    Implements a symmetric divergence between p(y|x) and p(y|flip(x)).
    """
    p = F.softmax(logits, dim=1)
    q = F.softmax(logits_flip, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    log_q = F.log_softmax(logits_flip, dim=1)

    if ctype == "symmetric_kl":
        kl_pq = F.kl_div(log_q, p, reduction="batchmean")
        kl_qp = F.kl_div(log_p, q, reduction="batchmean")
        return 0.5 * (kl_pq + kl_qp)
    elif ctype == "js":
        m = 0.5 * (p + q)
        js = 0.5 * (F.kl_div(log_p, m, reduction="batchmean") +
                    F.kl_div(log_q, m, reduction="batchmean"))
        return js
    elif ctype == "mse":
        return F.mse_loss(p, q)
    else:
        raise ValueError(f"Unknown consistency type: {ctype}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, config):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    cons_losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward with mixed precision
        with autocast(enabled=config.mixed_precision):
            outputs = model(inputs)
            loss_cls = criterion(outputs, targets)

            # Mirror-symmetry consistency regularization
            if config.use_scr:
                inputs_flip = torch.flip(inputs, dims=[3])  # horizontal flip (W axis)
                outputs_flip = model(inputs_flip)
                loss_cons = consistency_loss(outputs, outputs_flip, config.cons_type)
                loss = loss_cls + config.lambda_cons * loss_cons
            else:
                loss_cons = torch.zeros(1, device=device)
                loss = loss_cls

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        prec1, = accuracy(outputs, targets, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        cls_losses.update(loss_cls.item(), inputs.size(0))
        cons_losses.update(loss_cons.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # Logging
        if batch_idx % config.log_interval == 0:
            cons_str = f" | cons {cons_losses.val:.4f}" if config.use_scr else ""
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                  f"Loss {losses.val:.4f} ({losses.avg:.4f}) | "
                  f"cls {cls_losses.val:.4f}{cons_str} | "
                  f"Acc {top1.val:.2f} ({top1.avg:.2f}) | "
                  f"Time {time.time()-end:.1f}s")
            end = time.time()

    return losses.avg, top1.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    return {
        "loss": losses.avg,
        "top1": top1.avg,
        "top5": top5.avg,
    }


def run_experiment(config):
    """
    Run a single training experiment and return results.

    Returns dict with results.
    """
    print(f"\n{'='*60}")
    print(f"RUN: {config.run_name}")
    print(f"  Dataset:    {config.dataset}")
    print(f"  Model:      {config.model_name}")
    print(f"  Aug:        {config.aug_strategy}")
    print(f"  Epochs:     {config.epochs}")
    print(f"{'='*60}")

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data — native resolution (32x32 CIFAR, 64x64 Tiny ImageNet)
    INPUT_SIZE = 64 if config.dataset == "tinyimagenet" else 32
    from .data import get_dataloader
    train_loader = get_dataloader(
        config.dataset, config.aug_strategy, "train",
        batch_size=config.batch_size, num_workers=config.num_workers,
        input_size=INPUT_SIZE, data_fraction=config.data_fraction, seed=config.seed,
    )
    test_loader = get_dataloader(
        config.dataset, config.aug_strategy, "test",
        batch_size=config.batch_size, num_workers=config.num_workers,
        input_size=INPUT_SIZE,
    )
    if config.data_fraction < 1.0:
        print(f"  Data fraction: {config.data_fraction} (train size={len(train_loader.dataset)})")
    if config.use_scr:
        print(f"  SCR: ON  (lambda={config.lambda_cons}, type={config.cons_type})")

    # Model
    model = model_utils.build_model(
        config.model_name, config.num_classes, pretrained=config.pretrained,
    ).to(device)
    param_count = model_utils.count_parameters(model)
    print(f"  Parameters: {param_count:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    if config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    if config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    scaler = GradScaler(enabled=config.mixed_precision)

    # Training loop
    start_time = time.time()
    best_top1 = 0.0
    history = {"train_loss": [], "train_acc": [], "test_top1": [], "test_top5": [], "test_loss": []}

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config,
        )
        scheduler.step()

        # Evaluate every epoch
        test_metrics = evaluate(model, test_loader, criterion, device)
        is_best = test_metrics["top1"] > best_top1
        if is_best:
            best_top1 = test_metrics["top1"]
            # Save best checkpoint for downstream robustness eval
            if config.save_checkpoints:
                ckpt_dir = os.path.join(config.output_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                frac = config.data_fraction
                frac_tag = f"_f{int(frac*100)}" if frac < 1.0 else ""
                ckpt_path = os.path.join(
                    ckpt_dir, f"{config.dataset}_{config.model_name}_{config.aug_strategy}{frac_tag}.pth")
                torch.save({"model": model.state_dict(),
                            "config": vars(config)}, ckpt_path)

        # Log
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["test_top1"].append(round(test_metrics["top1"], 2))
        history["test_top5"].append(round(test_metrics["top5"], 2))
        history["test_loss"].append(round(test_metrics["loss"], 4))

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs or is_best:
            print(f"  >>> Epoch {epoch:3d}/{config.epochs} | "
                  f"Train Acc {train_acc:.2f} | Test Top1 {test_metrics['top1']:.2f} | "
                  f"Best {best_top1:.2f} | LR {scheduler.get_last_lr()[0]:.2e}")

    total_time = time.time() - start_time
    print(f"\n  Done! Best Top-1: {best_top1:.2f}% | Time: {total_time/60:.1f} min")

    return {
        "experiment": config.experiment,
        "run_name": config.run_name,
        "dataset": config.dataset,
        "model": config.model_name,
        "augmentation": config.aug_strategy,
        "use_scr": config.use_scr,
        "lambda_cons": config.lambda_cons,
        "cons_type": config.cons_type,
        "data_fraction": config.data_fraction,
        "epochs": config.epochs,
        "parameters": param_count,
        "best_top1": round(best_top1, 2),
        "final_top1": round(history["test_top1"][-1], 2),
        "final_top5": round(history["test_top5"][-1], 2),
        "train_time_min": round(total_time / 60, 1),
        "history": history,
    }


def save_results(result: dict, output_dir: str):
    """Save experiment result to JSON. Filename encodes aug + data fraction to avoid overwrites."""
    os.makedirs(output_dir, exist_ok=True)
    aug = result.get("augmentation", "std")
    frac = result.get("data_fraction", 1.0)
    frac_tag = f"_f{int(frac*100)}" if frac < 1.0 else ""
    method_tag = "_scr" if result.get("use_scr") else ""
    filename = (f"{result['experiment']}__{result['dataset']}__{result['model']}"
                f"__{aug}{method_tag}{frac_tag}.json")
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {filepath}")
    return filepath
