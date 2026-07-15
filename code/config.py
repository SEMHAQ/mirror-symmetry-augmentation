"""Configuration — fast experiments on native CIFAR (32x32) with ResNet."""

from dataclasses import dataclass

PATHS = {
    "data": "/mnt/e/Project/MDPI/data",
    "output": "/mnt/e/Project/MDPI/symmetry-paper-2",
}

AUG_STRATEGIES = {
    "none":     "no augmentation",
    "standard": "RandomResizedCrop + ColorJitter + RandomHorizontalFlip",
    "hflip":    "RandomHorizontalFlip only",
    "symmetry": "hflip + vflip + rot90 + rot15",
    "full":     "symmetry + color jitter + crop",
}

MODELS = ["resnet50", "resnet101"]
INPUT_SIZE = 32


@dataclass
class Config:
    experiment: str = "exp"
    dataset: str = "cifar100"
    model_name: str = "resnet50"        # matches trainer
    aug_strategy: str = "standard"       # matches trainer
    run_name: str = ""                   # matches trainer
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3          # matches trainer
    weight_decay: float = 5e-4
    num_workers: int = 4
    seed: int = 42
    mixed_precision: bool = True
    log_interval: int = 50
    output_dir: str = f"{PATHS['output']}/results"
    pretrained: bool = True
    num_classes: int = 10
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    label_smoothing: float = 0.1
    device: str = "cuda"
    momentum: float = 0.9

    # ---- Symmetry Consistency Regularization (SCR) ----
    use_scr: bool = False              # enable mirror-symmetry consistency loss
    lambda_cons: float = 0.5           # weight of consistency loss
    cons_type: str = "symmetric_kl"    # symmetric_kl | mse | js
    # ---- Data efficiency experiments ----
    data_fraction: float = 1.0         # fraction of training data (1.0 = full)
    # ---- Checkpointing ----
    save_checkpoints: bool = False     # save best checkpoint for robustness eval

    def __post_init__(self):
        if not self.run_name:
            tag = "SCR" if self.use_scr else "baseline"
            self.run_name = f"{self.experiment}_{self.dataset}_{self.model_name}_{tag}"
