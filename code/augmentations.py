"""
Symmetry-aware data augmentation strategies.

Supports two input regimes:
  - small (<=64): native CIFAR size for CIFAR-ResNet
  - large (224): upsampled for ViT / Tiny ImageNet
"""

import random
import numpy as np
from PIL import ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
#  Custom transforms
# ---------------------------------------------------------------------------
class RandomRot90:
    """Randomly rotate by 0 / 90 / 180 / 270 degrees."""
    def __call__(self, x):
        return TF.rotate(x, angle=random.choice([0, 90, 180, 270]))


class Identity:
    def __call__(self, x):
        return x


class SymmetryAwareHFlip:
    """
    Mirror-flip horizontally with probability proportional to the image's
    mirror-symmetry score.

    Rationale (the paper's method contribution):
      A horizontal flip is label-preserving ONLY when the image is mirror-
      symmetric enough that the flipped version stays on the natural-image
      manifold. Highly asymmetric images become unnatural after flipping and
      act as label noise. So we estimate each image's mirror symmetry and
      flip with probability proportional to it.

    Symmetry score s in (0,1]:
      s = 1 / (1 + relative_asymmetry),  where
      relative_asymmetry = mean|x - flip(x)| / mean|x|
      (scale-invariant: a uniformly bright image is symmetric regardless of scale)

    Flip probability is normalized so the expected flip rate ~ `base_prob`,
    making it comparable to a fixed-p baseline.
    """
    def __init__(self, base_prob=0.5, alpha=1.0):
        self.base_prob = base_prob
        self.alpha = alpha  # strength of symmetry modulation

    def _symmetry_score(self, pil_img):
        arr = np.asarray(pil_img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        flipped = arr[:, ::-1, :]
        abs_diff = np.abs(arr - flipped).mean()
        scale = arr.mean() + 1e-6
        rel_asym = abs_diff / scale
        return 1.0 / (1.0 + rel_asym)  # (0,1]

    def __call__(self, pil_img):
        s = self._symmetry_score(pil_img)
        # modulate around base_prob: p = base_prob * (1 + alpha*(s - s_ref))
        # s_ref ~ 0.8 typical for natural images -> p ~ base_prob when s~s_ref
        s_ref = 0.8
        p = self.base_prob * (1.0 + self.alpha * (s - s_ref))
        p = min(1.0, max(0.0, p))
        if random.random() < p:
            return ImageOps.mirror(pil_img)
        return pil_img


# ---------------------------------------------------------------------------
#  Dataset statistics
# ---------------------------------------------------------------------------
DN = {  # (mean, std)
    "cifar10":       ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100":      ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tinyimagenet":  ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
}

# ---------------------------------------------------------------------------
#  Resize helper
# ---------------------------------------------------------------------------
def _resize_for(size: int):
    """
    Return a resize transform appropriate for the target size.
      - small (≤64): exact resize (CIFAR native)
      - large (>64) : ImageNet-style resize → center crop
    """
    if size <= 64:
        return T.Resize(size)
    return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size)])


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def get_eval_transform(dataset: str, input_size: int = 224):
    """Evaluation transform: resize → to-tensor → normalize."""
    mean, std = DN[dataset]
    return T.Compose([
        _resize_for(input_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def get_train_transform(dataset: str, strategy: str, input_size: int = 224):
    """Training transform for a given augmentation strategy."""
    mean, std = DN[dataset]
    norm = T.Normalize(mean, std)

    # -- augmentation-free baseline -------------------------------------------
    if strategy == "none":
        return T.Compose([
            _resize_for(input_size),
            T.ToTensor(),
            norm,
        ])

    # -- standard torchvision recipe (baseline) -------------------------------
    if strategy == "standard":
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            norm,
        ])

    # -- individual symmetry operations ---------------------------------------
    if strategy == "hflip":
        return T.Compose([
            _resize_for(input_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            norm,
        ])

    # -- METHOD: symmetry-aware adaptive horizontal flip ----------------------
    if strategy == "adaptive_hflip":
        return T.Compose([
            _resize_for(input_size),
            SymmetryAwareHFlip(base_prob=0.5, alpha=1.0),
            T.ToTensor(),
            norm,
        ])

    if strategy == "vflip":
        return T.Compose([
            _resize_for(input_size),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            norm,
        ])

    if strategy == "rot90":
        return T.Compose([
            _resize_for(input_size),
            RandomRot90(),
            T.ToTensor(),
            norm,
        ])

    if strategy == "rot":
        return T.Compose([
            _resize_for(input_size),
            T.RandomRotation(degrees=30),
            T.ToTensor(),
            norm,
        ])

    if strategy == "rot60":
        return T.Compose([
            _resize_for(input_size),
            T.RandomRotation(degrees=60),
            T.ToTensor(),
            norm,
        ])

    if strategy == "hflip_vflip":
        return T.Compose([
            _resize_for(input_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            norm,
        ])

    # -- composite symmetry ---------------------------------------------------
    if strategy == "symmetry":
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            RandomRot90(),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            norm,
        ])

    # -- full: symmetry + photometric + crop ----------------------------------
    if strategy == "full":
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            RandomRot90(),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            norm,
        ])

    # -- dynamic rotation angle: rot15, rot45, rot90 (ablation) ---------------
    if strategy.startswith("rot") and strategy[3:].isdigit():
        angle = int(strategy[3:])
        return T.Compose([
            _resize_for(input_size),
            T.RandomRotation(degrees=angle),
            T.ToTensor(),
            norm,
        ])

    raise ValueError(f"Unknown augmentation strategy: {strategy}")
