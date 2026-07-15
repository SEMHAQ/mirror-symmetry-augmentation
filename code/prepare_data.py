#!/usr/bin/env python3
"""Prepare datasets: extract Tiny ImageNet and CIFAR-10-C."""

import os
import zipfile
import shutil
from pathlib import Path

DATA = "/mnt/e/Project/MDPI/data"


def extract_tiny_imagenet():
    """Extract Tiny ImageNet zip and organize validation set."""
    zip_path = f"{DATA}/tiny-imagenet-200.zip"
    out_dir = f"{DATA}/tiny-imagenet-200"

    if os.path.isdir(f"{out_dir}/train") and os.path.isdir(f"{out_dir}/val"):
        print("[Tiny ImageNet] Already extracted")
        return

    print("[Tiny ImageNet] Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA)

    # Organize validation set: need class subfolders
    val_dir = f"{out_dir}/val"
    annotations = f"{val_dir}/val_annotations.txt"

    if os.path.exists(annotations):
        print("[Tiny ImageNet] Organizing val set...")
        with open(annotations) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img, cls = parts[0], parts[1]
                    cls_dir = f"{val_dir}/{cls}"
                    os.makedirs(cls_dir, exist_ok=True)
                    src = f"{val_dir}/images/{img}"
                    dst = f"{cls_dir}/{img}"
                    if os.path.exists(src):
                        shutil.move(src, dst)

        # Remove old images dir
        images_dir = f"{val_dir}/images"
        if os.path.isdir(images_dir):
            shutil.rmtree(images_dir)

    print("[Tiny ImageNet] Ready!")


def extract_cifar10c():
    """Extract CIFAR-10-C from archive.zip."""
    zip_path = f"{DATA}/archive.zip"
    out_dir = f"{DATA}/CIFAR-10-C"

    if os.path.isdir(out_dir):
        npy_files = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
        if len(npy_files) >= 19:
            print("[CIFAR-10-C] Already extracted")
            return

    print("[CIFAR-10-C] Extracting (2.4GB zip)...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA)
    print("[CIFAR-10-C] Ready!")


if __name__ == "__main__":
    extract_tiny_imagenet()
    extract_cifar10c()
