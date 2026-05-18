"""
preprocessing/augmentation.py
=============================
Pipeline augmentation cho ảnh dùng `tf.keras.preprocessing.image.ImageDataGenerator`.

Cung cấp:
    * `build_train_generator(...)`  — sinh batch ảnh train có augmentation.
    * `build_eval_generator(...)`   — chỉ rescale, dành cho valid/test.
    * `visualize_augmentation(...)` — vẽ ảnh sau augment để kiểm tra.

Augmentation áp dụng:
    rotation, horizontal flip, zoom, brightness, width/height shift, shear.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE: int = 224
BATCH_SIZE: int = 32
SEED: int = 42


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------
def build_train_generator(train_dir: Path,
                          img_size: int = IMG_SIZE,
                          batch_size: int = BATCH_SIZE,
                          class_mode: str = "categorical"):
    """Generator cho tập train với augmentation đầy đủ."""
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.20,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode="nearest",
    )
    return datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True,
        seed=SEED,
    )


def build_eval_generator(eval_dir: Path,
                         img_size: int = IMG_SIZE,
                         batch_size: int = BATCH_SIZE,
                         class_mode: str = "categorical",
                         shuffle: bool = False):
    """Generator cho valid / test — chỉ rescale, không augment, không shuffle."""
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        directory=str(eval_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle,
        seed=SEED,
    )


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def visualize_augmentation(train_dir: Path, out_path: Path,
                           n_samples: int = 8,
                           img_size: int = IMG_SIZE) -> None:
    """Lưu 1 ảnh thật + 8 phiên bản augment để kiểm tra trực quan."""
    gen = build_train_generator(train_dir, img_size=img_size, batch_size=1)
    # lấy 1 ảnh gốc (đã rescale [0,1])
    x_batch, _ = next(gen)
    base_img = x_batch[0]

    fig, axes = plt.subplots(2, n_samples // 2 + 1,
                             figsize=((n_samples // 2 + 1) * 2.5, 5))
    axes = axes.ravel()
    axes[0].imshow(base_img)
    axes[0].set_title("Augmented #1")
    axes[0].axis("off")
    for i in range(1, n_samples + 1):
        x_aug, _ = next(gen)
        axes[i].imshow(x_aug[0])
        axes[i].set_title(f"Augmented #{i + 1}")
        axes[i].axis("off")
    # ẩn ô trống nếu có
    for j in range(n_samples + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Mẫu augmentation", fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[saved] {out_path}")


# ----------------------------------------------------------------------------
# CLI demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    p.add_argument("--out", type=Path, default=Path("results/augmentation_demo.png"))
    args = p.parse_args()
    visualize_augmentation(args.train_dir, args.out)
