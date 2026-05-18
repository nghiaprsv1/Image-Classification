"""
preprocessing/preprocess.py
===========================
Tiền xử lý dữ liệu:
    1. Quét thư mục `dataset/raw/{fresh,rotten}`.
    2. Resize ảnh về 224×224, convert RGB.
    3. Lưu sang `dataset/processed/`.
    4. Split train / valid / test (mặc định 70/15/15) → `dataset/{train,valid,test}/`.
    5. Trực quan hóa: sample images + class distribution.

Cách dùng:
    python preprocessing/preprocess.py --src dataset/raw --dst dataset
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------
# 1. Resize & save processed
# ----------------------------------------------------------------------------
def process_image(src: Path, dst: Path, img_size: int) -> bool:
    """Resize + convert RGB + lưu jpg. Trả về True nếu thành công."""
    try:
        with Image.open(src) as im:
            im = im.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=92)
        return True
    except (UnidentifiedImageError, OSError):
        return False


def preprocess_all(src_root: Path, processed_root: Path, img_size: int) -> dict:
    """Resize toàn bộ ảnh trong src_root sang processed_root."""
    summary = {}
    for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        cls = cls_dir.name
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
        ok = 0
        for f in tqdm(files, desc=f"resize {cls}"):
            new_name = f"{cls}_{f.stem}.jpg"
            dst = processed_root / cls / new_name
            if process_image(f, dst, img_size):
                ok += 1
        summary[cls] = ok
        print(f"  → {cls}: {ok}/{len(files)} ảnh OK")
    return summary


# ----------------------------------------------------------------------------
# 2. Split train / valid / test
# ----------------------------------------------------------------------------
def split_dataset(processed_root: Path, dst_root: Path,
                  ratios=(0.7, 0.15, 0.15)) -> dict:
    """Chia stratified theo class. Copy file sang dst_root/{train,valid,test}/<class>/"""
    assert abs(sum(ratios) - 1.0) < 1e-6, "Tỉ lệ phải cộng = 1.0"
    splits = ["train", "valid", "test"]

    # Xóa split cũ nếu có để tránh lẫn dữ liệu
    for s in splits:
        (dst_root / s).mkdir(parents=True, exist_ok=True)
        for sub in (dst_root / s).iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)

    summary = {}
    for cls_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        cls = cls_dir.name
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * ratios[0])
        n_valid = int(n * ratios[1])
        parts = {
            "train": files[:n_train],
            "valid": files[n_train:n_train + n_valid],
            "test":  files[n_train + n_valid:],
        }
        for s, group in parts.items():
            out = dst_root / s / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in group:
                shutil.copy2(f, out / f.name)
        summary[cls] = {s: len(g) for s, g in parts.items()}
        print(f"  {cls}: train={summary[cls]['train']} "
              f"valid={summary[cls]['valid']} test={summary[cls]['test']}")
    return summary


# ----------------------------------------------------------------------------
# 3. Visualize
# ----------------------------------------------------------------------------
def plot_samples(root: Path, out_path: Path, n_per_class: int = 4) -> None:
    """Hiển thị vài ảnh mẫu mỗi class."""
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    fig, axes = plt.subplots(len(classes), n_per_class,
                             figsize=(n_per_class * 2.5, len(classes) * 2.5))
    if len(classes) == 1:
        axes = np.expand_dims(axes, 0)
    for i, cls in enumerate(classes):
        files = list((root / cls).glob("*"))
        random.shuffle(files)
        for j in range(n_per_class):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(files):
                ax.imshow(Image.open(files[j]))
            if j == 0:
                ax.set_title(cls, fontsize=12, loc="left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[saved] {out_path}")


def plot_split_distribution(summary: dict, out_path: Path) -> None:
    """Bar chart số ảnh từng split."""
    classes = list(summary.keys())
    splits = ["train", "valid", "test"]
    x = np.arange(len(classes))
    width = 0.25
    plt.figure(figsize=(7, 4))
    for i, s in enumerate(splits):
        vals = [summary[c][s] for c in classes]
        plt.bar(x + i * width, vals, width, label=s)
    plt.xticks(x + width, classes)
    plt.ylabel("Số ảnh")
    plt.title("Phân bố train / valid / test")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[saved] {out_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("dataset/raw"))
    p.add_argument("--dst", type=Path, default=Path("dataset"))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--valid", type=float, default=0.15)
    p.add_argument("--test",  type=float, default=0.15)
    p.add_argument("--skip-resize", action="store_true")
    args = p.parse_args()

    processed_root = args.dst / "processed"

    if not args.skip_resize:
        print(f"[1/3] Resize → {args.img_size}×{args.img_size}")
        preprocess_all(args.src, processed_root, args.img_size)

    print("\n[2/3] Split dataset")
    summary = split_dataset(processed_root, args.dst,
                            ratios=(args.train, args.valid, args.test))

    print("\n[3/3] Visualize")
    plot_samples(processed_root, Path("results/sample_images.png"))
    plot_split_distribution(summary, Path("results/split_distribution.png"))
    print("[done] preprocess")


if __name__ == "__main__":
    main()
