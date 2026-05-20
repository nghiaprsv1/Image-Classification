"""
tools/augment_offline.py
========================
Sinh ảnh augmentation từ ảnh gốc trong `dataset/raw/{class}/` để cân bằng
dataset đến đúng `--per-class N` ảnh.

Mỗi ảnh aug được tạo bằng tổ hợp ngẫu nhiên các phép biến đổi:
  - Rotate ±20°
  - Horizontal flip (50%)
  - Brightness ±20%
  - Contrast ±15%
  - Zoom 0.9 → 1.1
  - Color jitter nhẹ

Tên file output: `aug_<src_idx>_<aug_idx>.jpg`
  - `src_idx`: số thứ tự của ảnh gốc trong folder
  - `aug_idx`: thứ tự augmentation (1, 2, 3, ...)

Cách dùng:
    python tools/augment_offline.py --per-class 5500
    python tools/augment_offline.py --raw dataset/raw --per-class 7000 --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# UTF-8 output trên Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from tqdm import tqdm


def random_augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Áp dụng tổ hợp ngẫu nhiên các phép biến đổi lên `img`."""
    # 1. Horizontal flip
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)

    # 2. Rotate ±20° (giữ nguyên kích thước, fill background trắng)
    angle = rng.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.BILINEAR,
                     fillcolor=(255, 255, 255))

    # 3. Brightness 0.8 → 1.2
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 4. Contrast 0.85 → 1.15
    factor = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 5. Color saturation 0.85 → 1.15
    factor = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Color(img).enhance(factor)

    # 6. Zoom (crop center 90-100% rồi resize lại)
    if rng.random() < 0.5:
        w, h = img.size
        scale = rng.uniform(0.9, 1.0)
        cw, ch = int(w * scale), int(h * scale)
        left = (w - cw) // 2
        top = (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))
        img = img.resize((w, h), Image.BILINEAR)

    return img


def augment_class(class_dir: Path, target: int, seed: int = 42) -> int:
    """Đảm bảo `class_dir` có đúng `target` ảnh. Trả về số ảnh aug đã tạo."""
    if not class_dir.exists():
        print(f"[skip] {class_dir} không tồn tại")
        return 0

    # Lấy danh sách ảnh gốc (loại file aug_* nếu chạy lại)
    src_files = [f for f in class_dir.iterdir()
                 if f.is_file() and not f.stem.startswith("aug_")
                 and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    n_src = len(src_files)
    n_existing = sum(1 for f in class_dir.iterdir() if f.is_file())
    n_need = target - n_existing

    print(f"[{class_dir.name}] hiện có {n_existing} ({n_src} gốc), "
          f"target {target}, cần aug thêm {n_need}")

    if n_need <= 0:
        print(f"  [skip] đã đủ hoặc vượt target")
        return 0
    if n_src == 0:
        print(f"  [error] không có ảnh gốc để aug")
        return 0

    rng = random.Random(seed)
    aug_count = 0
    aug_per_src = (n_need + n_src - 1) // n_src  # ceil
    print(f"  -> {aug_per_src} aug/ảnh gốc, fill round-robin")

    pbar = tqdm(total=n_need, desc=f"aug {class_dir.name}")
    src_idx = 0
    # Đếm aug_idx hiện có cho mỗi src để tránh overwrite khi chạy lại
    aug_idx_per_src = {}
    for f in class_dir.iterdir():
        if f.is_file() and f.stem.startswith("aug_"):
            # tên: aug_<src_stem>_<aug_idx>
            try:
                parts = f.stem.rsplit("_", 1)
                src_stem = parts[0][len("aug_"):]
                a = int(parts[1])
                aug_idx_per_src[src_stem] = max(aug_idx_per_src.get(src_stem, 0), a)
            except (ValueError, IndexError):
                continue

    while aug_count < n_need:
        src = src_files[src_idx % n_src]
        a_idx = aug_idx_per_src.get(src.stem, 0) + 1
        aug_idx_per_src[src.stem] = a_idx
        target_path = class_dir / f"aug_{src.stem}_{a_idx:02d}.jpg"
        if target_path.exists():  # safety: skip nếu đã có
            src_idx += 1
            continue
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                aug = random_augment(im, rng)
                aug.save(target_path, format="JPEG", quality=92)
                aug_count += 1
                pbar.update(1)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"  [warn] {src.name}: {type(e).__name__}")
        src_idx += 1
        if src_idx > n_src * 20:  # safety guard
            print(f"  [warn] vượt giới hạn iteration, dừng")
            break

    pbar.close()
    return aug_count


def main() -> None:
    p = argparse.ArgumentParser(description="Offline augment để cân bằng dataset")
    p.add_argument("--raw", type=Path, default=Path("dataset/raw"),
                   help="Thư mục chứa các class folder (default: dataset/raw)")
    p.add_argument("--per-class", type=int, required=True,
                   help="Số ảnh muốn có cho mỗi class")
    p.add_argument("--classes", type=str, default="fresh,rotten",
                   help="Comma-separated class names")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    print(f"=== Offline Augmentation ===\nRaw: {args.raw}\n"
          f"Target: {args.per_class}/class\nClasses: {classes}\n")

    total_aug = 0
    for cls in classes:
        n = augment_class(args.raw / cls, args.per_class, args.seed)
        total_aug += n
        print()

    # Final summary
    print("=== Sau khi augment ===")
    for cls in classes:
        d = args.raw / cls
        if d.exists():
            n_total = sum(1 for f in d.iterdir() if f.is_file())
            n_aug = sum(1 for f in d.iterdir()
                        if f.is_file() and f.stem.startswith("aug_"))
            n_real = n_total - n_aug
            print(f"  {cls:<8s}: {n_total:>5,} ({n_real:,} thật + {n_aug:,} aug)")
    print(f"\n[done] tổng aug đã tạo: {total_aug:,}")


if __name__ == "__main__":
    main()
