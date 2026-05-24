"""
tools/topup_multiclass_from_kaggle.py
=====================================
Bổ sung dataset MULTI-CLASS (16 lớp) từ Freshness44 (Kaggle) vào
``dataset/raw/<Fruit>_<Status>/`` cho đến khi mỗi class đạt đúng
``--per-class N`` ảnh.

Khác với ``topup_from_kaggle.py`` (chỉ binary fresh/rotten), script này
giữ nguyên 16 thư mục class (Apple_Fresh, Apple_Rotten, ..., Tomato_Rotten)
và topup từng class một từ folder Kaggle có cùng tên.

Cấu trúc Freshness44 cache (đã xác nhận tồn tại):
    D:/kaggle_cache/datasets/siavash93/freshness44/versions/1/Freshness44/
        Apple_Fresh/    Apple_Rotten/    Banana_Fresh/    ...

Cách dùng::

    python tools/topup_multiclass_from_kaggle.py --per-class 750
    python tools/topup_multiclass_from_kaggle.py --per-class 750 --no-dedup
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Set

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

import imagehash
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


DEFAULT_KAGGLE_ROOT = Path(
    "D:/kaggle_cache/datasets/siavash93/freshness44/versions/1/Freshness44"
)

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def list_class_images(kaggle_class_dir: Path) -> List[Path]:
    """Liệt kê toàn bộ ảnh trong 1 folder class của Kaggle."""
    if not kaggle_class_dir.exists():
        return []
    return [
        f for f in kaggle_class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXT
    ]


def compute_existing_hashes(class_dir: Path) -> Set[imagehash.ImageHash]:
    """Tính phash của các ảnh đã có trong class_dir để tránh thêm trùng."""
    hashes: Set[imagehash.ImageHash] = set()
    files = [f for f in class_dir.iterdir() if f.is_file()]
    for f in tqdm(files, desc=f"hash {class_dir.name}", leave=False):
        try:
            with Image.open(f) as im:
                im = im.convert("RGB")
                hashes.add(imagehash.phash(im, hash_size=8))
        except (UnidentifiedImageError, OSError, SyntaxError):
            pass
    return hashes


def next_kaggle_idx(class_dir: Path) -> int:
    """Tìm số thứ tự kế tiếp cho file kaggle_*.jpg trong class_dir."""
    nums = []
    for f in class_dir.iterdir():
        if f.is_file() and f.stem.startswith("kaggle_"):
            try:
                nums.append(int(f.stem.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return max(nums, default=0) + 1


def topup_class(
    class_dir: Path,
    kaggle_imgs: List[Path],
    target: int,
    seed: int = 42,
    dedup: bool = True,
) -> int:
    """Copy ảnh từ Kaggle vào class_dir đến khi đạt target. Trả về số đã copy."""
    class_dir.mkdir(parents=True, exist_ok=True)
    n_existing = sum(1 for f in class_dir.iterdir() if f.is_file())
    n_need = target - n_existing
    print(f"[{class_dir.name}] hiện có {n_existing:,}, target {target:,}, "
          f"cần thêm {max(n_need,0):,}")
    if n_need <= 0:
        return 0
    if not kaggle_imgs:
        print(f"  [warn] không có ảnh Kaggle cho class {class_dir.name}")
        return 0

    existing_hashes: Set[imagehash.ImageHash] = set()
    if dedup:
        existing_hashes = compute_existing_hashes(class_dir)

    rng = random.Random(seed)
    pool = kaggle_imgs.copy()
    rng.shuffle(pool)

    start_idx = next_kaggle_idx(class_dir)
    copied = skipped_dup = skipped_bad = 0

    pbar = tqdm(total=n_need, desc=f"topup {class_dir.name}", leave=False)
    for src in pool:
        if copied >= n_need:
            break
        try:
            with Image.open(src) as im:
                im_rgb = im.convert("RGB")
                h = imagehash.phash(im_rgb, hash_size=8) if dedup else None
        except (UnidentifiedImageError, OSError, SyntaxError):
            skipped_bad += 1
            continue
        if dedup and h in existing_hashes:
            skipped_dup += 1
            continue
        ext = src.suffix.lower() or ".jpg"
        target_path = class_dir / f"kaggle_{start_idx + copied:05d}{ext}"
        try:
            shutil.copy2(src, target_path)
            copied += 1
            pbar.update(1)
            if dedup and h is not None:
                existing_hashes.add(h)
        except OSError as e:
            print(f"  [warn] copy fail {src.name}: {e}")
    pbar.close()
    print(f"  [done] copied={copied:,} | skip_dup={skipped_dup:,} | "
          f"skip_bad={skipped_bad:,}")
    return copied


def main() -> None:
    p = argparse.ArgumentParser(
        description="Topup MULTI-CLASS dataset từ Freshness44 (Kaggle)"
    )
    p.add_argument("--kaggle-root", type=Path, default=DEFAULT_KAGGLE_ROOT)
    p.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    p.add_argument("--per-class", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-dedup", action="store_true")
    args = p.parse_args()

    raw_classes = sorted(
        d.name for d in args.raw.iterdir() if d.is_dir()
    )
    print(f"=== Topup MULTI-CLASS từ Kaggle Freshness44 ===")
    print(f"Source : {args.kaggle_root}")
    print(f"Target : {args.per_class:,}/class | classes={len(raw_classes)}")
    print(f"Total target = {args.per_class * len(raw_classes):,}\n")

    total_copied = 0
    for cls_name in raw_classes:
        kaggle_dir = args.kaggle_root / cls_name
        kaggle_imgs = list_class_images(kaggle_dir)
        if not kaggle_imgs:
            print(f"[{cls_name}] [skip] không tìm thấy {kaggle_dir}")
            continue
        n = topup_class(
            args.raw / cls_name, kaggle_imgs, args.per_class,
            seed=args.seed, dedup=not args.no_dedup,
        )
        total_copied += n

    print(f"\n=== Sau topup ===")
    grand = 0
    for cls_name in raw_classes:
        d = args.raw / cls_name
        if d.exists():
            files = list(d.iterdir())
            n_kaggle = sum(1 for f in files if f.stem.startswith("kaggle_"))
            n_aug = sum(1 for f in files if f.stem.startswith("aug_"))
            n_crawl = len(files) - n_kaggle - n_aug
            grand += len(files)
            print(f"  {cls_name:<22}: {len(files):>5,}  "
                  f"(crawl={n_crawl:,}, aug={n_aug:,}, kaggle={n_kaggle:,})")
    print(f"\n[done] tổng đã copy: {total_copied:,} | tổng dataset: {grand:,}")


if __name__ == "__main__":
    main()
