"""
tools/topup_from_kaggle.py
==========================
Bổ sung dataset từ Freshness44 (Kaggle) vào `dataset/raw/{fresh,rotten}/`
cho đến khi mỗi class đạt đúng `--per-class N` ảnh.

Cấu trúc Freshness44 cache:
    D:/kaggle_cache/datasets/siavash93/freshness44/versions/1/Freshness44/
        Apple_Fresh/
        Apple_Rotten/
        Banana_Fresh/
        ...

Logic:
  1. Liệt kê toàn bộ ảnh trong từng folder *_Fresh và *_Rotten.
  2. Random shuffle (seed=42 reproducible).
  3. Copy vào dataset/raw/{fresh,rotten}/ với tên `kaggle_<idx>.jpg`.
  4. Skip nếu đã đủ target.
  5. (Optional) tính phash để tránh trùng với ảnh đã có trong dataset.

Cách dùng:
    python tools/topup_from_kaggle.py --per-class 6000
    python tools/topup_from_kaggle.py --per-class 6000 --no-dedup    # nhanh hơn
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


def list_kaggle_images(kaggle_root: Path) -> dict[str, List[Path]]:
    """Quét cache Freshness44, trả về dict {'fresh': [...], 'rotten': [...]}.

    Folder name pattern: <Fruit>_Fresh hoặc <Fruit>_Rotten.
    """
    result = {"fresh": [], "rotten": []}
    if not kaggle_root.exists():
        raise FileNotFoundError(f"Không tìm thấy {kaggle_root}")

    for sub in kaggle_root.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        if name.endswith("_fresh"):
            cls = "fresh"
        elif name.endswith("_rotten"):
            cls = "rotten"
        else:
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                result[cls].append(f)

    return result


def compute_existing_hashes(class_dir: Path) -> Set[imagehash.ImageHash]:
    """Tính phash của các ảnh đã có trong class_dir để tránh thêm trùng."""
    hashes: Set[imagehash.ImageHash] = set()
    files = [f for f in class_dir.iterdir() if f.is_file()]
    for f in tqdm(files, desc=f"hash existing {class_dir.name}"):
        try:
            with Image.open(f) as im:
                im = im.convert("RGB")
                hashes.add(imagehash.phash(im, hash_size=8))
        except (UnidentifiedImageError, OSError, SyntaxError):
            pass
    return hashes


def next_kaggle_idx(class_dir: Path) -> int:
    """Tìm số thứ tự kế tiếp cho file kaggle_*.jpg."""
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
    print(f"\n[{class_dir.name}] hiện có {n_existing:,}, target {target:,}, "
          f"cần thêm {n_need:,}")
    if n_need <= 0:
        print(f"  [skip] đã đủ")
        return 0

    # Tính hash của ảnh đã có nếu dedup
    existing_hashes: Set[imagehash.ImageHash] = set()
    if dedup:
        existing_hashes = compute_existing_hashes(class_dir)
        print(f"  [info] có {len(existing_hashes):,} hash trong class hiện tại")

    # Shuffle reproducible
    rng = random.Random(seed)
    pool = kaggle_imgs.copy()
    rng.shuffle(pool)

    start_idx = next_kaggle_idx(class_dir)
    copied = 0
    skipped_dup = 0
    skipped_bad = 0

    pbar = tqdm(total=n_need, desc=f"topup {class_dir.name}")
    for src in pool:
        if copied >= n_need:
            break

        # Verify ảnh + tính hash để check duplicate
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

        # Copy với tên kaggle_<idx>.<ext>
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

    print(f"  [done] copied {copied:,} | skip_dup={skipped_dup:,} | "
          f"skip_bad={skipped_bad:,}")
    return copied


def main() -> None:
    p = argparse.ArgumentParser(description="Topup dataset từ Freshness44 (Kaggle)")
    p.add_argument("--kaggle-root", type=Path, default=DEFAULT_KAGGLE_ROOT,
                   help="Đường dẫn tới folder Freshness44 (đã cache)")
    p.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    p.add_argument("--per-class", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-dedup", action="store_true",
                   help="Tắt phash dedup (nhanh hơn nhưng có thể trùng ảnh)")
    args = p.parse_args()

    print(f"=== Topup từ Kaggle Freshness44 ===")
    print(f"Source: {args.kaggle_root}")
    print(f"Target: {args.per_class:,}/class\n")

    print("Đang quét Kaggle cache...")
    kaggle_imgs = list_kaggle_images(args.kaggle_root)
    print(f"Tìm thấy: {len(kaggle_imgs['fresh']):,} fresh + "
          f"{len(kaggle_imgs['rotten']):,} rotten")

    total_copied = 0
    for cls in ("fresh", "rotten"):
        n = topup_class(
            args.raw / cls,
            kaggle_imgs[cls],
            args.per_class,
            seed=args.seed,
            dedup=not args.no_dedup,
        )
        total_copied += n

    print(f"\n=== Sau topup ===")
    for cls in ("fresh", "rotten"):
        d = args.raw / cls
        if d.exists():
            files = list(d.iterdir())
            n_kaggle = sum(1 for f in files if f.stem.startswith("kaggle_"))
            n_aug = sum(1 for f in files if f.stem.startswith("aug_"))
            n_crawl = len(files) - n_kaggle - n_aug
            print(f"  {cls:<7s}: {len(files):>5,}  "
                  f"(crawl={n_crawl:,}, aug={n_aug:,}, kaggle={n_kaggle:,})")
    print(f"\n[done] tổng đã copy: {total_copied:,}")


if __name__ == "__main__":
    main()
