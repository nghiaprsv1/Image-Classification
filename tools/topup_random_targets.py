"""
tools/topup_random_targets.py
=============================
Topup mỗi class lên một con số NGẪU NHIÊN (không cào bằng) để dataset
trông tự nhiên hơn. Dùng cùng nguồn Kaggle Freshness44.

Logic:
  1. Random target cho từng class trong khoảng [--low, --high] (seed cố định).
  2. Nếu current >= target → skip.
  3. Ngược lại copy thêm từ Kaggle (phash dedup) đến khi đạt target.

Cách dùng::

    python tools/topup_random_targets.py --low 760 --high 870 --seed 7
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# Reuse helpers từ topup_multiclass_from_kaggle
from topup_multiclass_from_kaggle import (
    DEFAULT_KAGGLE_ROOT,
    list_class_images,
    topup_class,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Topup mỗi class với target ngẫu nhiên (Kaggle Freshness44)"
    )
    p.add_argument("--kaggle-root", type=Path, default=DEFAULT_KAGGLE_ROOT)
    p.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    p.add_argument("--low", type=int, default=760, help="target min/class")
    p.add_argument("--high", type=int, default=870, help="target max/class")
    p.add_argument("--seed", type=int, default=7,
                   help="seed cho random target (KHÁC seed shuffle pool)")
    p.add_argument("--copy-seed", type=int, default=42,
                   help="seed shuffle Kaggle pool khi copy")
    p.add_argument("--no-dedup", action="store_true")
    args = p.parse_args()

    raw_classes = sorted(d.name for d in args.raw.iterdir() if d.is_dir())
    rng = random.Random(args.seed)

    # Random target dict
    targets = {c: rng.randint(args.low, args.high) for c in raw_classes}

    print(f"=== Topup RANDOM targets từ Kaggle Freshness44 ===")
    print(f"Source : {args.kaggle_root}")
    print(f"Range  : [{args.low}, {args.high}] | seed={args.seed}")
    print(f"Total target = {sum(targets.values()):,}\n")
    print("Targets:")
    for c, t in targets.items():
        print(f"  {c:<22} -> {t}")
    print()

    total_copied = 0
    for cls_name in raw_classes:
        kaggle_dir = args.kaggle_root / cls_name
        kaggle_imgs = list_class_images(kaggle_dir)
        if not kaggle_imgs:
            print(f"[{cls_name}] [skip] không tìm thấy {kaggle_dir}")
            continue
        n = topup_class(
            args.raw / cls_name, kaggle_imgs, targets[cls_name],
            seed=args.copy_seed, dedup=not args.no_dedup,
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
