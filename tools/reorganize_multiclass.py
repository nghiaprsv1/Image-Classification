"""
tools/reorganize_multiclass.py
==============================
Chuyển dataset từ cấu trúc binary (fresh/rotten) sang multi-class
(Apple_Fresh, Apple_Rotten, Banana_Fresh, ..., Lime_Rotten).

Chiến lược: dùng filename (keyword prefix) + folder hiện tại để map fruit + state.
Ảnh không thuộc 8 loại quả mục tiêu → chuyển vào dataset/raw_unused/.

Mục tiêu: 8 loại × 2 = 16 class folders.

Cách dùng:
    python tools/reorganize_multiclass.py                # MOVE thật
    python tools/reorganize_multiclass.py --dry-run      # preview
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


# 8 loại quả mục tiêu
TARGET_FRUITS = ['Apple', 'Pomegranate', 'Banana', 'Orange',
                 'Bellpepper', 'Tomato', 'Guava', 'Lime']

# Keyword pattern -> fruit name (case-insensitive substring match)
FRUIT_KEYWORDS = {
    'Apple':       ['apple', 'tao_'],
    'Pomegranate': ['pomegranate', 'luu'],
    'Banana':      ['banana', 'chuoi'],
    'Orange':      ['orange', 'cam_'],
    'Bellpepper':  ['bell_pepper', 'bellpepper', 'bell pepper', 'pepper', 'ot_chuong'],
    'Tomato':      ['tomato', 'ca_chua'],
    'Guava':       ['guava', 'oi_'],
    'Lime':        ['lime_', 'lemon', 'chanh'],
}


def match_fruit(stem: str) -> str | None:
    """Map filename stem -> fruit name. None nếu không khớp."""
    s = stem.lower()
    for fruit, keywords in FRUIT_KEYWORDS.items():
        if any(kw in s for kw in keywords):
            return fruit
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Re-organize binary -> multi-class")
    p.add_argument("--src", type=Path, default=Path("dataset/raw"),
                   help="Folder hiện tại (binary fresh/rotten)")
    p.add_argument("--dst", type=Path, default=Path("dataset/raw_multi"),
                   help="Folder đích (16 class folders)")
    p.add_argument("--unused", type=Path, default=Path("dataset/raw_unused"),
                   help="Folder cho ảnh không thuộc 8 loại quả")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"=== Re-organize: {args.src} -> {args.dst} ===")
    if args.dry_run:
        print("[DRY-RUN] không di chuyển file thật\n")

    # Tạo destination folders
    if not args.dry_run:
        for fruit in TARGET_FRUITS:
            for state in ('Fresh', 'Rotten'):
                (args.dst / f"{fruit}_{state}").mkdir(parents=True, exist_ok=True)
        args.unused.mkdir(parents=True, exist_ok=True)

    # Quét và phân loại
    counts = defaultdict(int)
    unused_count = 0

    for folder_state in ('fresh', 'rotten'):
        folder = args.src / folder_state
        if not folder.exists():
            print(f"[skip] {folder} không tồn tại")
            continue

        for f in folder.iterdir():
            if not f.is_file():
                continue

            # Bỏ prefix aug_ trước khi match
            stem = f.stem
            clean_stem = stem[4:] if stem.startswith('aug_') else stem

            fruit = match_fruit(clean_stem)
            if fruit is None:
                # Không thuộc 8 loại -> chuyển sang unused
                if not args.dry_run:
                    target = args.unused / f"{folder_state}__{f.name}"
                    if not target.exists():
                        shutil.move(str(f), str(target))
                unused_count += 1
                continue

            # Trạng thái lấy từ folder gốc (đáng tin hơn)
            state = folder_state.capitalize()  # Fresh / Rotten
            class_name = f"{fruit}_{state}"
            counts[class_name] += 1

            if not args.dry_run:
                # Đảm bảo unique tên file
                target = args.dst / class_name / f.name
                if target.exists():
                    target = args.dst / class_name / f"dup_{f.name}"
                shutil.move(str(f), str(target))

    # Print summary
    print("\n=== Kết quả ===")
    print(f"{'Class':<22s} {'Số ảnh':>8s}")
    print("-" * 32)
    for fruit in TARGET_FRUITS:
        for state in ('Fresh', 'Rotten'):
            cls = f"{fruit}_{state}"
            print(f"{cls:<22s} {counts[cls]:>8,}")
    total = sum(counts.values())
    print("-" * 32)
    print(f"{'TỔNG (16 class)':<22s} {total:>8,}")
    print(f"{'Unused (chuyển ra)':<22s} {unused_count:>8,}")

    # Cần bổ sung để đạt 350/class
    print("\n=== Cần crawl bổ sung để đạt 350/class ===")
    TARGET = 350
    need_total = 0
    for fruit in TARGET_FRUITS:
        for state in ('Fresh', 'Rotten'):
            cls = f"{fruit}_{state}"
            need = max(0, TARGET - counts[cls])
            need_total += need
            if need > 0:
                print(f"  {cls:<22s}: hiện {counts[cls]:>4,}, cần thêm {need:>4,}")
    print(f"\nTổng cần crawl thêm: {need_total:,}")


if __name__ == "__main__":
    main()
