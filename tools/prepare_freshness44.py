"""
tools/prepare_freshness44.py
============================
Tải dataset **Freshness44** (Kaggle: siavash93/freshness44) và sắp xếp lại
về cấu trúc `dataset/raw/{fresh,rotten}/` để khớp với pipeline của project.

Freshness44:
    - 53.616 ảnh, 22 loại rau/quả
    - Mỗi ảnh có 2 nhãn: `type` (apple, banana, ...) và `freshness` (fresh/rotten)
    - Đã clean: dedup MD5, chuẩn JPEG
    - Tổng size ~6.7 GB

Cách dùng:
    # Cách 1: dùng kagglehub (khuyến nghị — chỉ cần đăng nhập Kaggle 1 lần)
    pip install kagglehub
    python tools/prepare_freshness44.py

    # Cách 2: tự tải zip từ Kaggle UI rồi giải nén, sau đó:
    python tools/prepare_freshness44.py --src "C:/path/to/Freshness44"

    # Tuỳ chọn:
    python tools/prepare_freshness44.py --mode symlink     # tiết kiệm dung lượng
    python tools/prepare_freshness44.py --max-per-class 5000  # lấy mẫu để train nhanh
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm import tqdm

KAGGLE_HANDLE = "siavash93/freshness44"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
random.seed(SEED)


# ----------------------------------------------------------------------------
# 1. Tải qua kagglehub (không cần API token)
# ----------------------------------------------------------------------------
def download_via_kagglehub() -> Path:
    """Tải Freshness44 về cache của kagglehub. Trả về thư mục chứa data."""
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "[error] Thiếu kagglehub. Chạy:  pip install kagglehub\n"
            "        Hoặc dùng --src nếu đã tự tải dataset về."
        )

    print(f"[1/3] Tải {KAGGLE_HANDLE} qua kagglehub …")
    print("       (lần đầu sẽ mở browser để đăng nhập Kaggle)")
    path = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    print(f"       → {path}")
    return path


# ----------------------------------------------------------------------------
# 2. Phân loại ảnh theo "fresh" / "rotten" dựa vào tên thư mục cha
# ----------------------------------------------------------------------------
def classify_image(path: Path) -> Optional[str]:
    """Trả về 'fresh' / 'rotten' / None dựa vào path."""
    parts_lower = [p.lower() for p in path.parts]
    has_fresh = any("fresh" in p for p in parts_lower)
    has_rotten = any(("rotten" in p) or ("stale" in p) or ("spoil" in p)
                     for p in parts_lower)
    if has_rotten:           # ưu tiên rotten (ảnh có thể nằm trong "rotten_apple")
        return "rotten"
    if has_fresh:
        return "fresh"
    return None


# ----------------------------------------------------------------------------
# 3. Sắp xếp về dataset/raw/{fresh,rotten}/
# ----------------------------------------------------------------------------
def organise(src: Path, dst: Path, mode: str = "copy",
             max_per_class: Optional[int] = None) -> dict:
    """
    Quét tất cả ảnh trong `src`, phân loại fresh/rotten, copy/symlink sang `dst`.

    mode: "copy" (an toàn, tốn dung lượng) | "symlink" (nhanh, tiết kiệm)
    """
    print(f"\n[2/3] Quét ảnh trong {src} …")
    all_files = [p for p in src.rglob("*") if p.suffix.lower() in VALID_EXTS]
    print(f"       Tìm thấy {len(all_files):,} ảnh")

    # Phân loại
    by_class: dict[str, list[Path]] = {"fresh": [], "rotten": []}
    skipped = 0
    for f in all_files:
        cls = classify_image(f)
        if cls is None:
            skipped += 1
            continue
        by_class[cls].append(f)

    print(f"       fresh = {len(by_class['fresh']):,}, "
          f"rotten = {len(by_class['rotten']):,}, "
          f"skipped (không xác định) = {skipped:,}")

    # Sample nếu cần (để train nhanh khi máy yếu)
    if max_per_class:
        for cls in by_class:
            if len(by_class[cls]) > max_per_class:
                by_class[cls] = random.sample(by_class[cls], max_per_class)
                print(f"       → sample {cls}: {max_per_class:,} ảnh")

    # Copy / symlink
    print(f"\n[3/3] Sắp xếp về {dst} (mode={mode}) …")
    summary = {"fresh": 0, "rotten": 0}
    type_counter: Counter = Counter()

    for cls, files in by_class.items():
        out_dir = dst / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in tqdm(files, desc=f"  {cls}"):
            # Tên file mới: <type>_<index>.<ext> để giữ context fruit type
            fruit_type = guess_type(f)
            type_counter[fruit_type] += 1
            new_name = f"{fruit_type}_{type_counter[fruit_type]:05d}{f.suffix.lower()}"
            target = out_dir / new_name
            if target.exists():
                continue
            try:
                if mode == "symlink":
                    os.symlink(f.resolve(), target)
                else:
                    shutil.copy2(f, target)
                summary[cls] += 1
            except OSError as e:
                # Windows: symlink cần admin → fallback sang copy
                if mode == "symlink":
                    shutil.copy2(f, target)
                    summary[cls] += 1
                else:
                    print(f"  [warn] {f.name}: {e}")

    return summary


def guess_type(path: Path) -> str:
    """Đoán loại quả từ tên thư mục cha (apple, banana, ...). Mặc định 'unknown'."""
    for part in reversed(path.parent.parts):
        p = part.lower()
        if "fresh" in p or "rotten" in p or "stale" in p:
            continue
        # Lấy chữ cái + bỏ ký tự đặc biệt
        clean = "".join(c for c in p if c.isalnum())
        if clean and not clean.isdigit():
            return clean
    return "unknown"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Chuẩn bị dataset Freshness44")
    p.add_argument("--src", type=Path, default=None,
                   help="Đường dẫn Freshness44 đã tải sẵn. Bỏ trống → dùng kagglehub.")
    p.add_argument("--dst", type=Path, default=Path("dataset/raw"),
                   help="Thư mục đích (mặc định: dataset/raw)")
    p.add_argument("--mode", choices=["copy", "symlink"], default="copy",
                   help="copy (an toàn) | symlink (nhanh, ít dung lượng)")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="Giới hạn số ảnh/class (vd 5000) để train nhanh")
    args = p.parse_args()

    src = args.src or download_via_kagglehub()
    if not src.exists():
        sys.exit(f"[error] Không tìm thấy {src}")

    summary = organise(src, args.dst, mode=args.mode, max_per_class=args.max_per_class)

    print("\n=== Tổng kết ===")
    for cls, n in summary.items():
        print(f"  {cls:<8s}: {n:,} ảnh → {args.dst / cls}")

    print("\n[done] Bước tiếp theo:")
    print("       python preprocessing/preprocess.py --src dataset/raw --dst dataset")


if __name__ == "__main__":
    main()
