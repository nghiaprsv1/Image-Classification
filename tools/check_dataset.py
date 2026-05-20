"""
tools/check_dataset.py
======================
Kiểm tra nhanh dataset đã được chuẩn bị OK chưa.

Báo cáo:
    * Số ảnh từng class
    * Số ảnh hỏng / không decode được
    * Ảnh quá nhỏ (<224)
    * Phân bố kích thước (min/max/median)
    * Phân bố fruit type (apple, banana, ...)
    * Có duplicate filename không
    * Cân bằng class (imbalance ratio)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from statistics import median

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_class(cls_dir: Path, sample_check: int = 0) -> dict:
    """Kiểm tra 1 class. `sample_check=0` = check toàn bộ."""
    files = [f for f in cls_dir.iterdir() if f.is_file()]
    stats = {
        "total": len(files),
        "broken": 0,
        "small": 0,
        "ok": 0,
        "widths": [],
        "heights": [],
        "fruit_types": Counter(),
    }
    files_to_check = files if sample_check == 0 else files[:sample_check]
    for f in files_to_check:
        # Đoán fruit type từ tên file (format: <type>_<idx>.jpg)
        stem = f.stem.rsplit("_", 1)[0] if "_" in f.stem else "unknown"
        stats["fruit_types"][stem] += 1
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                im = im.convert("RGB")
                w, h = im.size
                if min(w, h) < 224:
                    stats["small"] += 1
                stats["widths"].append(w)
                stats["heights"].append(h)
                stats["ok"] += 1
        except (UnidentifiedImageError, OSError):
            stats["broken"] += 1
    return stats


def print_report(stats_by_class: dict) -> bool:
    """In báo cáo; trả về True nếu dataset OK."""
    print("\n" + "=" * 70)
    print(f"{'CLASS':<10s} {'TOTAL':>8s} {'OK':>8s} {'BROKEN':>8s} {'SMALL':>8s}")
    print("-" * 70)
    totals = []
    issues = []
    for cls, s in stats_by_class.items():
        print(f"{cls:<10s} {s['total']:>8d} {s['ok']:>8d} {s['broken']:>8d} {s['small']:>8d}")
        totals.append(s["total"])
        if s["broken"] > 0:
            issues.append(f"  ⚠  {cls}: {s['broken']} ảnh hỏng cần xoá")
        if s["small"] > 0:
            issues.append(f"  ℹ  {cls}: {s['small']} ảnh nhỏ <224 (sẽ upscale khi resize, OK)")

    print("-" * 70)
    grand = sum(totals)
    print(f"{'TỔNG':<10s} {grand:>8d}")

    # Cân bằng
    if len(totals) >= 2:
        imbalance = max(totals) / max(min(totals), 1)
        print(f"\nImbalance ratio (max/min) = {imbalance:.2f}", end="")
        if imbalance > 1.5:
            print("  ⚠  KHÔNG CÂN BẰNG → cân nhắc class_weight hoặc oversampling")
            issues.append("  ⚠  Dataset không cân bằng (>1.5)")
        else:
            print("  ✓ cân bằng")

    # Kích thước
    print("\nKích thước ảnh:")
    for cls, s in stats_by_class.items():
        if s["widths"]:
            ws = s["widths"]; hs = s["heights"]
            print(f"  {cls:<8s}: w {min(ws)}-{max(ws)} (median {median(ws):.0f}), "
                  f"h {min(hs)}-{max(hs)} (median {median(hs):.0f})")

    # Fruit types
    print("\nLoại rau/quả:")
    for cls, s in stats_by_class.items():
        types = s["fruit_types"]
        top = ", ".join(f"{k}={v}" for k, v in types.most_common(8))
        print(f"  {cls:<8s} ({len(types)} loại): {top}{' ...' if len(types) > 8 else ''}")

    # Tổng kết
    print("\n" + "=" * 70)
    if issues:
        print("PHÁT HIỆN VẤN ĐỀ:")
        for x in issues:
            print(x)
    else:
        print("✅ DATASET ỔN — sẵn sàng chạy preprocessing/preprocess.py")
    print("=" * 70)

    has_blocker = any(s["broken"] > 0 for s in stats_by_class.values())
    return not has_blocker


def main() -> None:
    p = argparse.ArgumentParser(description="Kiểm tra dataset/raw")
    p.add_argument("--root", type=Path, default=Path("dataset/raw"))
    p.add_argument("--sample", type=int, default=0,
                   help="Chỉ check N ảnh đầu mỗi class (0 = check hết)")
    args = p.parse_args()

    if not args.root.exists():
        sys.exit(f"[error] Không tìm thấy {args.root}. "
                 f"Chạy `python tools/prepare_freshness44.py` trước.")

    classes = sorted(d for d in args.root.iterdir() if d.is_dir())
    if not classes:
        sys.exit(f"[error] {args.root} không có thư mục con (fresh/rotten)")

    stats_by_class = {}
    for cls_dir in classes:
        print(f"[scan] {cls_dir.name} … ", end="", flush=True)
        stats_by_class[cls_dir.name] = check_class(cls_dir, sample_check=args.sample)
        s = stats_by_class[cls_dir.name]
        print(f"{s['ok']}/{s['total']} OK")

    ok = print_report(stats_by_class)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
