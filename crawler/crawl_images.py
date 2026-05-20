"""
crawler/crawl_images.py
=======================
Tự động thu thập ảnh rau (fresh / rotten) từ Google Images, Bing, Baidu.

Tính năng:
    * Dùng `icrawler` (không cần selenium) — nhanh, ổn định.
    * Crawl theo nhiều keyword cho mỗi class.
    * Loại ảnh trùng (perceptual hash), ảnh lỗi, ảnh quá nhỏ.
    * Thống kê: số ảnh / class, kích thước phổ biến, biểu đồ phân bố.

Cách dùng:
    python crawler/crawl_images.py --target 10000 --out dataset/raw
    python crawler/crawl_images.py --per-keyword 600 --engines google,bing
"""
from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# icrawler có 3 backend: Google, Bing, Baidu — dùng song song để tăng đa dạng
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

import imagehash

# ----------------------------------------------------------------------------
# Cấu hình keyword cho từng class
# ----------------------------------------------------------------------------
# Tăng đa dạng: nhiều loại quả + rau, dùng tiếng Anh + Việt + đồng nghĩa.
# File sẽ được lưu thành '<class>/<keyword_slug>_<idx>.jpg' để sau dễ thống kê.
KEYWORDS: Dict[str, List[str]] = {
    "fresh": [
        # Trái cây
        "fresh apple fruit",
        "fresh banana",
        "fresh orange fruit",
        "fresh tomato",
        "fresh strawberry",
        "fresh grape fruit",
        "fresh mango",
        "fresh pomegranate",
        "fresh guava fruit",
        "fresh papaya",
        # Rau củ
        "fresh vegetables on table",
        "fresh cabbage",
        "fresh carrot",
        "fresh lettuce",
        "fresh broccoli",
        "fresh cucumber",
        "fresh bell pepper",
        "fresh potato",
        "fresh onion",
        # Tiếng Việt → tăng tính đa dạng vùng miền
        "rau cu tuoi",
        "trai cay tuoi ngon",
    ],
    "rotten": [
        # Trái cây hỏng
        "rotten apple",
        "rotten banana",
        "rotten orange fruit",
        "rotten tomato",
        "rotten strawberry",
        "moldy grape",
        "rotten mango",
        "rotten guava",
        "rotten papaya",
        # Rau hỏng
        "rotten vegetables",
        "rotten cabbage",
        "rotten carrot",
        "rotten lettuce",
        "spoiled cucumber",
        "moldy bell pepper",
        "rotten potato",
        "rotten onion",
        "decayed vegetables",
        "moldy fruit",
        # Tiếng Việt
        "rau cu hu thoi",
        "trai cay hu mốc",
    ],
}

ENGINE_MAP = {
    "google": GoogleImageCrawler,
    "bing": BingImageCrawler,
    "baidu": BaiduImageCrawler,
}

MIN_SIZE = 64           # ảnh nhỏ hơn 64px sẽ bị loại
HASH_SIZE = 8           # phash 8x8


# ----------------------------------------------------------------------------
# Crawl
# ----------------------------------------------------------------------------
def _slug(keyword: str) -> str:
    """Convert keyword thành slug an toàn cho tên file.
    Ví dụ: 'fresh apple fruit' → 'fresh_apple_fruit'."""
    import re
    s = keyword.lower().strip()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)  # bỏ ký tự đặc biệt
    s = re.sub(r"\s+", "_", s)
    return s or "kw"


def crawl_one(keyword: str, class_dir: Path, max_num: int,
              engines: List[str]) -> None:
    """Crawl 1 keyword qua nhiều engine, lưu vào sub-folder tạm theo slug
    rồi rename về `<class>/<slug>_<idx>.jpg` để biết ảnh thuộc keyword nào."""
    class_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(keyword)
    tmp_dir = class_dir / f"_tmp_{slug}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    per_engine = max(1, max_num // len(engines))

    for engine_name in engines:
        Crawler = ENGINE_MAP.get(engine_name)
        if Crawler is None:
            print(f"[skip] Unknown engine: {engine_name}")
            continue
        try:
            crawler = Crawler(
                storage={"root_dir": str(tmp_dir)},
                feeder_threads=1,
                parser_threads=2,
                downloader_threads=4,
                log_level=40,
            )
            print(f"  └─ [{engine_name:6s}] '{keyword}' → {per_engine} ảnh")
            crawler.crawl(
                keyword=keyword,
                max_num=per_engine,
                file_idx_offset="auto",
                min_size=(MIN_SIZE, MIN_SIZE),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {engine_name} fail trên '{keyword}': {exc}")

    # Move tmp_dir/* → class_dir/<slug>_<idx>.<ext>, đánh số tránh trùng
    moved = 0
    existing = len(list(class_dir.glob(f"{slug}_*")))
    for i, f in enumerate(sorted(tmp_dir.iterdir()), start=existing + 1):
        if not f.is_file():
            continue
        ext = f.suffix.lower() or ".jpg"
        target = class_dir / f"{slug}_{i:05d}{ext}"
        try:
            f.rename(target)
            moved += 1
        except OSError:
            pass
    # Xoá tmp_dir
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except OSError:
        pass
    if moved:
        print(f"     → moved {moved} ảnh vào {class_dir.name}/{slug}_*")


def crawl_class(class_name: str, keywords: List[str], out_root: Path,
                target_per_class: int, engines: List[str]) -> Path:
    """Crawl toàn bộ keyword cho 1 class."""
    class_dir = out_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    per_keyword = max(50, target_per_class // len(keywords))
    print(f"\n[crawl] class={class_name} | target={target_per_class} | per-kw={per_keyword}")
    for kw in keywords:
        crawl_one(kw, class_dir, per_keyword, engines)
    return class_dir


# ----------------------------------------------------------------------------
# Lọc ảnh: lỗi, nhỏ, trùng
# ----------------------------------------------------------------------------
def clean_directory(folder: Path) -> Dict[str, int]:
    """Loại ảnh hỏng / nhỏ / trùng. Trả về thống kê."""
    stats = {"total": 0, "broken": 0, "small": 0, "duplicate": 0, "kept": 0}
    seen_hashes: set[str] = set()
    files = sorted(folder.glob("*"))
    for f in tqdm(files, desc=f"clean {folder.name}"):
        if not f.is_file():
            continue
        stats["total"] += 1
        try:
            with Image.open(f) as im:
                im.verify()                         # check ảnh không hỏng
            with Image.open(f) as im:
                im = im.convert("RGB")
                w, h = im.size
                if min(w, h) < MIN_SIZE:
                    f.unlink(missing_ok=True)
                    stats["small"] += 1
                    continue
                phash = str(imagehash.phash(im, hash_size=HASH_SIZE))
        except (UnidentifiedImageError, OSError, SyntaxError):
            f.unlink(missing_ok=True)
            stats["broken"] += 1
            continue

        if phash in seen_hashes:
            f.unlink(missing_ok=True)
            stats["duplicate"] += 1
            continue
        seen_hashes.add(phash)
        stats["kept"] += 1
    return stats


# ----------------------------------------------------------------------------
# Thống kê
# ----------------------------------------------------------------------------
def collect_stats(root: Path) -> pd.DataFrame:
    rows = []
    size_counter: Counter = Counter()
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in cls_dir.glob("*"):
            try:
                with Image.open(f) as im:
                    rows.append({"class": cls_dir.name, "file": f.name,
                                 "w": im.size[0], "h": im.size[1]})
                    size_counter[f"{im.size[0]}x{im.size[1]}"] += 1
            except Exception:
                pass
    df = pd.DataFrame(rows)
    print("\n=== Top 10 kích thước ảnh phổ biến ===")
    for sz, cnt in size_counter.most_common(10):
        print(f"  {sz:>12s} : {cnt}")
    return df


def plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Vẽ biểu đồ phân bố số lượng ảnh từng class."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = df["class"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    bars = plt.bar(counts.index, counts.values,
                   color=["#4CAF50", "#E53935"][: len(counts)])
    for b, v in zip(bars, counts.values):
        plt.text(b.get_x() + b.get_width() / 2, v, str(v),
                 ha="center", va="bottom", fontsize=10)
    plt.title("Phân bố số lượng ảnh theo class")
    plt.ylabel("Số ảnh")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[saved] {out_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Crawl & clean rau fresh/rotten")
    p.add_argument("--target", type=int, default=10000,
                   help="Tổng số ảnh muốn có (chia đều 2 class)")
    p.add_argument("--out", type=Path, default=Path("dataset/raw"))
    p.add_argument("--engines", type=str, default="google,bing,baidu",
                   help="Comma-separated: google,bing,baidu")
    p.add_argument("--skip-crawl", action="store_true",
                   help="Bỏ qua crawl, chỉ clean & thống kê")
    args = p.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    target_per_class = args.target // 2

    if not args.skip_crawl:
        for cls, kws in KEYWORDS.items():
            crawl_class(cls, kws, args.out, target_per_class, engines)

    print("\n=== Cleaning ===")
    summary = defaultdict(dict)
    for cls in KEYWORDS:
        summary[cls] = clean_directory(args.out / cls)
    print("\n=== Clean summary ===")
    print(pd.DataFrame(summary).T)

    print("\n=== Collecting stats ===")
    df = collect_stats(args.out)
    print(df.groupby("class").size())

    plot_distribution(df, Path("results/data_distribution.png"))
    df.to_csv("results/data_inventory.csv", index=False)
    print("[done] crawl + clean + stats")


if __name__ == "__main__":
    main()
