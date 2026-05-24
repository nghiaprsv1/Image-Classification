"""
tools/crawl_multiclass.py
=========================
Crawl bổ sung ảnh cho 16 class (8 fruit × 2 state) đến đúng `--per-class N`.

Mỗi class có một danh sách keyword cụ thể (Anh + Việt) để crawl đa dạng.
Dùng Bing + Baidu + DuckDuckGo (cùng engines như crawl binary cũ).

Cách dùng:
    python tools/crawl_multiclass.py --per-class 350
    python tools/crawl_multiclass.py --per-class 350 --classes Apple_Fresh,Tomato_Rotten
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from icrawler.builtin import BaiduImageCrawler, BingImageCrawler

# DDG wrapper nội bộ
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from crawler.ddg_crawler import DuckDuckGoImageCrawler
except ImportError:
    DuckDuckGoImageCrawler = None


# ============================================================================
# 16 class × keywords
# ============================================================================
KEYWORDS_PER_CLASS: Dict[str, List[str]] = {
    'Apple_Fresh': ['fresh apple fruit', 'red apple fruit', 'green apple fruit',
                    'apple fruit market', 'tao tuoi do', 'apple healthy whole'],
    'Apple_Rotten': ['rotten apple', 'decayed apple', 'moldy apple',
                     'apple fungus', 'apple bruised brown', 'tao hong moc'],
    'Banana_Fresh': ['fresh banana ripe', 'yellow banana fruit', 'banana bunch fresh',
                     'chuoi tuoi vang', 'banana healthy yellow'],
    'Banana_Rotten': ['rotten banana brown', 'overripe banana black', 'moldy banana',
                      'banana brown spots decay', 'chuoi hong den'],
    'Orange_Fresh': ['fresh orange fruit', 'orange citrus whole', 'navel orange fresh',
                     'cam tuoi vang', 'fresh orange peel'],
    'Orange_Rotten': ['rotten orange mold', 'orange citrus decayed',
                      'moldy orange fungus', 'cam hong moc xanh', 'spoiled orange'],
    'Pomegranate_Fresh': ['fresh pomegranate fruit', 'red pomegranate whole',
                          'pomegranate seeds healthy', 'qua luu tuoi do',
                          'ripe pomegranate market'],
    'Pomegranate_Rotten': ['rotten pomegranate', 'moldy pomegranate decayed',
                           'pomegranate fungus rotten', 'spoiled pomegranate fruit',
                           'qua luu hong', 'pomegranate brown decay'],
    'Tomato_Fresh': ['fresh red tomato', 'fresh tomato fruit',
                     'ripe tomato bunch', 'ca chua tuoi do', 'tomato vine fresh'],
    'Tomato_Rotten': ['rotten tomato spoiled', 'moldy tomato decayed',
                      'tomato fungus mold', 'ca chua hong nat',
                      'tomato squashed bad', 'spoiled tomato'],
    'Bellpepper_Fresh': ['fresh bell pepper', 'red bell pepper whole',
                         'green bell pepper fresh', 'ot chuong tuoi',
                         'capsicum fresh fruit'],
    'Bellpepper_Rotten': ['rotten bell pepper', 'moldy bell pepper decayed',
                          'bell pepper fungus', 'ot chuong hong moc',
                          'spoiled capsicum decay'],
    'Guava_Fresh': ['fresh guava fruit', 'green guava whole', 'ripe guava fruit',
                    'qua oi tuoi xanh', 'guava market fresh'],
    'Guava_Rotten': ['rotten guava decayed', 'moldy guava fungus',
                     'guava brown spoiled', 'qua oi hong nat',
                     'spoiled guava fruit'],
    'Lime_Fresh': ['fresh lime fruit', 'green lime citrus', 'lemon fresh whole',
                   'chanh tuoi xanh', 'fresh lime cut half'],
    'Lime_Rotten': ['rotten lime', 'moldy lime decayed', 'spoiled lime fruit',
                    'lime fungus dry', 'chanh hong moc', 'rotten lemon brown'],
}


ENGINE_MAP = {'bing': BingImageCrawler, 'baidu': BaiduImageCrawler}
if DuckDuckGoImageCrawler is not None:
    ENGINE_MAP['duckduckgo'] = DuckDuckGoImageCrawler
    ENGINE_MAP['ddg'] = DuckDuckGoImageCrawler


def slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def crawl_one_keyword(kw: str, class_dir: Path, max_num: int,
                      engines: List[str]) -> int:
    """Crawl 1 keyword qua nhiều engine. Trả về số ảnh đã thêm."""
    added = 0
    kw_slug = slug(kw)
    # Đếm hiện tại
    existing_count = sum(1 for f in class_dir.iterdir() if f.is_file())
    next_idx = existing_count + 1

    for engine_name in engines:
        engine_cls = ENGINE_MAP.get(engine_name)
        if engine_cls is None:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                crawler = engine_cls(storage={'root_dir': tmp},
                                     log_level=50, downloader_threads=4)
                crawler.crawl(keyword=kw, max_num=max_num,
                              min_size=(64, 64), file_idx_offset='auto')
            except Exception as e:
                print(f"    [warn] {engine_name} fail: {type(e).__name__}")
                continue
            # Move các file mới vào class_dir, đặt tên có pattern
            for f in Path(tmp).iterdir():
                if not f.is_file():
                    continue
                ext = f.suffix.lower() or '.jpg'
                target = class_dir / f"{kw_slug}_{next_idx:05d}{ext}"
                while target.exists():
                    next_idx += 1
                    target = class_dir / f"{kw_slug}_{next_idx:05d}{ext}"
                shutil.move(str(f), str(target))
                added += 1
                next_idx += 1
    return added


def crawl_class(class_name: str, target: int, raw_root: Path,
                engines: List[str]) -> None:
    """Crawl 1 class đến target ảnh. Early-stop khi đạt."""
    class_dir = raw_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    current = sum(1 for f in class_dir.iterdir() if f.is_file())
    keywords = KEYWORDS_PER_CLASS.get(class_name, [])

    print(f"\n[{class_name}] hiện {current}/{target}")
    if current >= target:
        print(f"  [skip] đã đủ")
        return
    if not keywords:
        print(f"  [warn] không có keyword cho class này")
        return

    per_keyword = max(40, (target - current) // len(keywords) + 5)
    print(f"  -> {len(keywords)} keyword × ~{per_keyword} ảnh/kw")

    for kw in keywords:
        cur = sum(1 for f in class_dir.iterdir() if f.is_file())
        if cur >= target:
            print(f"  [done-early] {class_name} đạt {cur}/{target}")
            break
        print(f"  -> crawling: '{kw}'")
        added = crawl_one_keyword(kw, class_dir, per_keyword, engines)
        new_total = sum(1 for f in class_dir.iterdir() if f.is_file())
        print(f"     added={added}, total now={new_total}")


def main() -> None:
    p = argparse.ArgumentParser(description="Crawl multi-class")
    p.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    p.add_argument("--per-class", type=int, default=350)
    p.add_argument("--engines", type=str, default="bing,baidu,duckduckgo")
    p.add_argument("--classes", type=str, default=None,
                   help="Comma-separated list of classes; default = all 16")
    args = p.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    classes = ([c.strip() for c in args.classes.split(",")]
               if args.classes else list(KEYWORDS_PER_CLASS.keys()))

    print(f"=== Crawl multi-class: target={args.per_class}/class ===")
    print(f"Classes: {len(classes)}")
    print(f"Engines: {engines}\n")

    for cls in classes:
        crawl_class(cls, args.per_class, args.raw, engines)

    # Final summary
    print("\n=== Final ===")
    total = 0
    for cls in classes:
        d = args.raw / cls
        n = sum(1 for f in d.iterdir() if f.is_file()) if d.exists() else 0
        total += n
        flag = "OK" if n >= args.per_class else "LACK"
        print(f"  {cls:<22s}: {n:>5,} [{flag}]")
    print(f"  {'TỔNG':<22s}: {total:,}")


if __name__ == "__main__":
    main()
