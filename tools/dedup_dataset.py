"""
tools/dedup_dataset.py
======================
Quét toàn bộ `dataset/raw/{fresh,rotten}/`, tìm và xoá ảnh trùng (perceptual hash).

Tính năng:
  - phash 16x16 (chính xác hơn 8x8 mặc định)
  - Hamming distance threshold (default = 4) → ảnh "gần giống" cũng bị coi
    là trùng (giống nhau ~98%).
  - Ưu tiên giữ ảnh **thật** (không có prefix `aug_`) khi 2 ảnh trùng.
  - Cross-class check: cảnh báo nếu cùng ảnh xuất hiện ở cả 2 class.
  - Dry-run mode để preview trước khi xoá.

Cách dùng:
    python tools/dedup_dataset.py --dry-run               # preview
    python tools/dedup_dataset.py                          # thực sự xoá
    python tools/dedup_dataset.py --threshold 6           # nới hơn
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

import imagehash
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


HASH_SIZE = 16   # 16x16 = 256 bit hash, đủ chính xác cho ảnh ~1500px


def compute_hashes(folder: Path) -> Dict[Path, imagehash.ImageHash]:
    """Tính phash cho mọi ảnh trong folder. Bỏ qua ảnh hỏng."""
    hashes: Dict[Path, imagehash.ImageHash] = {}
    files = [f for f in folder.iterdir() if f.is_file()]
    for f in tqdm(files, desc=f"hash {folder.name}"):
        try:
            with Image.open(f) as im:
                im = im.convert("RGB")
                hashes[f] = imagehash.phash(im, hash_size=HASH_SIZE)
        except (UnidentifiedImageError, OSError, SyntaxError):
            # Ảnh hỏng — sẽ được xoá riêng
            pass
    return hashes


def find_duplicates(
    hashes: Dict[Path, imagehash.ImageHash],
    threshold: int,
) -> List[Tuple[Path, Path, int]]:
    """Tìm các cặp ảnh có Hamming distance ≤ threshold.

    Trả về list (path_a, path_b, distance).
    """
    paths = list(hashes.keys())
    n = len(paths)
    pairs: List[Tuple[Path, Path, int]] = []

    # Bucket theo prefix 16-bit của hash để giảm O(n²) → O(n*k)
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(paths):
        # Lấy 16 bit đầu của hash làm bucket key
        prefix = int(str(hashes[p])[:4], 16)
        buckets[prefix].append(i)

    seen_pairs = set()
    pbar = tqdm(total=n, desc=f"compare ({len(buckets)} buckets)")
    for indices in buckets.values():
        for i in indices:
            pbar.update(1)
            for j in indices:
                if i >= j:
                    continue
                key = (i, j)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                d = hashes[paths[i]] - hashes[paths[j]]
                if d <= threshold:
                    pairs.append((paths[i], paths[j], d))
    pbar.close()
    return pairs


def pick_loser(a: Path, b: Path) -> Path:
    """Trong cặp duplicate, chọn cái nào XOÁ.

    Ưu tiên giữ ảnh thật (không phải aug_*). Nếu cả 2 đều aug hoặc cả 2 đều
    thật → giữ cái có size lớn hơn.
    """
    a_aug = a.stem.startswith("aug_")
    b_aug = b.stem.startswith("aug_")
    if a_aug and not b_aug:
        return a       # xoá a (aug)
    if b_aug and not a_aug:
        return b       # xoá b (aug)
    # cả 2 cùng loại — giữ file lớn hơn
    sa = a.stat().st_size
    sb = b.stat().st_size
    return a if sa < sb else b


def dedup_within(folder: Path, threshold: int, dry_run: bool) -> int:
    """Dedup trong 1 folder. Trả về số ảnh đã xoá."""
    print(f"\n=== {folder.name} ===")
    if not folder.exists():
        print(f"  [skip] không tồn tại")
        return 0

    hashes = compute_hashes(folder)
    print(f"  Tính phash: {len(hashes)} ảnh hợp lệ")

    pairs = find_duplicates(hashes, threshold)
    print(f"  Tìm thấy: {len(pairs)} cặp duplicate (threshold={threshold})")

    # Build set ảnh sẽ xoá (1 ảnh có thể trùng nhiều ảnh khác)
    losers: set[Path] = set()
    for a, b, _ in pairs:
        # Nếu 1 trong 2 đã trong losers thì bỏ qua
        if a in losers or b in losers:
            continue
        losers.add(pick_loser(a, b))

    print(f"  Sẽ xoá: {len(losers)} ảnh (giữ ảnh thật khi có thể)")

    if dry_run:
        print(f"  [DRY-RUN] không xoá. Sample 5 ảnh:")
        for p in list(losers)[:5]:
            print(f"    - {p.name}")
        return len(losers)

    deleted = 0
    for p in losers:
        try:
            p.unlink()
            deleted += 1
        except OSError as e:
            print(f"  [warn] không xoá được {p.name}: {e}")
    print(f"  [done] đã xoá {deleted} ảnh")
    return deleted


def dedup_cross_class(
    fresh_dir: Path, rotten_dir: Path, threshold: int, dry_run: bool
) -> int:
    """Tìm ảnh xuất hiện ở cả fresh và rotten — xoá khỏi CẢ 2 class.

    Lý do xoá khỏi cả 2: nếu 1 ảnh được gắn vừa fresh vừa rotten thì label
    bị nhiễu (ambiguous), không nên dùng để train binary classifier.
    """
    print(f"\n=== Cross-class check ===")
    if not (fresh_dir.exists() and rotten_dir.exists()):
        return 0

    h_fresh = compute_hashes(fresh_dir)
    h_rotten = compute_hashes(rotten_dir)

    # So sánh từng cặp (fresh, rotten) qua bucket prefix
    buckets_fresh: Dict[int, List[Path]] = defaultdict(list)
    for p, h in h_fresh.items():
        buckets_fresh[int(str(h)[:4], 16)].append(p)

    cross_pairs: List[Tuple[Path, Path, int]] = []
    for p_r, h_r in h_rotten.items():
        prefix = int(str(h_r)[:4], 16)
        for p_f in buckets_fresh.get(prefix, []):
            d = h_r - h_fresh[p_f]
            if d <= threshold:
                cross_pairs.append((p_f, p_r, d))

    print(f"  Tìm thấy {len(cross_pairs)} cặp ảnh xuất hiện ở CẢ 2 class")
    if not cross_pairs:
        return 0

    # Xoá khỏi CẢ 2 class (label ambiguous)
    losers_fresh = {p_f for p_f, _, _ in cross_pairs}
    losers_rotten = {p_r for _, p_r, _ in cross_pairs}
    total_losers = len(losers_fresh) + len(losers_rotten)
    print(f"  Sẽ xoá {len(losers_fresh)} ảnh phía fresh + "
          f"{len(losers_rotten)} ảnh phía rotten = {total_losers} ảnh")
    print(f"  (xoá CẢ 2 vì label ambiguous, ảnh không phù hợp train)")

    if dry_run:
        for p_f, p_r, d in cross_pairs[:5]:
            print(f"    fresh: {p_f.name}  <->  rotten: {p_r.name}  (d={d})")
        return total_losers

    deleted = 0
    for p in losers_fresh | losers_rotten:
        try:
            p.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted


def main() -> None:
    p = argparse.ArgumentParser(description="Dedup dataset bằng perceptual hash")
    p.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    p.add_argument("--threshold", type=int, default=4,
                   help="Hamming distance tối đa để coi là duplicate (default=4)")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ preview, không xoá file")
    args = p.parse_args()

    print(f"=== Dedup dataset (threshold={args.threshold}, "
          f"dry_run={args.dry_run}) ===")

    total_deleted = 0
    for cls in ("fresh", "rotten"):
        total_deleted += dedup_within(args.raw / cls, args.threshold, args.dry_run)

    total_deleted += dedup_cross_class(
        args.raw / "fresh", args.raw / "rotten",
        args.threshold, args.dry_run,
    )

    print(f"\n=== TỔNG: {total_deleted} ảnh "
          f"{'sẽ bị' if args.dry_run else 'đã'} xoá ===")
    # Final count
    for cls in ("fresh", "rotten"):
        d = args.raw / cls
        if d.exists():
            n = sum(1 for f in d.iterdir() if f.is_file())
            print(f"  {cls:<8s}: {n:,} ảnh")


if __name__ == "__main__":
    main()
