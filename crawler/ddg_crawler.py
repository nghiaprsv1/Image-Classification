"""
crawler/ddg_crawler.py
=======================
Wrapper cho DuckDuckGo Image Search — tương thích interface tối thiểu của
icrawler (`crawl(keyword, max_num, ...)`), để có thể plug-in vào
`crawler/crawl_images.py` cùng với BingImageCrawler / BaiduImageCrawler.

Lý do tự viết:
- icrawler không hỗ trợ DuckDuckGo native.
- DDG không yêu cầu API key, không bị Google block.

Cách dùng:
    crawler = DuckDuckGoImageCrawler(storage={'root_dir': 'out/'})
    crawler.crawl(keyword='fresh apple', max_num=50,
                  min_size=(64, 64), file_idx_offset='auto')
"""
from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import requests
from PIL import Image, UnidentifiedImageError

try:
    from ddgs import DDGS
except ImportError as e:
    raise ImportError(
        "ddgs chưa được cài. Chạy: pip install ddgs"
    ) from e


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


class DuckDuckGoImageCrawler:
    """Crawler tối giản dùng DuckDuckGo Images.

    Chỉ implement `__init__(storage=...)` và `crawl(keyword, max_num, ...)`
    để tương thích với cách gọi của icrawler trong crawl_images.py.
    """

    def __init__(
        self,
        storage: Optional[dict] = None,
        feeder_threads: int = 1,        # noqa: ARG002 (tương thích icrawler)
        parser_threads: int = 1,        # noqa: ARG002
        downloader_threads: int = 8,
        log_level: int = 50,            # noqa: ARG002 (DDG ko log nhiều)
    ) -> None:
        self.storage = storage or {"root_dir": "."}
        self.root_dir = Path(self.storage["root_dir"])
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.downloader_threads = downloader_threads

    # ------------------------------------------------------------------
    def _next_idx(self) -> int:
        """Tìm index tiếp theo để tránh ghi đè file đã tồn tại."""
        existing = list(self.root_dir.glob("*.jpg")) + list(self.root_dir.glob("*.png"))
        if not existing:
            return 1
        nums = []
        for f in existing:
            try:
                nums.append(int(f.stem.split("_")[-1]))
            except (ValueError, IndexError):
                continue
        return max(nums, default=0) + 1

    # ------------------------------------------------------------------
    def _download_one(
        self,
        url: str,
        out_path: Path,
        min_size: Tuple[int, int],
        timeout: int = 8,
    ) -> bool:
        """Tải 1 ảnh, kiểm tra valid + min_size. Trả về True nếu lưu được."""
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout, stream=True)
            r.raise_for_status()
            data = r.content
            if len(data) < 1024:           # < 1KB chắc chắn không phải ảnh thật
                return False
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
                if min(w, h) < min(min_size):
                    return False
                # Đảm bảo lưu thành JPEG/PNG hợp lệ
                fmt = (im.format or "JPEG").upper()
                if fmt not in ("JPEG", "PNG", "WEBP"):
                    return False
            out_path.write_bytes(data)
            return True
        except (requests.RequestException, UnidentifiedImageError,
                OSError, ValueError, Exception):  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    def crawl(
        self,
        keyword: str,
        max_num: int = 50,
        file_idx_offset="auto",          # noqa: ARG002 (tương thích api)
        min_size: Tuple[int, int] = (64, 64),
        **_kwargs,
    ) -> int:
        """Tải tối đa `max_num` ảnh cho `keyword`. Trả về số ảnh đã lưu."""
        # Fetch URLs từ DDG (over-fetch 2x để bù ảnh fail/duplicate)
        urls = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.images(keyword, max_results=max_num * 2):
                    u = r.get("image") or r.get("url")
                    if u:
                        urls.append(u)
                    if len(urls) >= max_num * 2:
                        break
        except Exception:  # noqa: BLE001 — DDG có thể rate-limit, ko crash
            return 0

        # Download song song
        start_idx = self._next_idx()
        saved = 0
        with ThreadPoolExecutor(max_workers=self.downloader_threads) as pool:
            futures = {}
            for i, url in enumerate(urls):
                if saved >= max_num:
                    break
                ext = ".jpg"          # mặc định JPEG; PIL convert nếu cần
                target = self.root_dir / f"ddg_{start_idx + i:06d}{ext}"
                fut = pool.submit(self._download_one, url, target, min_size)
                futures[fut] = target

            for fut in as_completed(futures):
                target = futures[fut]
                if fut.result():
                    saved += 1
                    if saved >= max_num:
                        break
                else:
                    # Xoá file rỗng/lỗi nếu lỡ tạo
                    if target.exists() and target.stat().st_size == 0:
                        try:
                            target.unlink()
                        except OSError:
                            pass
        # Pause nhẹ để tránh rate-limit DDG
        time.sleep(0.5)
        return saved
