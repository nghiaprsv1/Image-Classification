"""Tests cho crawler/crawl_images.py — tập trung vào hàm clean & stats.

KHÔNG test phần crawl thật vì cần Internet — đánh dấu @pytest.mark.network.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from crawler.crawl_images import clean_directory, collect_stats, plot_distribution


def _make(path: Path, size=(96, 96), color=(0, 200, 0)):
    """Ảnh đơn sắc — pHash giống ảnh đơn sắc khác (đặc tính của pHash)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="JPEG")


def _make_unique(path: Path, seed: int, size: int = 96):
    """Ảnh có pattern ngẫu nhiên theo `seed` → pHash sẽ khác nhau."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="JPEG", quality=92)


class TestCleanDirectory:
    def test_removes_broken_small_duplicate(self, tmp_path: Path):
        d = tmp_path / "fresh"
        # 3 ảnh hợp lệ với pattern KHÁC NHAU (pHash khác)
        _make_unique(d / "a.jpg", seed=1)
        _make_unique(d / "b.jpg", seed=2)
        _make_unique(d / "c.jpg", seed=3)
        # 1 ảnh duplicate (copy nguyên a.jpg → cùng pHash)
        (d / "dup.jpg").write_bytes((d / "a.jpg").read_bytes())
        # 1 ảnh quá nhỏ (< 64)
        _make_unique(d / "small.jpg", seed=4, size=20)
        # 1 file rác
        (d / "bad.jpg").write_bytes(b"garbage" * 10)

        stats = clean_directory(d)

        assert stats["total"] == 6
        assert stats["broken"] == 1
        assert stats["small"] == 1
        assert stats["duplicate"] == 1
        assert stats["kept"] == 3
        # Đếm file còn lại trên đĩa
        remaining = list(d.glob("*.jpg"))
        assert len(remaining) == 3


class TestCollectStats:
    def test_returns_dataframe(self, tmp_path: Path):
        root = tmp_path / "raw"
        for i in range(2):
            _make(root / "fresh" / f"f{i}.jpg", color=(10 * i + 5, 200, 10))
            _make(root / "rotten" / f"r{i}.jpg", color=(150, 50, 10 * i + 5))
        df = collect_stats(root)
        assert isinstance(df, pd.DataFrame)
        assert set(df["class"].unique()) == {"fresh", "rotten"}
        assert (df["w"] > 0).all()
        assert (df["h"] > 0).all()


class TestPlotDistribution:
    def test_writes_png(self, tmp_path: Path):
        df = pd.DataFrame({
            "class": ["fresh"] * 5 + ["rotten"] * 4,
            "file":  [f"x{i}.jpg" for i in range(9)],
            "w": [96] * 9, "h": [96] * 9,
        })
        out = tmp_path / "dist.png"
        plot_distribution(df, out)
        assert out.exists() and out.stat().st_size > 0
