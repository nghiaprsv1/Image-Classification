"""Pytest fixtures dùng chung.

Tạo dataset ảnh synthetic — không cần crawl thật, không cần Internet,
không cần ImageNet weights — pipeline test chạy được trên CI/CPU thuần.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Headless backend để matplotlib không mở cửa sổ khi savefig
os.environ.setdefault("MPLBACKEND", "Agg")

# Cho phép import package: từ root project (project/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------------
# Helper sinh ảnh giả
# ----------------------------------------------------------------------------
def _make_image(path: Path, size: int = 96, color: tuple[int, int, int] = (200, 50, 50),
                noise: bool = True) -> None:
    """Tạo 1 ảnh JPEG synthetic ở `path`."""
    rng = np.random.default_rng(abs(hash(str(path))) % (2**32))
    arr = np.full((size, size, 3), color, dtype=np.uint8)
    if noise:
        arr = arr.astype(np.int16) + rng.integers(-30, 30, arr.shape, dtype=np.int16)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="JPEG", quality=90)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture
def raw_dataset(tmp_path: Path) -> Path:
    """Tạo `dataset/raw/{fresh,rotten}/` với vài ảnh.

    Trả về đường dẫn raw root. Có chứa:
      * fresh/  : 6 ảnh hợp lệ
      * rotten/ : 6 ảnh hợp lệ
      * fresh/broken.jpg  : file rác (không phải ảnh)
      * fresh/tiny.jpg    : ảnh quá nhỏ (32×32)
    """
    raw = tmp_path / "dataset" / "raw"
    for i in range(6):
        _make_image(raw / "fresh" / f"f{i}.jpg", size=96, color=(80, 200, 80))
        _make_image(raw / "rotten" / f"r{i}.jpg", size=96, color=(120, 60, 30))
    # File rác — không phải ảnh
    (raw / "fresh" / "broken.jpg").write_bytes(b"this is not an image")
    # Ảnh quá nhỏ — vẫn là ảnh hợp lệ nhưng < MIN_SIZE 64
    _make_image(raw / "fresh" / "tiny.jpg", size=32, color=(80, 200, 80))
    return raw


@pytest.fixture
def split_dataset_dir(tmp_path: Path) -> Path:
    """Tạo `dataset/{train,valid,test}/{fresh,rotten}/` với ảnh 224×224 sẵn.

    Đủ ảnh cho ImageDataGenerator hoạt động (batch=2).
    Trả về thư mục root chứa train/ valid/ test/.
    """
    root = tmp_path / "dataset"
    counts = {"train": 8, "valid": 4, "test": 4}
    palette = {"fresh": (80, 200, 80), "rotten": (120, 60, 30)}
    for split, n in counts.items():
        for cls, color in palette.items():
            for i in range(n):
                _make_image(root / split / cls / f"{cls}_{i}.jpg",
                            size=224, color=color)
    return root


@pytest.fixture
def tiny_image(tmp_path: Path) -> Path:
    """1 ảnh 224×224 dùng để test predict."""
    p = tmp_path / "sample.jpg"
    _make_image(p, size=224, color=(80, 200, 80))
    return p


@pytest.fixture
def class_names() -> list[str]:
    return ["fresh", "rotten"]
