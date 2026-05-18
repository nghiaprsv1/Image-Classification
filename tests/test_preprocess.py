"""Tests cho preprocessing/preprocess.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocessing.preprocess import (
    plot_samples,
    plot_split_distribution,
    preprocess_all,
    process_image,
    split_dataset,
)


# ----------------------------------------------------------------------------
# process_image
# ----------------------------------------------------------------------------
class TestProcessImage:
    def test_resize_and_save(self, tmp_path: Path):
        src = tmp_path / "in.png"
        Image.new("RGB", (300, 200), color=(10, 20, 30)).save(src)
        dst = tmp_path / "out" / "out.jpg"

        ok = process_image(src, dst, img_size=128)

        assert ok is True
        assert dst.exists()
        with Image.open(dst) as im:
            assert im.size == (128, 128)
            assert im.mode == "RGB"

    def test_returns_false_for_invalid_file(self, tmp_path: Path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        dst = tmp_path / "out.jpg"

        ok = process_image(bad, dst, img_size=64)

        assert ok is False
        assert not dst.exists()

    def test_converts_grayscale_to_rgb(self, tmp_path: Path):
        src = tmp_path / "gray.png"
        Image.new("L", (100, 100), color=128).save(src)
        dst = tmp_path / "rgb.jpg"

        assert process_image(src, dst, 64)
        with Image.open(dst) as im:
            assert im.mode == "RGB"


# ----------------------------------------------------------------------------
# preprocess_all
# ----------------------------------------------------------------------------
class TestPreprocessAll:
    def test_processes_all_classes(self, raw_dataset: Path, tmp_path: Path):
        processed = tmp_path / "processed"
        summary = preprocess_all(raw_dataset, processed, img_size=64)

        assert "fresh" in summary and "rotten" in summary
        assert summary["fresh"] >= 6     # ảnh hợp lệ + tiny vẫn được resize lên 64
        assert summary["rotten"] == 6
        # File rác bị bỏ qua
        assert not (processed / "fresh" / "fresh_broken.jpg").exists()


# ----------------------------------------------------------------------------
# split_dataset
# ----------------------------------------------------------------------------
class TestSplitDataset:
    def test_split_ratios_and_no_overlap(self, raw_dataset: Path, tmp_path: Path):
        # Bước 1: resize sang processed
        processed = tmp_path / "processed"
        preprocess_all(raw_dataset, processed, img_size=64)

        # Bước 2: split
        dst = tmp_path / "split"
        summary = split_dataset(processed, dst, ratios=(0.7, 0.15, 0.15))

        for cls, parts in summary.items():
            assert set(parts.keys()) == {"train", "valid", "test"}
            assert sum(parts.values()) > 0
        # Không file nào lặp giữa train/valid/test
        all_files = []
        for split in ("train", "valid", "test"):
            for cls in ("fresh", "rotten"):
                d = dst / split / cls
                if d.exists():
                    all_files.extend(p.name for p in d.iterdir())
        assert len(all_files) == len(set(all_files))

    def test_invalid_ratios_raises(self, raw_dataset: Path, tmp_path: Path):
        processed = tmp_path / "processed"
        preprocess_all(raw_dataset, processed, img_size=64)
        with pytest.raises(AssertionError):
            split_dataset(processed, tmp_path / "split", ratios=(0.5, 0.3, 0.3))


# ----------------------------------------------------------------------------
# Visualisations
# ----------------------------------------------------------------------------
class TestVisualisations:
    def test_plot_samples_creates_png(self, raw_dataset: Path, tmp_path: Path):
        out = tmp_path / "samples.png"
        plot_samples(raw_dataset, out, n_per_class=3)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_split_distribution(self, tmp_path: Path):
        summary = {
            "fresh":  {"train": 7, "valid": 2, "test": 1},
            "rotten": {"train": 6, "valid": 2, "test": 2},
        }
        out = tmp_path / "split.png"
        plot_split_distribution(summary, out)
        assert out.exists() and out.stat().st_size > 0
