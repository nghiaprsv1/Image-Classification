"""Tests cho preprocessing/augmentation.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from preprocessing.augmentation import (
    build_eval_generator,
    build_train_generator,
    visualize_augmentation,
)


class TestBuildTrainGenerator:
    def test_shape_and_normalize(self, split_dataset_dir: Path):
        gen = build_train_generator(
            split_dataset_dir / "train", img_size=64, batch_size=4
        )
        x, y = next(gen)

        assert x.shape == (4, 64, 64, 3)
        assert y.shape == (4, 2)
        # rescale 1/255 → pixel trong [0,1]
        assert x.min() >= 0.0
        assert x.max() <= 1.0
        # Categorical one-hot: tổng mỗi hàng = 1
        assert np.allclose(y.sum(axis=1), 1.0)

    def test_class_indices_alphabetical(self, split_dataset_dir: Path):
        gen = build_train_generator(
            split_dataset_dir / "train", img_size=64, batch_size=2
        )
        # Theo quy ước: fresh=0, rotten=1
        assert gen.class_indices == {"fresh": 0, "rotten": 1}


class TestBuildEvalGenerator:
    def test_no_shuffle_and_correct_count(self, split_dataset_dir: Path):
        gen = build_eval_generator(
            split_dataset_dir / "test", img_size=64, batch_size=2, shuffle=False
        )
        # Theo fixture: 4 ảnh fresh + 4 ảnh rotten
        assert gen.samples == 8
        assert gen.class_indices == {"fresh": 0, "rotten": 1}

    def test_eval_pixels_normalized(self, split_dataset_dir: Path):
        gen = build_eval_generator(
            split_dataset_dir / "valid", img_size=64, batch_size=2
        )
        x, _ = next(gen)
        assert x.min() >= 0.0 and x.max() <= 1.0


class TestVisualizeAugmentation:
    def test_writes_png(self, split_dataset_dir: Path, tmp_path: Path):
        out = tmp_path / "aug.png"
        visualize_augmentation(split_dataset_dir / "train", out, n_samples=4, img_size=64)
        assert out.exists() and out.stat().st_size > 0
