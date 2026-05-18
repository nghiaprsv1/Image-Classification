"""Tests cho app/predict.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.predict import (
    Prediction,
    collect_images,
    load_and_preprocess,
    predict_one,
    visualize,
)


# ----------------------------------------------------------------------------
# load_and_preprocess
# ----------------------------------------------------------------------------
class TestLoadAndPreprocess:
    def test_shape_and_range(self, tiny_image: Path):
        x = load_and_preprocess(tiny_image, img_size=224)
        assert x.shape == (1, 224, 224, 3)
        assert x.dtype == np.float32
        assert x.min() >= 0.0
        assert x.max() <= 1.0

    def test_resize_changes_size(self, tiny_image: Path):
        x = load_and_preprocess(tiny_image, img_size=64)
        assert x.shape == (1, 64, 64, 3)

    def test_converts_rgba_to_rgb(self, tmp_path: Path):
        src = tmp_path / "rgba.png"
        Image.new("RGBA", (50, 50), color=(10, 20, 30, 255)).save(src)
        x = load_and_preprocess(src, img_size=32)
        assert x.shape == (1, 32, 32, 3)


# ----------------------------------------------------------------------------
# collect_images
# ----------------------------------------------------------------------------
class TestCollectImages:
    def test_single_file(self, tiny_image: Path):
        assert collect_images(tiny_image) == [tiny_image]

    def test_directory_recursive(self, split_dataset_dir: Path):
        files = collect_images(split_dataset_dir / "test")
        assert len(files) > 0
        # Tất cả là file ảnh
        assert all(f.suffix.lower() in {".jpg", ".jpeg", ".png"} for f in files)

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            collect_images(tmp_path / "nope")


# ----------------------------------------------------------------------------
# predict_one (dùng model giả)
# ----------------------------------------------------------------------------
class _DummyModel:
    """Model giả luôn predict 'fresh' với confidence 0.9."""
    def predict(self, x, verbose=0):
        return np.array([[0.9, 0.1]] * len(x))


class TestPredictOne:
    def test_returns_prediction_dataclass(self, tiny_image: Path, class_names):
        pred = predict_one(_DummyModel(), tiny_image, class_names, img_size=224)

        assert isinstance(pred, Prediction)
        assert pred.image_path == tiny_image
        assert pred.label == "fresh"
        assert pred.score == pytest.approx(0.9)
        assert pred.probs.shape == (2,)
        assert np.allclose(pred.probs.sum(), 1.0)


# ----------------------------------------------------------------------------
# visualize
# ----------------------------------------------------------------------------
class TestVisualize:
    def test_writes_visualisation(self, tiny_image: Path, tmp_path: Path):
        pred = Prediction(
            image_path=tiny_image,
            label="fresh",
            score=0.91,
            probs=np.array([0.91, 0.09]),
        )
        out = visualize(pred, tmp_path / "out")
        assert out.exists() and out.stat().st_size > 0
