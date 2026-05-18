"""Tests cho models/mobilenet_model.py & models/resnet_model.py.

Dùng `weights=None` để KHÔNG cần tải ImageNet weights — test chạy offline,
nhanh, không phụ thuộc mạng.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.mobilenet_model import build_mobilenet, _get_base as _get_base_mb
from models.mobilenet_model import unfreeze_for_finetune as unfreeze_mobilenet
from models.resnet_model import build_resnet, _get_base as _get_base_rn
from models.resnet_model import unfreeze_for_finetune as unfreeze_resnet


# ----------------------------------------------------------------------------
# MobileNetV2
# ----------------------------------------------------------------------------
class TestMobileNet:
    def test_output_shape_softmax(self):
        model = build_mobilenet(num_classes=2, img_size=96, weights=None)
        assert model.input_shape == (None, 96, 96, 3)
        assert model.output_shape == (None, 2)

        x = np.random.rand(2, 96, 96, 3).astype("float32")
        y = model.predict(x, verbose=0)
        assert y.shape == (2, 2)
        # Softmax → mỗi hàng cộng = 1
        assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)
        assert (y >= 0).all() and (y <= 1).all()

    def test_base_is_frozen_after_build(self):
        model = build_mobilenet(num_classes=2, img_size=96, weights=None)
        base = _get_base_mb(model)
        assert base.trainable is False

    def test_unfreeze_top_layers(self):
        model = build_mobilenet(num_classes=2, img_size=96, weights=None)
        unfreeze_mobilenet(model, n_layers=10)
        base = _get_base_mb(model)
        assert base.trainable is True
        n_trainable = sum(1 for L in base.layers if L.trainable)
        assert n_trainable >= 1
        from tensorflow.keras import layers as KL
        for L in base.layers:
            if isinstance(L, KL.BatchNormalization):
                assert L.trainable is False


# ----------------------------------------------------------------------------
# ResNet50
# ----------------------------------------------------------------------------
class TestResNet:
    @pytest.fixture(scope="class")
    def model(self):
        # ResNet50 hơi nặng — dùng scope class để chỉ build 1 lần
        return build_resnet(num_classes=2, img_size=96, weights=None)

    def test_output_shape_softmax(self, model):
        assert model.input_shape == (None, 96, 96, 3)
        assert model.output_shape == (None, 2)
        x = np.random.rand(1, 96, 96, 3).astype("float32")
        y = model.predict(x, verbose=0)
        assert y.shape == (1, 2)
        assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)

    def test_base_frozen(self, model):
        assert _get_base_rn(model).trainable is False

    def test_unfreeze(self, model):
        unfreeze_resnet(model, n_layers=10)
        base = _get_base_rn(model)
        assert base.trainable is True
        n_trainable = sum(1 for L in base.layers if L.trainable)
        assert n_trainable >= 1


# ----------------------------------------------------------------------------
# Compile + 1 step train (smoke test pipeline đầy đủ)
# ----------------------------------------------------------------------------
class TestTrainOneStep:
    def test_mobilenet_can_train_one_step(self):
        from tensorflow.keras.optimizers import Adam
        model = build_mobilenet(num_classes=2, img_size=64, weights=None)
        model.compile(
            optimizer=Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        x = np.random.rand(4, 64, 64, 3).astype("float32")
        y = np.eye(2)[np.array([0, 1, 0, 1])].astype("float32")
        history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
        assert "loss" in history.history
        assert np.isfinite(history.history["loss"][0])
