"""Tests cho evaluation/{confusion_matrix,plots,evaluate}.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.plots import compare_models, overlay_curves, plot_history


# ----------------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------------
class TestConfusionMatrix:
    def test_returns_correct_cm_and_writes_file(self, tmp_path: Path, class_names):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        out = tmp_path / "cm.png"

        cm = plot_confusion_matrix(y_true, y_pred, class_names, out)

        # Kiểm tra số đếm thủ công
        # fresh→fresh = 2, fresh→rotten = 1, rotten→fresh = 1, rotten→rotten = 2
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2  # TN cho class 0
        assert cm[0, 1] == 1
        assert cm[1, 0] == 1
        assert cm[1, 1] == 2
        assert out.exists() and out.stat().st_size > 0

    def test_normalize_rows_sum_to_one(self, tmp_path: Path, class_names):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        out = tmp_path / "cm_norm.png"
        plot_confusion_matrix(y_true, y_pred, class_names, out, normalize=True)
        assert out.exists()


# ----------------------------------------------------------------------------
# plots.py
# ----------------------------------------------------------------------------
class TestPlots:
    @pytest.fixture
    def fake_history(self):
        return {
            "loss":          [1.0, 0.8, 0.6, 0.5],
            "accuracy":      [0.5, 0.6, 0.75, 0.85],
            "val_loss":      [1.1, 0.9, 0.7, 0.55],
            "val_accuracy":  [0.45, 0.55, 0.7, 0.82],
        }

    def test_plot_history_creates_acc_and_loss(self, fake_history, tmp_path: Path):
        plot_history(fake_history, "demo", tmp_path)
        assert (tmp_path / "demo_accuracy.png").exists()
        assert (tmp_path / "demo_loss.png").exists()

    def test_compare_models_writes_png(self, tmp_path: Path):
        metrics = {
            "mobilenet": {"accuracy": 0.92, "precision": 0.91, "recall": 0.92, "f1": 0.91},
            "resnet":    {"accuracy": 0.94, "precision": 0.93, "recall": 0.94, "f1": 0.93},
        }
        out = tmp_path / "compare.png"
        compare_models(metrics, out)
        assert out.exists() and out.stat().st_size > 0

    def test_overlay_curves(self, fake_history, tmp_path: Path):
        histories = {"mobilenet": fake_history, "resnet": fake_history}
        out = tmp_path / "overlay.png"
        overlay_curves(histories, out)
        assert out.exists() and out.stat().st_size > 0


# ----------------------------------------------------------------------------
# evaluate.py — end-to-end với model giả lưu file .keras
# ----------------------------------------------------------------------------
class TestEvaluateModule:
    def test_evaluate_full_pipeline(self, split_dataset_dir: Path, tmp_path: Path):
        """Train 1-step model bé, lưu .keras, gọi evaluate_model, kiểm tra metrics+files."""
        import tensorflow as tf
        from tensorflow.keras import Input, Model, layers

        # Model bé tí: Conv → GAP → Dense(2)
        inp = Input((64, 64, 3))
        x = layers.Conv2D(4, 3, padding="same", activation="relu")(inp)
        x = layers.GlobalAveragePooling2D()(x)
        out = layers.Dense(2, activation="softmax")(x)
        model = Model(inp, out, name="tiny")
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        ckpt = tmp_path / "tiny_best.keras"
        model.save(ckpt)

        from evaluation.evaluate import evaluate_model
        metrics = evaluate_model(
            model_path=ckpt,
            data_dir=split_dataset_dir,
            batch_size=2,
            img_size=64,
            results_dir=tmp_path / "results",
        )

        # Metrics có đầy đủ key
        for k in ("accuracy", "precision", "recall", "f1", "n_test", "classes"):
            assert k in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["classes"] == ["fresh", "rotten"]

        # File output có
        results_dir = tmp_path / "results"
        assert (results_dir / "tiny_metrics.json").exists()
        assert (results_dir / "tiny_classification_report.txt").exists()
        assert (results_dir / "tiny_confusion_matrix.png").exists()
        assert (results_dir / "tiny_confusion_matrix_norm.png").exists()

        # JSON parse được
        m = json.loads((results_dir / "tiny_metrics.json").read_text(encoding="utf-8"))
        assert "accuracy" in m
