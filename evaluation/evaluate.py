"""
evaluation/evaluate.py
======================
Đánh giá model trên tập test:
    * Tính accuracy, precision, recall, F1 (macro).
    * Lưu confusion matrix (raw + normalized).
    * Lưu classification report (txt + json).
    * Lưu metrics tổng hợp `<name>_metrics.json` để so sánh sau.

Cách dùng:
    python evaluation/evaluate.py --model checkpoints/mobilenet_best.keras
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from evaluation.confusion_matrix import plot_confusion_matrix
from preprocessing.augmentation import IMG_SIZE, build_eval_generator


def evaluate_model(model_path: Path,
                   data_dir: Path,
                   batch_size: int = 32,
                   img_size: int = IMG_SIZE,
                   results_dir: Path = Path("results")) -> dict:
    """Đánh giá 1 model trên data_dir/test/."""
    name = model_path.stem.replace("_best", "").replace("_final", "")
    print(f"\n[evaluate] model={name}  path={model_path}")

    model = tf.keras.models.load_model(model_path)
    test_gen = build_eval_generator(
        data_dir / "test", img_size=img_size, batch_size=batch_size, shuffle=False
    )
    class_names = list(test_gen.class_indices.keys())

    # Predict cả tập test
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes  # nhãn thật theo thứ tự generator (do shuffle=False)

    # Tính metrics (macro để cân bằng giữa các class)
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_test":    int(len(y_true)),
        "classes":   class_names,
    }

    print("\n=== Metrics ===")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k:<10s}: {metrics[k]:.4f}")

    # Classification report (chi tiết per-class)
    report_txt = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    print("\n=== Classification Report ===\n" + report_txt)

    # Lưu kết quả
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{name}_classification_report.txt").write_text(report_txt, encoding="utf-8")
    with open(results_dir / f"{name}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    with open(results_dir / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        out_path=results_dir / f"{name}_confusion_matrix.png",
        normalize=False,
        title=f"{name} — Confusion Matrix",
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        out_path=results_dir / f"{name}_confusion_matrix_norm.png",
        normalize=True,
        title=f"{name} — Confusion Matrix (normalized)",
    )

    print(f"\n[done] Lưu kết quả tại {results_dir}/{name}_*")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="Đường dẫn .keras / .h5")
    p.add_argument("--data-dir", type=Path, default=Path("dataset"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args()
    evaluate_model(args.model, args.data_dir, args.batch_size, args.img_size, args.results_dir)


if __name__ == "__main__":
    main()
