"""
evaluation/plots.py
===================
Vẽ biểu đồ:
    * Accuracy / Loss curves từ history JSON.
    * So sánh hai model (MobileNetV2 vs ResNet50) — bar chart metrics + curve overlay.

Cách dùng:
    python evaluation/plots.py --history results/mobilenet_history.json --name mobilenet
    python evaluation/plots.py    # chế độ so sánh: đọc cả 2 history + 2 metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------
# 1. Đường cong accuracy / loss của 1 model
# ----------------------------------------------------------------------------
def plot_history(history: Dict[str, list], name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    # Accuracy
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["accuracy"], "b-o", label="train", markersize=3)
    if "val_accuracy" in history:
        plt.plot(epochs, history["val_accuracy"], "r-s", label="valid", markersize=3)
    plt.title(f"{name} — Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_accuracy.png", dpi=120)
    plt.close()

    # Loss
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["loss"], "b-o", label="train", markersize=3)
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], "r-s", label="valid", markersize=3)
    plt.title(f"{name} — Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_loss.png", dpi=120)
    plt.close()
    print(f"[saved] {out_dir}/{name}_accuracy.png, {name}_loss.png")


# ----------------------------------------------------------------------------
# 2. So sánh 2 model
# ----------------------------------------------------------------------------
def compare_models(metrics: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """Bar chart so sánh accuracy / precision / recall / f1 giữa các model."""
    models = list(metrics.keys())
    keys = ["accuracy", "precision", "recall", "f1"]
    width = 0.2
    x = np.arange(len(keys))

    plt.figure(figsize=(8, 4.5))
    for i, m in enumerate(models):
        vals = [metrics[m].get(k, 0.0) for k in keys]
        bars = plt.bar(x + i * width, vals, width, label=m)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8)

    plt.xticks(x + width * (len(models) - 1) / 2, [k.title() for k in keys])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("So sánh các mô hình trên tập Test")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[saved] {out_path}")


def overlay_curves(histories: Dict[str, Dict[str, list]], out_path: Path) -> None:
    """Vẽ overlay val_accuracy của các model lên cùng 1 biểu đồ."""
    plt.figure(figsize=(7, 4))
    for name, h in histories.items():
        if "val_accuracy" in h:
            plt.plot(range(1, len(h["val_accuracy"]) + 1),
                     h["val_accuracy"], "-o", markersize=3, label=name)
    plt.title("Val Accuracy — So sánh các model")
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[saved] {out_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, default=None,
                   help="History JSON của 1 model. Bỏ qua để chạy chế độ so sánh.")
    p.add_argument("--name", type=str, default="model")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    # Single-model mode
    if args.history is not None:
        with open(args.history, encoding="utf-8") as f:
            history = json.load(f)
        plot_history(history, args.name, args.results_dir)
        return

    # Compare mode: tự tìm các *_history.json + *_metrics.json
    histories, metrics = {}, {}
    for hp in args.results_dir.glob("*_history.json"):
        name = hp.stem.replace("_history", "")
        with open(hp, encoding="utf-8") as f:
            histories[name] = json.load(f)
        plot_history(histories[name], name, args.results_dir)
    for mp in args.results_dir.glob("*_metrics.json"):
        name = mp.stem.replace("_metrics", "")
        with open(mp, encoding="utf-8") as f:
            metrics[name] = json.load(f)

    if histories:
        overlay_curves(histories, args.results_dir / "compare_val_accuracy.png")
    if metrics:
        compare_models(metrics, args.results_dir / "compare_metrics.png")
        print("\n=== Bảng so sánh ===")
        keys = ["accuracy", "precision", "recall", "f1"]
        header = f"{'model':<15s} " + " ".join(f"{k:>10s}" for k in keys)
        print(header)
        print("-" * len(header))
        for name, m in metrics.items():
            print(f"{name:<15s} " + " ".join(f"{m.get(k, 0):>10.4f}" for k in keys))


if __name__ == "__main__":
    main()
