"""
evaluation/confusion_matrix.py
==============================
Vẽ confusion matrix dạng heatmap (raw count + normalized).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str],
                          out_path: Path,
                          normalize: bool = False,
                          title: str | None = None) -> np.ndarray:
    """Vẽ heatmap confusion matrix và lưu file ảnh. Trả về CM (numpy)."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(all="ignore"):
            cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_disp = np.nan_to_num(cm_disp)
        fmt = ".2f"
    else:
        cm_disp = cm.astype(int)
        fmt = "d"

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_disp,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title or ("Confusion Matrix (normalized)" if normalize else "Confusion Matrix"))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[saved] {out_path}")
    return cm
