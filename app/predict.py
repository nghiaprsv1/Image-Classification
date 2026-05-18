"""
app/predict.py
==============
Ứng dụng dự đoán: load model đã train, dự đoán Fresh / Rotten cho ảnh đầu vào.

Cách dùng:
    # 1 ảnh
    python app/predict.py --image path/to/img.jpg --model checkpoints/mobilenet_best.keras

    # nhiều ảnh trong thư mục
    python app/predict.py --image path/to/folder --model checkpoints/mobilenet_best.keras

    # hiển thị visualisation
    python app/predict.py --image img.jpg --model checkpoints/mobilenet_best.keras --show

Output:
    label   — fresh / rotten
    score   — confidence của lớp dự đoán (%)
    + (tuỳ chọn) ảnh kết quả lưu tại results/predictions/
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# Phải khớp với thứ tự alphabet ImageDataGenerator: fresh=0, rotten=1
DEFAULT_CLASSES = ["fresh", "rotten"]
IMG_SIZE = 224
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Prediction:
    image_path: Path
    label: str
    score: float
    probs: np.ndarray  # shape (num_classes,)


# ----------------------------------------------------------------------------
# Pre-process 1 ảnh
# ----------------------------------------------------------------------------
def load_and_preprocess(path: Path, img_size: int = IMG_SIZE) -> np.ndarray:
    """Đọc ảnh → RGB → resize → normalize [0,1] → (1, H, W, 3)."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def collect_images(target: Path) -> List[Path]:
    """Nếu target là file → trả [target], nếu là folder → quét đệ quy."""
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in VALID_EXTS)
    raise FileNotFoundError(target)


# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
def predict_one(model, path: Path, classes: List[str], img_size: int) -> Prediction:
    x = load_and_preprocess(path, img_size)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return Prediction(
        image_path=path,
        label=classes[idx],
        score=float(probs[idx]),
        probs=probs,
    )


def visualize(pred: Prediction, out_dir: Path) -> Path:
    """Lưu ảnh kèm overlay nhãn + confidence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{pred.image_path.stem}.png"

    img = Image.open(pred.image_path).convert("RGB")
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    color = "#2E7D32" if pred.label == "fresh" else "#C62828"
    plt.title(f"{pred.label.upper()}  ({pred.score * 100:.1f}%)",
              color=color, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Dự đoán Fresh / Rotten cho ảnh rau")
    p.add_argument("--image", type=Path, required=True,
                   help="Đường dẫn 1 ảnh hoặc 1 thư mục ảnh")
    p.add_argument("--model", type=Path, required=True,
                   help="Đường dẫn checkpoint .keras / .h5")
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES),
                   help="Danh sách class theo thứ tự index")
    p.add_argument("--show", action="store_true", help="Lưu ảnh visualisation")
    p.add_argument("--out-dir", type=Path, default=Path("results/predictions"))
    args = p.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    print(f"[info] loading {args.model}")
    model = tf.keras.models.load_model(args.model)

    paths = collect_images(args.image)
    if not paths:
        print("[warn] Không tìm thấy ảnh hợp lệ.")
        return

    print(f"[info] predict {len(paths)} ảnh")
    print(f"{'IMAGE':<60s} {'LABEL':<10s} {'SCORE':>8s}")
    print("-" * 82)
    for ip in paths:
        try:
            pred = predict_one(model, ip, classes, args.img_size)
        except (UnidentifiedImageError, OSError) as e:
            print(f"[skip] {ip.name}: {e}")
            continue
        print(f"{str(ip)[:58]:<60s} {pred.label:<10s} {pred.score * 100:>7.2f}%")
        if args.show:
            saved = visualize(pred, args.out_dir)
            print(f"       └─ saved: {saved}")


if __name__ == "__main__":
    main()
