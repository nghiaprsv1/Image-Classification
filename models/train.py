"""
models/train.py
===============
Pipeline huấn luyện 2 pha (freeze → fine-tune) cho MobileNetV2 / ResNet50.

Cách dùng:
    python models/train.py --model mobilenet --epochs 30 --batch-size 32
    python models/train.py --model resnet    --epochs 30 --batch-size 16

Sinh ra:
    checkpoints/<name>_best.keras       — model tốt nhất (val_accuracy)
    checkpoints/<name>_final.keras      — model cuối cùng
    logs/<name>/                        — TensorBoard log
    results/<name>_history.json         — lịch sử train (accuracy, loss, ...)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Cho phép import package từ root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tensorflow as tf
from tensorflow.keras import callbacks as cb
from tensorflow.keras import metrics as km
from tensorflow.keras.optimizers import Adam

from preprocessing.augmentation import (
    BATCH_SIZE,
    IMG_SIZE,
    build_eval_generator,
    build_train_generator,
)
from models.mobilenet_model import build_mobilenet
from models.mobilenet_model import unfreeze_for_finetune as unfreeze_mobilenet
from models.resnet_model import build_resnet
from models.resnet_model import unfreeze_for_finetune as unfreeze_resnet

# ----------------------------------------------------------------------------
# Cấu hình GPU memory growth (tránh chiếm hết VRAM laptop)
# ----------------------------------------------------------------------------
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

SEED = 42
tf.keras.utils.set_random_seed(SEED)


MODEL_REGISTRY = {
    "mobilenet": (build_mobilenet, unfreeze_mobilenet, 30),
    "resnet":    (build_resnet,    unfreeze_resnet,    40),
}


def get_callbacks(name: str, ckpt_dir: Path, log_dir: Path,
                  patience_es: int = 7, patience_lr: int = 3) -> list:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return [
        cb.ModelCheckpoint(
            filepath=str(ckpt_dir / f"{name}_best.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        cb.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        cb.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
        cb.TensorBoard(log_dir=str(log_dir), update_freq="epoch"),
        cb.CSVLogger(filename=str(log_dir / "training.csv"), append=False),
    ]


def compile_model(model, lr: float):
    """Compile chuẩn: Adam + categorical_crossentropy + acc/precision/recall."""
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            km.Precision(name="precision"),
            km.Recall(name="recall"),
        ],
    )


def merge_history(h1, h2) -> dict:
    """Nối lịch sử của 2 pha train."""
    out = {k: list(v) for k, v in h1.history.items()}
    for k, v in h2.history.items():
        out.setdefault(k, []).extend(list(v))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_REGISTRY), required=True)
    p.add_argument("--data-dir", type=Path, default=Path("dataset"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--log-dir",  type=Path, default=Path("logs"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--epochs", type=int, default=30, help="epoch cho pha 1")
    p.add_argument("--ft-epochs", type=int, default=15, help="epoch cho fine-tune")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ft-lr", type=float, default=1e-5)
    p.add_argument("--no-finetune", action="store_true")
    p.add_argument("--mixed-precision", action="store_true",
                   help="Bật mixed precision (float16) — tăng tốc 1.5-2x trên Colab T4/V100/A100")
    p.add_argument("--xla", action="store_true",
                   help="Bật XLA JIT compile — tăng tốc thêm ~10-20% trên GPU")
    args = p.parse_args()

    # ------------------------- TỐI ƯU GPU -------------------------
    if args.mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[opt] mixed_float16 policy ON")
    if args.xla:
        tf.config.optimizer.set_jit(True)
        print("[opt] XLA JIT ON")

    # ------------------------- DATA -------------------------
    train_gen = build_train_generator(
        args.data_dir / "train", img_size=args.img_size, batch_size=args.batch_size
    )
    valid_gen = build_eval_generator(
        args.data_dir / "valid", img_size=args.img_size, batch_size=args.batch_size
    )
    num_classes = train_gen.num_classes
    print(f"[info] classes = {train_gen.class_indices}")

    # ------------------------- MODEL -------------------------
    builder, unfreezer, n_unfreeze = MODEL_REGISTRY[args.model]
    model = builder(num_classes=num_classes, img_size=args.img_size)
    compile_model(model, lr=args.lr)
    model.summary()

    cbs = get_callbacks(args.model, args.ckpt_dir, args.log_dir / args.model)

    # ------------------------- PHASE 1: FREEZE -------------------------
    print("\n=== Phase 1: train classifier head (base frozen) ===")
    h1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
        callbacks=cbs,
        verbose=2,
    )

    # ------------------------- PHASE 2: FINE-TUNE -------------------------
    history = h1.history
    if not args.no_finetune:
        print("\n=== Phase 2: fine-tune top layers ===")
        model = unfreezer(model, n_layers=n_unfreeze)
        compile_model(model, lr=args.ft_lr)
        h2 = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=args.ft_epochs,
            callbacks=cbs,
            verbose=2,
        )
        history = merge_history(h1, h2)

    # ------------------------- SAVE -------------------------
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.ckpt_dir / f"{args.model}_final.keras")

    with open(args.results_dir / f"{args.model}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[done] Saved → {args.ckpt_dir}/{args.model}_*.keras")


if __name__ == "__main__":
    main()
