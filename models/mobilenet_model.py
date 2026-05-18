"""
models/mobilenet_model.py
=========================
Transfer Learning với MobileNetV2 (pretrained ImageNet).

Kiến trúc:
    Input(224,224,3)
        → MobileNetV2 base (frozen)
        → GlobalAveragePooling2D
        → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.3)
        → Dense(num_classes, softmax)

Hàm chính:
    * `build_mobilenet(num_classes, ...)`            — model phase 1 (freeze base).
    * `unfreeze_for_finetune(model, n_layers=30)`    — mở khoá top N layer cho fine-tune.
"""
from __future__ import annotations

from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = 224
BASE_NAME_PREFIX = "mobilenetv2"  # MobileNetV2 layer name luôn bắt đầu thế này


def _get_base(model: Model) -> Model:
    """Trả về sub-model backbone (MobileNetV2) bên trong model classifier."""
    for layer in model.layers:
        if isinstance(layer, Model) and layer.name.startswith(BASE_NAME_PREFIX):
            return layer
    raise ValueError("Không tìm thấy MobileNetV2 base trong model")


def build_mobilenet(num_classes: int = 2,
                    img_size: int = IMG_SIZE,
                    dropout: float = 0.3,
                    dense_units: int = 128,
                    weight_decay: float = 1e-4,
                    weights: str | None = "imagenet") -> Model:
    """Tạo model MobileNetV2 với base bị freeze.

    `weights="imagenet"` (mặc định) hoặc `None` (random init — dùng cho test offline).
    """
    base = MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False         # Freeze toàn bộ base

    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay),
        name="fc1",
    )(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="MobileNetV2_Veggie")
    return model


def unfreeze_for_finetune(model: Model, n_layers: int = 30) -> Model:
    """
    Unfreeze N layer cuối của MobileNetV2 base để fine-tune.

    Lưu ý: gọi `model.compile(...)` lại sau khi unfreeze.
    """
    base = _get_base(model)
    base.trainable = True
    # Freeze BatchNorm để tránh phá hỏng running stats
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return model


if __name__ == "__main__":
    m = build_mobilenet(num_classes=2)
    m.summary()
