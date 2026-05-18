"""
models/resnet_model.py
======================
Transfer Learning với ResNet50 (pretrained ImageNet).

Kiến trúc:
    Input(224,224,3)
        → ResNet50 base (frozen)
        → GlobalAveragePooling2D
        → Dropout(0.4) → Dense(256, ReLU) → BatchNorm → Dropout(0.4)
        → Dense(num_classes, softmax)
"""
from __future__ import annotations

from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.applications import ResNet50

IMG_SIZE = 224
BASE_NAME_PREFIX = "resnet50"


def _get_base(model: Model) -> Model:
    """Trả về sub-model backbone (ResNet50)."""
    for layer in model.layers:
        if isinstance(layer, Model) and layer.name.startswith(BASE_NAME_PREFIX):
            return layer
    raise ValueError("Không tìm thấy ResNet50 base trong model")


def build_resnet(num_classes: int = 2,
                 img_size: int = IMG_SIZE,
                 dropout: float = 0.4,
                 dense_units: int = 256,
                 weight_decay: float = 1e-4,
                 weights: str | None = "imagenet") -> Model:
    """Tạo model ResNet50 với base bị freeze.

    `weights="imagenet"` (mặc định) hoặc `None` (random init — dùng cho test offline).
    """
    base = ResNet50(
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

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
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name="ResNet50_Veggie")


def unfreeze_for_finetune(model: Model, n_layers: int = 40) -> Model:
    """Unfreeze N layer cuối ResNet50 cho fine-tune."""
    base = _get_base(model)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    # Giữ BatchNorm ở chế độ inference để ổn định
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return model


if __name__ == "__main__":
    m = build_resnet(num_classes=2)
    m.summary()
