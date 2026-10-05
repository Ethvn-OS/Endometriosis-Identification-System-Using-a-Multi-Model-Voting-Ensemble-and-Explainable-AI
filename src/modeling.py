from __future__ import annotations

import keras
from keras.applications import DenseNet121, EfficientNetV2S, ResNet50
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.resnet import preprocess_input as resnet_preprocess_input


MODEL_NAMES = ("ResNet50", "EfficientNetV2S", "DenseNet121")


@keras.saving.register_keras_serializable(package="endometriosis")
def resnet50_from_unit_interval(inputs):
    """Convert loader output in [0, 1] to ResNet50 ImageNet input."""

    return resnet_preprocess_input(inputs * 255.0)


@keras.saving.register_keras_serializable(package="endometriosis")
def densenet121_from_unit_interval(inputs):
    """Convert loader output in [0, 1] to DenseNet121 ImageNet input."""

    return densenet_preprocess_input(inputs * 255.0)


def build_classifier(
    model_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = "imagenet",
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Build one frozen-backbone binary classifier with correct preprocessing."""

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of {MODEL_NAMES}.")

    inputs = keras.Input(shape=input_shape, name="image")
    if model_name == "ResNet50":
        prepared = keras.layers.Lambda(
            resnet50_from_unit_interval, name="imagenet_preprocessing"
        )(inputs)
        trunk = ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )
    elif model_name == "DenseNet121":
        prepared = keras.layers.Lambda(
            densenet121_from_unit_interval, name="imagenet_preprocessing"
        )(inputs)
        trunk = DenseNet121(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        prepared = keras.layers.Rescaling(255.0, name="imagenet_preprocessing")(
            inputs
        )
        trunk = EfficientNetV2S(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
            include_preprocessing=True,
        )

    trunk.trainable = False
    features = trunk(prepared, training=False)
    features = keras.layers.Dropout(dropout_rate, name="dropout")(features)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="head")(features)
    return keras.Model(inputs, outputs, name=f"{model_name}_transfer")


def compile_classifier(model: keras.Model, learning_rate: float) -> None:
    """Compile a binary classifier for frozen-backbone training."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        weighted_metrics=[
            keras.metrics.BinaryAccuracy(name="slice_accuracy"),
            keras.metrics.AUC(name="slice_auc"),
        ],
    )


def soft_vote(probabilities):
    """Return equal-weight soft-voting probabilities."""

    return keras.ops.mean(keras.ops.stack(probabilities, axis=0), axis=0)
