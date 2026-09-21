import numpy as np


def to_three_channels(image: np.ndarray) -> np.ndarray:
    """
    Expand a 2D grayscale slice into a 3-channel array (RGB-compatible).

    Accepts a single-channel array of shape ``(H, W)`` or ``(H, W, 1)`` and
    returns the same values repeated across three channels, preserving dtype.
    Already-3-channel arrays are returned unchanged.

    This is required because the pretrained CNN backbones (ResNet50,
    EfficientNetV2, DenseNet121) expect ``(H, W, 3)`` inputs.
    """

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2D slice (H, W) or (H, W, 1), got {image.ndim} dimensions."
        )

    if image.ndim == 3 and image.shape[2] == 3:
        return image

    if image.ndim == 3 and image.shape[2] != 1:
        raise ValueError(
            f"Unsupported channel count: {image.shape[2]}. Expected 1 or 3."
        )

    single_channel = image if image.ndim == 2 else image[:, :, 0]

    return np.stack([single_channel] * 3, axis=-1)