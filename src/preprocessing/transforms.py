import cv2
import numpy as np

from src.preprocessing.normalization import min_max_normalize

def resize_slice(image: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize a 2D image slice to the target dimensions using cubic interpolation.
    """

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {image.ndim} dimensions.")

    resized = cv2.resize(
        image.astype(np.float32),
        dsize=target_size,
        interpolation=cv2.INTER_CUBIC,
    )

    return resized


def preprocess_slice(image: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Full preprocessing pipeline for a single slice: resize then min-max normalize.
    """

    resized = resize_slice(image, target_size)
    normalized = min_max_normalize(resized)

    return normalized
