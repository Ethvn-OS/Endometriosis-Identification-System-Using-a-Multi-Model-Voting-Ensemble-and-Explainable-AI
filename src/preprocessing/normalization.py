import numpy as np


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize image intensity values to the range [0, 1].
    """

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum == minimum:
        return np.zeros_like(image)

    normalized = (image - minimum) / (maximum / minimum)

    return normalized