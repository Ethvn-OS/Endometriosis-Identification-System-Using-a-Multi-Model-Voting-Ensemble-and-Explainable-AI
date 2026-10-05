import cv2
import numpy as np

from src.preprocessing.normalization import min_max_normalize


def resize_slice(
    image: np.ndarray, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Resize a 2D image to ``(height, width)`` without changing its dtype."""

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {image.ndim} dimensions.")
    target_h, target_w = target_size
    interpolation = (
        cv2.INTER_AREA
        if image.shape[0] >= target_h and image.shape[1] >= target_w
        else cv2.INTER_CUBIC
    )
    return cv2.resize(
        image.astype(np.float32),
        dsize=(target_w, target_h),
        interpolation=interpolation,
    )


def center_crop_or_pad(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Center-crop or zero-pad a 2D image to ``(height, width)``."""

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {image.ndim} dimensions.")

    target_h, target_w = target_size
    height, width = image.shape
    crop_top = max((height - target_h) // 2, 0)
    crop_left = max((width - target_w) // 2, 0)
    cropped = image[
        crop_top : crop_top + min(height, target_h),
        crop_left : crop_left + min(width, target_w),
    ]

    pad_h = target_h - cropped.shape[0]
    pad_w = target_w - cropped.shape[1]
    return np.pad(
        cropped,
        (
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        ),
        mode="constant",
        constant_values=0.0,
    )


def harmonize_slice(
    image: np.ndarray,
    source_spacing: tuple[float, float],
    target_spacing_mm: float,
    target_fov_mm: float,
    output_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Normalize, resample, crop/pad, and resize one axial MRI slice."""

    if target_spacing_mm <= 0 or target_fov_mm <= 0:
        raise ValueError("Target spacing and field of view must be positive.")

    normalized = min_max_normalize(image)
    resampled_h = max(1, int(round(image.shape[0] * source_spacing[0] / target_spacing_mm)))
    resampled_w = max(1, int(round(image.shape[1] * source_spacing[1] / target_spacing_mm)))
    resampled = resize_slice(normalized, (resampled_h, resampled_w))

    grid_size = int(round(target_fov_mm / target_spacing_mm))
    standardized = center_crop_or_pad(resampled, (grid_size, grid_size))
    output = resize_slice(standardized, output_size)
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def preprocess_slice(
    image: np.ndarray, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Legacy slice preprocessing used by the exploratory notebooks."""

    return min_max_normalize(resize_slice(image, target_size))
