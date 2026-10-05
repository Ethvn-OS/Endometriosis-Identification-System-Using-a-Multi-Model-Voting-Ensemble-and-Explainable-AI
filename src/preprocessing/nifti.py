from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


@dataclass(frozen=True)
class NiftiVolume:
    data: np.ndarray
    original_shape: tuple[int, int, int]
    original_spacing: tuple[float, float, float]
    original_orientation: tuple[str, str, str]
    canonical_spacing: tuple[float, float, float]
    canonical_orientation: tuple[str, str, str]
    original_obliquity_degrees: tuple[float, float, float]
    canonical_obliquity_degrees: tuple[float, float, float]


def load_nifti(file_path: str | Path) -> np.ndarray:
    """Load an unmodified NIfTI MRI volume as a NumPy array."""

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")
    return np.asarray(nib.load(file_path).get_fdata(dtype=np.float32))


def load_reoriented_nifti(
    file_path: str | Path,
    target_orientation: tuple[str, str, str] = ("L", "P", "S"),
    target_inplane_spacing_mm: float | None = None,
) -> NiftiVolume:
    """Load and resample a 3D NIfTI onto an orthogonal shared orientation."""

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    image = nib.load(file_path)
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D MRI volume, got shape {image.shape}.")

    original_orientation = tuple(str(code) for code in nib.aff2axcodes(image.affine))
    original_obliquity = tuple(
        float(np.degrees(value)) for value in nib.affines.obliquity(image.affine)
    )
    current_source = nib.orientations.io_orientation(image.affine)
    target_source = nib.orientations.axcodes2ornt(target_orientation)
    source_transform = nib.orientations.ornt_transform(current_source, target_source)
    source_reoriented = image.as_reoriented(source_transform)

    inplane_spacing = (
        float(target_inplane_spacing_mm)
        if target_inplane_spacing_mm is not None
        else float(source_reoriented.header.get_zooms()[0])
    )
    slice_spacing = float(source_reoriented.header.get_zooms()[2])
    orthogonal = resample_to_output(
        source_reoriented,
        voxel_sizes=(inplane_spacing, inplane_spacing, slice_spacing),
        order=3,
    )

    current = nib.orientations.io_orientation(orthogonal.affine)
    target = nib.orientations.axcodes2ornt(target_orientation)
    transform = nib.orientations.ornt_transform(current, target)
    canonical = orthogonal.as_reoriented(transform)
    canonical_orientation = tuple(
        str(code) for code in nib.aff2axcodes(canonical.affine)
    )
    canonical_obliquity = tuple(
        float(np.degrees(value)) for value in nib.affines.obliquity(canonical.affine)
    )

    return NiftiVolume(
        data=np.asarray(canonical.get_fdata(dtype=np.float32)),
        original_shape=tuple(int(value) for value in image.shape),
        original_spacing=tuple(float(value) for value in image.header.get_zooms()[:3]),
        original_orientation=original_orientation,
        canonical_spacing=tuple(
            float(value) for value in canonical.header.get_zooms()[:3]
        ),
        canonical_orientation=canonical_orientation,
        original_obliquity_degrees=original_obliquity,
        canonical_obliquity_degrees=canonical_obliquity,
    )


def extract_axial_slices(volume: np.ndarray) -> list[np.ndarray]:
    """Extract axial slices from a 3D MRI volume."""

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume, got {volume.ndim} dimensions.")
    return [volume[:, :, index] for index in range(volume.shape[2])]


def foreground_fractions(
    volume: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """Estimate label-blind foreground coverage for every axial slice."""

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume, got {volume.ndim} dimensions.")
    if not np.isfinite(volume).all():
        raise ValueError("Volume contains NaN or infinite values.")

    low, high = np.percentile(volume, [low_percentile, high_percentile])
    if high <= low:
        return np.zeros(volume.shape[2], dtype=np.float32)
    # Background is near the low intensity floor; a midpoint threshold measures
    # only bright tissue and can incorrectly reject anatomy-rich MRI slices.
    threshold = float(low) + 0.01 * (float(high) - float(low))
    return np.mean(volume > threshold, axis=(0, 1), dtype=np.float64).astype(np.float32)
