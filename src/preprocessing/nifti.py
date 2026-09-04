from pathlib import Path
import nibabel as nib
import numpy as np


def load_nifti(file_path: str | Path) -> np.ndarray:
    """
    Load a NIfTI MRI volume and return it as a NumPy array.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    nii_image = nib.load(file_path)

    volume = nii_image.get_fdata()

    return np.asarray(volume)


def extract_axial_slices(volume: np.ndarray) -> list[np.ndarray]:
    """
    Extract axial slices from a 3D MRI volume.

    Returns:
        List of 2D  NumPy arrays.
    """

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume, got  {volume.ndim} dimensions."
        )

    slices  = [
        volume[:, :, index]
        for index in range(volume.shape[2])
    ]

    return slices