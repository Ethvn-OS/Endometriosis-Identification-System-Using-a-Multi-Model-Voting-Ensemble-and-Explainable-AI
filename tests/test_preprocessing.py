import unittest
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from src.preprocessing.nifti import foreground_fractions, load_reoriented_nifti
from src.preprocessing.transforms import center_crop_or_pad, harmonize_slice


class PreprocessingTest(unittest.TestCase):
    def test_crop_and_pad_preserves_center(self):
        image = np.arange(24, dtype=np.float32).reshape(4, 6)
        cropped = center_crop_or_pad(image, (4, 4))
        padded = center_crop_or_pad(cropped, (6, 6))
        self.assertEqual(cropped.shape, (4, 4))
        self.assertEqual(padded.shape, (6, 6))
        np.testing.assert_array_equal(padded[1:5, 1:5], cropped)

    def test_harmonized_slice_contract(self):
        image = np.zeros((32, 20), dtype=np.float32)
        image[8:24, 5:15] = 10.0
        output = harmonize_slice(
            image,
            source_spacing=(0.5, 1.0),
            target_spacing_mm=1.0,
            target_fov_mm=40.0,
            output_size=(24, 24),
        )
        self.assertEqual(output.shape, (24, 24))
        self.assertEqual(output.dtype, np.float32)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_foreground_fraction_rejects_constant_volume(self):
        fractions = foreground_fractions(np.zeros((8, 8, 3), dtype=np.float32))
        np.testing.assert_array_equal(fractions, np.zeros(3, dtype=np.float32))

    def test_foreground_fraction_counts_low_contrast_anatomy(self):
        volume = np.zeros((10, 10, 2), dtype=np.float32)
        volume[2:8, 2:8, 0] = 1.0
        volume[4:6, 4:6, 0] = 100.0

        fractions = foreground_fractions(volume)

        self.assertGreaterEqual(float(fractions[0]), 0.35)
        self.assertEqual(float(fractions[1]), 0.0)

    def test_oblique_volume_is_resampled_to_orthogonal_lps(self):
        angle = np.deg2rad(20.0)
        affine = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle), 0.0],
                [0.0, np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "oblique.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((8, 8, 4), dtype=np.float32), affine), path)
            volume = load_reoriented_nifti(
                path, ("L", "P", "S"), target_inplane_spacing_mm=1.0
            )
        self.assertEqual(volume.canonical_orientation, ("L", "P", "S"))
        self.assertTrue(np.allclose(volume.canonical_spacing[:2], (1.0, 1.0)))
        self.assertLess(max(volume.canonical_obliquity_degrees), 1e-4)

    def test_axis_permutation_preserves_world_z_spacing(self):
        affine = np.array(
            [
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "permuted.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), affine), path)
            volume = load_reoriented_nifti(
                path, ("L", "P", "S"), target_inplane_spacing_mm=1.0
            )
        self.assertAlmostEqual(volume.canonical_spacing[2], 1.0)


if __name__ == "__main__":
    unittest.main()
