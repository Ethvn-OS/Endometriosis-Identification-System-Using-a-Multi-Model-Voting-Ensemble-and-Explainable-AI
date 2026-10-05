import tempfile
import unittest
from pathlib import Path

import pandas as pd
import nibabel as nib
import numpy as np

from src.data.dataset import discover_patients, split_patients


class DatasetTest(unittest.TestCase):
    def test_discovery_uses_explicit_cohort_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for value, site in enumerate(("D1_MHS", "D2_TCPW"), start=1):
                patient = root / "UT-EndoMRI" / site / site.replace("_MHS", "-001").replace("_TCPW", "-001")
                patient.mkdir(parents=True)
                nib.save(
                    nib.Nifti1Image(
                        np.full((3, 3, 2), value, dtype=np.float32), np.eye(4)
                    ),
                    patient / f"{patient.name}_T2.nii.gz",
                )
            mog = root / "MOGaMBOv0" / "P1"
            mog.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(np.full((3, 3, 2), 3, dtype=np.float32), np.eye(4)),
                mog / "mriP1raw.nii",
            )

            config = {
                "preprocessing": {"sequences": ["T2"]},
                "datasets": {
                    "ut_endomri": {
                        "path": str(root / "UT-EndoMRI"),
                        "label": 1,
                        "cohort_role": "confirmed_positive",
                        "sequence": "T2",
                        "sequence_evidence": "test",
                        "label_evidence": "test",
                        "source_reference": "test",
                        "sites": [
                            {"name": "UT-D1", "directory": "D1_MHS", "expected_patients_raw": 1, "expected_unique_volumes": 1},
                            {"name": "UT-D2", "directory": "D2_TCPW", "expected_patients_raw": 1, "expected_unique_volumes": 1},
                        ],
                    },
                    "mogambo": {
                        "path": str(root / "MOGaMBOv0"),
                        "label": 0,
                        "cohort_role": "proxy_comparator",
                        "sequence": "T2",
                        "sequence_evidence": "test",
                        "label_evidence": "test",
                        "source_reference": "test",
                        "source_site": "MOGaMBO",
                        "expected_patients_raw": 1,
                        "expected_unique_volumes": 1,
                    },
                },
            }

            records = discover_patients(config)
            self.assertEqual(len(records), 3)
            self.assertEqual({record.sequence for record in records}, {"T2"})
            self.assertEqual(
                {record.patient_id: record.label for record in records},
                {"UT-D1-001": 1, "UT-D2-001": 1, "MOG-P1": 0},
            )

    def test_three_way_split_is_patient_disjoint_and_source_stratified(self):
        rows = []
        for site, label in (("UT-D1", 1), ("UT-D2", 1), ("MOGaMBO", 0)):
            for index in range(20):
                rows.append(
                    {
                        "patient_id": f"{site}-{index}",
                        "source_site": site,
                        "label": label,
                    }
                )
        result = split_patients(pd.DataFrame(rows), 0.70, 0.15, 0.15, seed=42)

        self.assertEqual(set(result["split"]), {"train", "validation", "test"})
        self.assertEqual(result["patient_id"].nunique(), len(result))
        for _, group in result.groupby("split"):
            self.assertEqual(set(group["source_site"]), {"UT-D1", "UT-D2", "MOGaMBO"})


if __name__ == "__main__":
    unittest.main()
