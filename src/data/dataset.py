from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEQUENCE_KEYWORDS = ("T1", "T1FS", "T2", "T2FS")
LABEL_KEYWORDS = ("ov", "ut", "em", "cy", "cds", "_r1", "_r2", "_r3", "re3", "_pat")


def _is_sequence_file(filename: str, sequences: list[str]) -> bool:
    """
    Determine whether a filename corresponds to an MRI sequence file
    rather than a segmentation label file.
    """

    for keyword in LABEL_KEYWORDS:
        if keyword in filename:
            return False

    for seq in sequences:
        if f"_{seq}." in filename or f"_{seq}_" in filename:
            return True

    return False


def get_sequence_files(patient_dir: Path, sequences: list[str]) -> dict[str, Path]:
    """
    Find available MRI sequence files for a patient directory.

    Returns:
        Mapping of sequence name to file path for each available sequence.
    """

    found: dict[str, Path] = {}

    for file in sorted(patient_dir.iterdir()):
        if not file.is_file():
            continue

        for seq in sequences:
            if f"_{seq}." in file.name and seq not in found:
                found[seq] = file

    return found


def discover_patients(raw_dir: Path, sequences: list[str]) -> dict[str, list[Path]]:
    """
    Walk the raw data directories and discover all patient folders.

    Scans D1_MHS and D2_TCPW under UT-EndoMRI, as well as healthy_pelvis.

    Returns:
        Mapping of patient ID to list of MRI sequence file paths.
    """

    patients: dict[str, list[Path]] = {}

    ut_endomri_dir = raw_dir / "UT-EndoMRI"

    for sub_dir_name in ("D1_MHS", "D2_TCPW"):
        sub_dir = ut_endomri_dir / sub_dir_name
        if not sub_dir.exists():
            continue

        for patient_dir in sorted(sub_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            seq_files = get_sequence_files(patient_dir, sequences)
            if seq_files:
                patients[patient_dir.name] = list(seq_files.values())

    healthy_dir = raw_dir / "healthy_pelvis"
    if healthy_dir.exists():
        for patient_dir in sorted(healthy_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            seq_files = get_sequence_files(patient_dir, sequences)
            if seq_files:
                patients[patient_dir.name] = list(seq_files.values())

    return patients


def assign_label(patient_id: str) -> int:
    """
    Assign a binary label based on dataset origin.

    UT-EndoMRI patient IDs start with 'D1' or 'D2' → label 1 (endometriosis).
    All other patient IDs (e.g. healthy_pelvis) → label 0 (healthy).
    """

    if patient_id.startswith("D1") or patient_id.startswith("D2"):
        return 1

    return 0


def split_patients(
    patient_ids: list[str],
    labels: list[int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Split patient IDs into training and validation sets at the patient level.

    Uses stratified splitting to preserve class balance across both subsets.

    Returns:
        Dictionary with 'train' and 'validation' keys mapping to patient ID lists.
    """

    train_ids, val_ids = train_test_split(
        patient_ids,
        test_size=1 - train_ratio,
        stratify=labels,
        random_state=seed,
    )

    return {
        "train": train_ids,
        "validation": val_ids,
    }


def save_manifest(manifest_rows: list[dict], processed_dir: Path) -> None:
    """
    Save the preprocessing manifest as a CSV file.
    """

    df = pd.DataFrame(manifest_rows)

    csv_path = processed_dir / "manifest.csv"
    df.to_csv(csv_path, index=False)

    print(f"Manifest saved: {csv_path} ({len(df)} slices)")