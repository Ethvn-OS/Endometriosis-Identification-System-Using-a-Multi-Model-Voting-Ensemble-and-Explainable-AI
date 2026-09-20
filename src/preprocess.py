import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (
    assign_label,
    discover_patients,
    save_manifest,
    split_patients,
)
from src.preprocessing.nifti import extract_axial_slices, load_nifti
from src.preprocessing.transforms import preprocess_slice


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """
    Load the project configuration from a YAML file.
    """

    config_path = Path(config_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict) -> None:
    """
    Execute the full preprocessing pipeline:

    1. Discover patients and their available MRI sequences
    2. Assign binary labels (endometriosis vs. healthy)
    3. Split patients into train/validation at the patient level
    4. Load NIfTI volumes, extract axial slices, resize, normalize, save
    5. Generate manifest CSV
    """

    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    sequences = config["preprocessing"]["sequences"]
    image_size = tuple(config["preprocessing"]["image_size"])
    train_ratio = config["split"]["train_ratio"]
    seed = config["split"]["seed"]

    # Step 1: Discover patients
    print("Discovering patients...")
    patients = discover_patients(raw_dir, sequences)

    if not patients:
        print("No patients found. Ensure datasets are placed in data/raw/.")
        return

    print(f"Found {len(patients)} patients.")

    # Step 2: Assign labels
    patient_ids = sorted(patients.keys())
    labels = [assign_label(pid) for pid in patient_ids]

    positive = sum(labels)
    negative = len(labels) - positive
    print(f"Class distribution: {positive} endometriosis, {negative} healthy.")

    # Step 3: Split patients
    print("Splitting patients...")
    splits = split_patients(patient_ids, labels, train_ratio, seed)

    for split_name, split_ids in splits.items():
        split_labels = [assign_label(pid) for pid in split_ids]
        pos = sum(split_labels)
        neg = len(split_labels) - pos
        print(f"  {split_name}: {len(split_ids)} patients ({pos} positive, {neg} negative)")

    # Step 4: Process each patient
    manifest_rows: list[dict] = []
    total_slices = 0

    for split_name, split_ids in splits.items():
        print(f"\nProcessing {split_name} split...")

        for patient_id in split_ids:
            label = assign_label(patient_id)
            seq_files = patients[patient_id]
            patient_out_dir = processed_dir / split_name / patient_id
            patient_out_dir.mkdir(parents=True, exist_ok=True)

            for seq_file in seq_files:
                seq_name = next(
                    seq for seq in sequences if f"_{seq}." in seq_file.name
                )

                try:
                    volume = load_nifti(seq_file)
                    slices = extract_axial_slices(volume)
                except Exception as e:
                    print(f"  Skipping {patient_id}/{seq_name}: {e}")
                    continue

                for i, slc in enumerate(slices):
                    processed = preprocess_slice(slc, image_size)

                    filename = f"{seq_name}_slice_{i:04d}.npy"
                    out_path = patient_out_dir / filename
                    np.save(out_path, processed.astype(np.float32))

                    manifest_rows.append({
                        "filename": str(out_path.relative_to(processed_dir)),
                        "patient_id": patient_id,
                        "sequence": seq_name,
                        "split": split_name,
                        "label": label,
                    })

                    total_slices += 1

            print(f"  {patient_id}: processed {len(seq_files)} sequences")

    # Step 5: Save manifest
    print(f"\nTotal slices processed: {total_slices}")
    save_manifest(manifest_rows, processed_dir)
    print("Preprocessing pipeline complete.")


if __name__ == "__main__":
    config = load_config()
    run_pipeline(config)
