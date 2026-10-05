from __future__ import annotations

import re
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    source_patient_id: str
    cohort: str
    source_site: str
    cohort_role: str
    label: int
    sequence: str
    sequence_evidence: str
    label_evidence: str
    source_reference: str
    source_sha256: str
    voxel_sha256: str
    volume_path: Path


def _natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _voxel_sha256(path: Path) -> str:
    image = nib.load(path)
    data = np.ascontiguousarray(image.get_fdata(dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(np.asarray(data.shape, dtype=np.int64).tobytes())
    digest.update(data.tobytes())
    return digest.hexdigest()


def get_sequence_files(patient_dir: Path, sequences: list[str]) -> dict[str, Path]:
    """Find the requested UT-EndoMRI sequence files in a patient directory."""

    candidates: dict[str, list[Path]] = {sequence: [] for sequence in sequences}
    for file in sorted(patient_dir.iterdir()):
        if not file.is_file():
            continue
        is_nifti = file.name.endswith(".nii") or file.name.endswith(".nii.gz")
        for sequence in sequences:
            if is_nifti and f"_{sequence}." in file.name:
                candidates[sequence].append(file)

    found: dict[str, Path] = {}
    for sequence, files in candidates.items():
        if len(files) > 1:
            raise ValueError(
                f"Expected at most one {sequence} NIfTI in {patient_dir}, "
                f"found {len(files)}."
            )
        if files:
            found[sequence] = files[0]
    return found


def deduplicate_records(
    records: list[PatientRecord],
) -> tuple[list[PatientRecord], list[dict[str, str]]]:
    """Keep one deterministic representative for each exact source-volume hash."""

    retained_by_hash: dict[str, PatientRecord] = {}
    unique: list[PatientRecord] = []
    exclusions: list[dict[str, str]] = []
    for record in records:
        retained = retained_by_hash.get(record.voxel_sha256)
        if retained is None:
            retained_by_hash[record.voxel_sha256] = record
            unique.append(record)
            continue
        if retained.label != record.label:
            raise ValueError(
                "An exact source-volume duplicate appears in opposing classes: "
                f"{retained.patient_id}, {record.patient_id}."
            )
        exclusions.append(
            {
                "excluded_patient_id": record.patient_id,
                "retained_patient_id": retained.patient_id,
                "source_sha256": record.source_sha256,
                "voxel_sha256": record.voxel_sha256,
                "reason": "exact_duplicate_voxel_content",
            }
        )
    return unique, exclusions


def discover_patients(
    config: dict, deduplicate: bool = True
) -> list[PatientRecord]:
    """Discover the explicitly configured T2 development cohorts."""

    sequences = config["preprocessing"]["sequences"]
    if sequences != ["T2"]:
        raise ValueError(
            "The primary experiment must use only T2. "
            f"Configured sequences: {sequences}"
        )

    datasets = config["datasets"]
    if int(datasets["ut_endomri"]["label"]) != 1:
        raise ValueError("UT-EndoMRI must be configured as label 1.")
    if int(datasets["mogambo"]["label"]) != 0:
        raise ValueError("MOGaMBO must be configured as label 0.")
    if datasets["mogambo"]["cohort_role"] != "proxy_comparator":
        raise ValueError("MOGaMBO must be identified as a proxy comparator.")
    records: list[PatientRecord] = []

    ut = datasets["ut_endomri"]
    ut_root = Path(ut["path"])
    for site in ut["sites"]:
        site_dir = ut_root / site["directory"]
        if not site_dir.is_dir():
            raise FileNotFoundError(f"UT-EndoMRI site directory not found: {site_dir}")

        site_records: list[PatientRecord] = []
        for patient_dir in sorted(site_dir.iterdir(), key=_natural_key):
            if not patient_dir.is_dir():
                continue
            t2_path = get_sequence_files(patient_dir, [ut["sequence"]]).get("T2")
            if t2_path is None:
                continue
            source_id = patient_dir.name
            site_records.append(
                PatientRecord(
                    patient_id=f"UT-{source_id}",
                    source_patient_id=source_id,
                    cohort="ut_endomri",
                    source_site=site["name"],
                    cohort_role=ut["cohort_role"],
                    label=int(ut["label"]),
                    sequence="T2",
                    sequence_evidence=ut["sequence_evidence"],
                    label_evidence=ut["label_evidence"],
                    source_reference=ut["source_reference"],
                    source_sha256=_sha256_file(t2_path),
                    voxel_sha256=_voxel_sha256(t2_path),
                    volume_path=t2_path,
                )
            )

        expected = int(site["expected_patients_raw"])
        if len(site_records) != expected:
            raise ValueError(
                f"Expected {expected} T2 patients for {site['name']}, "
                f"found {len(site_records)}."
            )
        records.extend(site_records)

    mog = datasets["mogambo"]
    mog_root = Path(mog["path"])
    if not mog_root.is_dir():
        raise FileNotFoundError(f"MOGaMBO directory not found: {mog_root}")

    mog_records: list[PatientRecord] = []
    for patient_dir in sorted(mog_root.iterdir(), key=_natural_key):
        if not patient_dir.is_dir():
            continue
        candidates = sorted(
            file
            for file in patient_dir.glob("mriP*raw.nii*")
            if file.name.endswith(".nii") or file.name.endswith(".nii.gz")
        )
        if len(candidates) != 1:
            if candidates:
                raise ValueError(
                    f"Expected one raw MRI volume in {patient_dir}, found {len(candidates)}."
                )
            continue
        source_id = patient_dir.name
        mog_records.append(
            PatientRecord(
                patient_id=f"MOG-{source_id}",
                source_patient_id=source_id,
                cohort="mogambo",
                source_site=mog["source_site"],
                cohort_role=mog["cohort_role"],
                label=int(mog["label"]),
                sequence="T2",
                sequence_evidence=mog["sequence_evidence"],
                label_evidence=mog["label_evidence"],
                source_reference=mog["source_reference"],
                source_sha256=_sha256_file(candidates[0]),
                voxel_sha256=_voxel_sha256(candidates[0]),
                volume_path=candidates[0],
            )
        )

    expected_mog = int(mog["expected_patients_raw"])
    if len(mog_records) != expected_mog:
        raise ValueError(
            f"Expected {expected_mog} MOGaMBO patients, found {len(mog_records)}."
        )
    records.extend(mog_records)

    patient_ids = [record.patient_id for record in records]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Duplicate namespaced patient IDs were discovered.")
    if not deduplicate:
        return records

    unique, _ = deduplicate_records(records)
    unique_counts = pd.Series(
        [record.source_site for record in unique], dtype="object"
    ).value_counts()
    for site in ut["sites"]:
        if int(unique_counts.get(site["name"], 0)) != int(site["expected_unique_volumes"]):
            raise ValueError(f"Unexpected unique-volume count for {site['name']}.")
    if int(unique_counts.get(mog["source_site"], 0)) != int(
        mog["expected_unique_volumes"]
    ):
        raise ValueError("Unexpected unique-volume count for MOGaMBO.")
    return unique


def records_frame(records: list[PatientRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = asdict(record)
        row["volume_path"] = record.volume_path.as_posix()
        rows.append(row)
    return pd.DataFrame(rows)


def split_patients(
    patient_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Create a source-stratified 70/15/15 patient-level split."""

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}.")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Train, validation, and test ratios must all be positive.")

    train_df, remainder_df = train_test_split(
        patient_df,
        train_size=train_ratio,
        stratify=patient_df["source_site"],
        random_state=seed,
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        remainder_df,
        test_size=relative_test_ratio,
        stratify=remainder_df["source_site"],
        random_state=seed,
    )

    assignments = []
    for split_name, frame in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        current = frame.copy()
        current["split"] = split_name
        assignments.append(current)

    return (
        pd.concat(assignments, ignore_index=True)
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def _validate_patient_splits(
    assignments: pd.DataFrame, patient_df: pd.DataFrame, config: dict
) -> None:
    valid_splits = {"train", "validation", "test"}
    if set(assignments["split"].dropna().unique()) != valid_splits:
        raise ValueError(
            f"Patient split file must contain exactly {sorted(valid_splits)}."
        )
    if assignments["patient_id"].duplicated().any():
        raise ValueError("Patient split file contains duplicate patient IDs.")
    if set(assignments["patient_id"]) != set(patient_df["patient_id"]):
        raise ValueError("Patient split file does not contain the exact discovered cohort.")
    if not set(assignments["label"]).issubset({0, 1}):
        raise ValueError("Patient labels must be binary values 0 and 1.")
    if set(assignments["label"]) != {0, 1}:
        raise ValueError("Both patient classes must be present.")

    split_cfg = config["split"]
    expected = split_patients(
        patient_df,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["seed"]),
    )
    actual_map = assignments.set_index("patient_id")["split"].sort_index()
    expected_map = expected.set_index("patient_id")["split"].sort_index()
    if not actual_map.equals(expected_map):
        raise ValueError(
            "Persisted patient assignments do not match the configured stratified split."
        )


def load_or_create_patient_splits(
    records: list[PatientRecord], config: dict
) -> pd.DataFrame:
    """Reuse a validated patient split, or create and persist it once."""

    patient_df = records_frame(records)
    split_path = Path(config["data"]["patient_splits_file"])
    split_cfg = config["split"]

    if split_path.exists() and not split_cfg.get("regenerate", False):
        saved = pd.read_csv(split_path)
        required = set(patient_df.columns) | {"split"}
        missing = required - set(saved.columns)
        if missing:
            raise ValueError(f"Patient split file is missing columns: {sorted(missing)}")

        compare_cols = sorted(set(patient_df.columns) - {"volume_path"})
        current = patient_df[compare_cols].sort_values("patient_id").reset_index(drop=True)
        existing = saved[compare_cols].sort_values("patient_id").reset_index(drop=True)
        if not current.equals(existing):
            raise ValueError(
                "The discovered cohort no longer matches patient_splits.csv. "
                "Review the dataset change, then set split.regenerate=true once."
            )
        _validate_patient_splits(saved, patient_df, config)
        return saved.sort_values("patient_id").reset_index(drop=True)

    assignments = split_patients(
        patient_df,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["seed"]),
    )
    _validate_patient_splits(assignments, patient_df, config)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(split_path, index=False)
    print(f"Patient splits saved: {split_path} ({len(assignments)} patients)")
    return assignments


def save_manifest(manifest_rows: list[dict], processed_dir: Path) -> None:
    """Save the slice-level preprocessing manifest."""

    df = pd.DataFrame(manifest_rows)
    csv_path = processed_dir / "manifest.csv"
    df.to_csv(csv_path, index=False)
    print(f"Manifest saved: {csv_path} ({len(df)} slices)")
