from __future__ import annotations

import json
import hashlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (
    PatientRecord,
    deduplicate_records,
    discover_patients,
    load_or_create_patient_splits,
    save_manifest,
)
from src.preprocessing.nifti import foreground_fractions, load_reoriented_nifti
from src.preprocessing.transforms import harmonize_slice


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load the project configuration from YAML."""

    config_path = Path(config_path)
    if not config_path.is_absolute():
        default_path = Path(__file__).resolve().parents[1] / config_path
        config_path = default_path if default_path.exists() else config_path.resolve()
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    project_root = config_path.parent.parent
    config["project_root"] = str(project_root)
    for key in ("raw_dir", "processed_dir", "patient_splits_file", "qc_review_file"):
        config["data"][key] = str((project_root / config["data"][key]).resolve())
    for dataset in config["datasets"].values():
        dataset["path"] = str((project_root / dataset["path"]).resolve())
    config["training"]["output_dir"] = str(
        (project_root / config["training"]["output_dir"]).resolve()
    )
    return config


def _clear_stale_files(processed_dir: Path) -> None:
    """Remove regenerated slices and reports while preserving patient splits."""

    for split_dir in ("train", "validation", "test"):
        target = processed_dir / split_dir
        if target.is_dir():
            shutil.rmtree(target)

    for filename in (
        "manifest.csv",
        "preprocessing_summary.json",
        "validation_report.json",
        "qc_montage.png",
        "geometry_baseline.json",
        "qc_boundary_montage.png",
        "deduplication_report.csv",
        "release_manifest.json",
        "qc_review.json",
    ):
        target = processed_dir / filename
        if target.exists():
            target.unlink()
    print("Cleared previous generated output; retained patient_splits.csv.")


def _record_lookup(records: list[PatientRecord]) -> dict[str, PatientRecord]:
    return {record.patient_id: record for record in records}


def run_pipeline(config: dict) -> None:
    """Build the harmonized, T2-only development dataset and manifest."""

    processed_dir = Path(config["data"]["processed_dir"])
    project_root = Path(config["project_root"])
    preprocessing = config["preprocessing"]
    image_size = tuple(int(value) for value in preprocessing["image_size"])
    target_orientation = tuple(preprocessing["canonical_orientation"])
    target_spacing = float(preprocessing["target_inplane_spacing_mm"])
    target_fov = float(preprocessing["target_fov_mm"])
    low_percentile, high_percentile = map(
        float, preprocessing["foreground_percentiles"]
    )
    min_foreground = float(preprocessing["min_foreground_fraction"])

    print("Discovering explicit T2 cohorts...")
    raw_records = discover_patients(config, deduplicate=False)
    records, duplicate_exclusions = deduplicate_records(raw_records)
    unique_counts = Counter(record.source_site for record in records)
    for site in config["datasets"]["ut_endomri"]["sites"]:
        if unique_counts[site["name"]] != int(site["expected_unique_volumes"]):
            raise ValueError(f"Unexpected unique-volume count for {site['name']}.")
    mog_config = config["datasets"]["mogambo"]
    if unique_counts[mog_config["source_site"]] != int(
        mog_config["expected_unique_volumes"]
    ):
        raise ValueError("Unexpected unique-volume count for MOGaMBO.")
    assignments = load_or_create_patient_splits(records, config)
    records_by_id = _record_lookup(records)
    split_by_patient = dict(zip(assignments["patient_id"], assignments["split"]))

    print(
        f"Found {len(raw_records)} source folders; retained {len(records)} unique volumes "
        f"after excluding {len(duplicate_exclusions)} exact duplicates."
    )
    for source_site, count in Counter(record.source_site for record in records).items():
        print(f"  {source_site}: {count}")
    for split_name, group in assignments.groupby("split"):
        print(
            f"  {split_name}: {len(group)} patients "
            f"({int(group['label'].sum())} positive, "
            f"{int((group['label'] == 0).sum())} proxy comparator)"
        )

    _clear_stale_files(processed_dir)
    import pandas as pd

    pd.DataFrame(duplicate_exclusions).to_csv(
        processed_dir / "deduplication_report.csv", index=False
    )
    manifest_rows: list[dict] = []
    patient_summary: list[dict] = []

    for patient_id in assignments.sort_values(["split", "patient_id"])["patient_id"]:
        record = records_by_id[patient_id]
        split_name = split_by_patient[patient_id]
        print(f"Processing {split_name}/{patient_id}...")

        volume = load_reoriented_nifti(
            record.volume_path,
            target_orientation,
            target_inplane_spacing_mm=target_spacing,
        )
        fractions = foreground_fractions(
            volume.data,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )
        patient_out_dir = processed_dir / split_name / patient_id
        patient_out_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        rejected_indices: list[int] = []
        rejected_slices: list[dict] = []
        for slice_index in range(volume.data.shape[2]):
            foreground = float(fractions[slice_index])
            if foreground < min_foreground:
                rejected_indices.append(slice_index)
                rejected_slices.append(
                    {"slice_index": slice_index, "foreground_fraction": foreground}
                )
                continue

            processed = harmonize_slice(
                volume.data[:, :, slice_index],
                source_spacing=volume.canonical_spacing[:2],
                target_spacing_mm=target_spacing,
                target_fov_mm=target_fov,
                output_size=image_size,
            )
            filename = f"T2_slice_{slice_index:04d}.npy"
            out_path = patient_out_dir / filename
            np.save(out_path, processed)
            processed_sha256 = hashlib.sha256(processed.tobytes()).hexdigest()

            manifest_rows.append(
                {
                    "filename": out_path.relative_to(processed_dir).as_posix(),
                    "patient_id": record.patient_id,
                    "source_patient_id": record.source_patient_id,
                    "cohort": record.cohort,
                    "source_site": record.source_site,
                    "cohort_role": record.cohort_role,
                    "label_basis": (
                        "diagnosed_endometriosis_cohort"
                        if record.label == 1
                        else "mogambo_proxy_comparator_membership"
                    ),
                    "sequence_evidence": record.sequence_evidence,
                    "label_evidence": record.label_evidence,
                    "source_reference": record.source_reference,
                    "source_sha256": record.source_sha256,
                    "voxel_sha256": record.voxel_sha256,
                    "processed_sha256": processed_sha256,
                    "volume_id": record.volume_path.name,
                    "raw_filename": record.volume_path.relative_to(project_root).as_posix(),
                    "slice_index": slice_index,
                    "foreground_fraction": foreground,
                    "sequence": record.sequence,
                    "split": split_name,
                    "label": record.label,
                    "original_shape_x": volume.original_shape[0],
                    "original_shape_y": volume.original_shape[1],
                    "original_shape_z": volume.original_shape[2],
                    "original_spacing_x": volume.original_spacing[0],
                    "original_spacing_y": volume.original_spacing[1],
                    "original_spacing_z": volume.original_spacing[2],
                    "original_orientation": "".join(volume.original_orientation),
                    "original_obliquity_x_deg": volume.original_obliquity_degrees[0],
                    "original_obliquity_y_deg": volume.original_obliquity_degrees[1],
                    "original_obliquity_z_deg": volume.original_obliquity_degrees[2],
                    "canonical_orientation": "".join(volume.canonical_orientation),
                    "canonical_obliquity_x_deg": volume.canonical_obliquity_degrees[0],
                    "canonical_obliquity_y_deg": volume.canonical_obliquity_degrees[1],
                    "canonical_obliquity_z_deg": volume.canonical_obliquity_degrees[2],
                    "resampling_spacing_x": target_spacing,
                    "resampling_spacing_y": target_spacing,
                    "harmonized_fov_x": target_fov,
                    "harmonized_fov_y": target_fov,
                    "output_height": image_size[0],
                    "output_width": image_size[1],
                    "effective_output_spacing_x": target_fov / image_size[0],
                    "effective_output_spacing_y": target_fov / image_size[1],
                }
            )
            kept += 1

        if kept == 0:
            raise ValueError(f"Foreground QC removed every slice for {patient_id}.")
        patient_summary.append(
            {
                "patient_id": patient_id,
                "split": split_name,
                "raw_slices": int(volume.data.shape[2]),
                "kept_slices": kept,
                "rejected_slice_indices": rejected_indices,
                "rejected_slices": rejected_slices,
                "source_site": record.source_site,
            }
        )
        print(f"  kept {kept}/{volume.data.shape[2]} slices")

    save_manifest(manifest_rows, processed_dir)
    fingerprint_config = json.loads(json.dumps(config))
    fingerprint_config.pop("project_root", None)
    fingerprint_config["split"].pop("regenerate", None)
    config_for_hash = json.dumps(fingerprint_config, sort_keys=True).encode("utf-8")
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project_root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = "unavailable", None

    source_files = sorted((project_root / "src").rglob("*.py")) + [
        project_root / "requirements.txt",
    ]
    source_file_sha256 = {
        path.relative_to(project_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    try:
        git_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=project_root
        )
        git_diff_sha256 = hashlib.sha256(git_diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        git_diff_sha256 = "unavailable"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": 1,
        "config_sha256": hashlib.sha256(config_for_hash).hexdigest(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256,
        "source_file_sha256": source_file_sha256,
        "python_version": platform.python_version(),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "nibabel", "opencv-python", "scikit-learn", "tensorflow", "keras")
        },
        "patients": len(records),
        "raw_patient_folders": len(raw_records),
        "excluded_exact_duplicates": len(duplicate_exclusions),
        "slices": len(manifest_rows),
        "sequences": ["T2"],
        "canonical_orientation": list(target_orientation),
        "target_inplane_spacing_mm": target_spacing,
        "target_fov_mm": target_fov,
        "output_size": list(image_size),
        "min_foreground_fraction": min_foreground,
        "patient_summary": patient_summary,
    }
    (processed_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Preprocessing complete: {len(records)} patients, {len(manifest_rows)} slices.")


if __name__ == "__main__":
    run_pipeline(load_config())
