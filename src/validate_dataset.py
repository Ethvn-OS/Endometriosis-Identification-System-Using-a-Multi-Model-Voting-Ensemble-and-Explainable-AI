from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import (
    VALID_SPLITS,
    assert_disjoint_patients,
    load_manifest,
    patient_normalized_weights,
)
from src.preprocess import load_config
from src.preprocessing.nifti import foreground_fractions, load_reoriented_nifti
from src.preprocessing.transforms import harmonize_slice


ORIGINAL_GEOMETRY_FEATURES = [
    "original_shape_x",
    "original_shape_y",
    "original_shape_z",
    "original_spacing_x",
    "original_spacing_y",
    "original_spacing_z",
]
MODEL_VISIBLE_GEOMETRY_FEATURES = [
    "effective_output_spacing_x",
    "effective_output_spacing_y",
    "harmonized_fov_x",
    "harmonized_fov_y",
    "output_height",
    "output_width",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_baseline(patient_df: pd.DataFrame, features: list[str]) -> dict:
    x = patient_df[features].to_numpy(dtype=np.float64)
    y = patient_df["label"].to_numpy(dtype=np.int32)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "features": features,
        "accuracy": float(accuracy_score(y, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def _write_qc_montage(df: pd.DataFrame, processed_dir: Path) -> Path:
    selected = []
    for source_site, site_df in df.groupby("source_site"):
        patients = (
            site_df.drop_duplicates("patient_id")
            .sort_values("original_spacing_x")
            .reset_index(drop=True)
        )
        for index in sorted({0, len(patients) // 2, len(patients) - 1}):
            patient_id = patients.iloc[index]["patient_id"]
            slices = site_df[site_df["patient_id"] == patient_id].sort_values(
                "slice_index"
            )
            selected.append((source_site, slices.iloc[len(slices) // 2]))

    figure, axes = plt.subplots(3, 3, figsize=(10, 10))
    for axis, (source_site, row) in zip(axes.flat, selected):
        image = np.load(processed_dir / row["filename"])
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(
            f"{source_site} | {row['patient_id']}\n"
            f"spacing={row['original_spacing_x']:.3f}mm"
        )
        axis.axis("off")
    figure.tight_layout()
    output = processed_dir / "qc_montage.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def _write_boundary_montage(
    summary: dict, manifest: pd.DataFrame, config: dict
) -> Path | None:
    candidates = [
        patient
        for patient in summary["patient_summary"]
        if patient["rejected_slice_indices"]
    ]
    if not candidates:
        return None
    candidates_frame = pd.DataFrame(candidates)
    boundary_rows = []
    for candidate in candidates:
        patient_rows = manifest[manifest["patient_id"] == candidate["patient_id"]]
        highest_rejected = max(
            candidate["rejected_slices"], key=lambda row: row["foreground_fraction"]
        )
        lowest_retained = patient_rows.loc[
            patient_rows["foreground_fraction"].idxmin()
        ]
        candidate = dict(candidate)
        candidate["highest_rejected_index"] = int(highest_rejected["slice_index"])
        candidate["highest_rejected_fraction"] = float(
            highest_rejected["foreground_fraction"]
        )
        candidate["lowest_retained_index"] = int(lowest_retained["slice_index"])
        candidate["lowest_retained_fraction"] = float(
            lowest_retained["foreground_fraction"]
        )
        candidate["boundary_gap"] = (
            candidate["lowest_retained_fraction"]
            - candidate["highest_rejected_fraction"]
        )
        boundary_rows.append(candidate)
    boundaries = pd.DataFrame(boundary_rows)
    rejected_patients = [
        site.sort_values(["boundary_gap", "patient_id"]).iloc[0].to_dict()
        for _, site in boundaries.groupby("source_site")
    ]

    project_root = Path(config["project_root"])
    preprocessing = config["preprocessing"]
    figure, axes = plt.subplots(
        len(rejected_patients), 2, figsize=(8, 4 * len(rejected_patients)), squeeze=False
    )
    for row_index, patient in enumerate(rejected_patients):
        patient_rows = manifest[manifest["patient_id"] == patient["patient_id"]]
        first_row = patient_rows.iloc[0]
        raw_path = project_root / first_row["raw_filename"]
        volume = load_reoriented_nifti(
            raw_path,
            tuple(preprocessing["canonical_orientation"]),
            target_inplane_spacing_mm=float(
                preprocessing["target_inplane_spacing_mm"]
            ),
        )
        fractions = foreground_fractions(
            volume.data, *map(float, preprocessing["foreground_percentiles"])
        )
        rejected_index = int(patient["highest_rejected_index"])
        retained_index = int(patient["lowest_retained_index"])
        for column, (slice_index, status) in enumerate(
            ((rejected_index, "highest rejected"), (retained_index, "lowest retained"))
        ):
            processed = harmonize_slice(
                volume.data[:, :, slice_index],
                source_spacing=volume.canonical_spacing[:2],
                target_spacing_mm=float(
                    preprocessing["target_inplane_spacing_mm"]
                ),
                target_fov_mm=float(preprocessing["target_fov_mm"]),
                output_size=tuple(preprocessing["image_size"]),
            )
            axis = axes[row_index, column]
            axis.imshow(processed, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(
                f"{patient['patient_id']} | {status}\n"
                f"slice={slice_index}, foreground={fractions[slice_index]:.3f}"
            )
            axis.axis("off")

    figure.tight_layout()
    output = Path(config["data"]["processed_dir"]) / "qc_boundary_montage.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def validate_dataset(config: dict, check_all_files: bool = True) -> dict:
    """Run hard training-readiness checks and save an auditable report."""

    processed_dir = Path(config["data"]["processed_dir"])
    project_root = Path(config["project_root"])
    manifest = load_manifest(processed_dir)
    if manifest.duplicated().any():
        raise ValueError("Manifest contains duplicate rows.")

    frames = {
        split: manifest[manifest["split"] == split].reset_index(drop=True)
        for split in VALID_SPLITS
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError("Train, validation, and test must all contain slices.")
    assert_disjoint_patients(frames)

    if set(manifest["sequence"]) != {"T2"}:
        raise ValueError(f"Unexpected sequences: {sorted(manifest['sequence'].unique())}")
    if set(manifest["canonical_orientation"]) != {"LPS"}:
        raise ValueError("Not every volume was resampled to orthogonal LPS.")
    obliquity_columns = [
        "canonical_obliquity_x_deg",
        "canonical_obliquity_y_deg",
        "canonical_obliquity_z_deg",
    ]
    if manifest[obliquity_columns].to_numpy(dtype=float).max() > 1e-4:
        raise ValueError("Canonical volumes are not on an orthogonal grid.")
    if not (manifest["foreground_fraction"] >= float(
        config["preprocessing"]["min_foreground_fraction"]
    )).all():
        raise ValueError("Manifest contains slices that fail foreground QC.")

    patient_df = manifest.drop_duplicates("patient_id").copy()
    expected = {
        site["name"]: int(site["expected_unique_volumes"])
        for site in config["datasets"]["ut_endomri"]["sites"]
    }
    expected[config["datasets"]["mogambo"]["source_site"]] = int(
        config["datasets"]["mogambo"]["expected_unique_volumes"]
    )
    actual = patient_df["source_site"].value_counts().to_dict()
    if actual != expected:
        raise ValueError(f"Patient counts do not match configured cohorts: {actual}")

    expected_semantics = {
        "UT-D1": (1, "confirmed_positive"),
        "UT-D2": (1, "confirmed_positive"),
        "MOGaMBO": (0, "proxy_comparator"),
    }
    for source_site, (label, role) in expected_semantics.items():
        site = patient_df[patient_df["source_site"] == source_site]
        if set(site["label"]) != {label} or set(site["cohort_role"]) != {role}:
            raise ValueError(f"Invalid label semantics for {source_site}.")

    patient_consistency = manifest.groupby("patient_id").agg(
        labels=("label", "nunique"),
        splits=("split", "nunique"),
        sequences=("sequence", "nunique"),
        source_hashes=("source_sha256", "nunique"),
        voxel_hashes=("voxel_sha256", "nunique"),
    )
    if (patient_consistency != 1).any().any():
        raise ValueError("At least one patient has inconsistent provenance or assignment.")
    if patient_df["source_sha256"].duplicated().any():
        raise ValueError("Exact duplicate source volumes appear under different patients.")
    if patient_df["voxel_sha256"].duplicated().any():
        raise ValueError("Exact duplicate voxel content appears under different patients.")

    deduplication = pd.read_csv(processed_dir / "deduplication_report.csv")
    expected_exclusions = sum(
        int(site["expected_patients_raw"]) - int(site["expected_unique_volumes"])
        for site in config["datasets"]["ut_endomri"]["sites"]
    ) + int(config["datasets"]["mogambo"]["expected_patients_raw"]) - int(
        config["datasets"]["mogambo"]["expected_unique_volumes"]
    )
    if len(deduplication) != expected_exclusions:
        raise ValueError("Deduplication report does not match configured source counts.")
    if set(deduplication["retained_patient_id"]) - set(patient_df["patient_id"]):
        raise ValueError("Deduplication report references an unretained canonical patient.")

    split_path = Path(config["data"]["patient_splits_file"])
    split_df = pd.read_csv(split_path)
    manifest_assignments = patient_df.set_index("patient_id")["split"].sort_index()
    saved_assignments = split_df.set_index("patient_id")["split"].sort_index()
    if not manifest_assignments.equals(saved_assignments):
        raise ValueError("Manifest assignments differ from patient_splits.csv.")

    target_fov = float(config["preprocessing"]["target_fov_mm"])
    output_h, output_w = map(int, config["preprocessing"]["image_size"])
    effective_x, effective_y = target_fov / output_h, target_fov / output_w
    constants = {
        "resampling_spacing_x": float(
            config["preprocessing"]["target_inplane_spacing_mm"]
        ),
        "resampling_spacing_y": float(
            config["preprocessing"]["target_inplane_spacing_mm"]
        ),
        "harmonized_fov_x": target_fov,
        "harmonized_fov_y": target_fov,
        "output_height": output_h,
        "output_width": output_w,
        "effective_output_spacing_x": effective_x,
        "effective_output_spacing_y": effective_y,
    }
    for column, expected_value in constants.items():
        if not np.allclose(manifest[column], expected_value):
            raise ValueError(f"Column {column} is not uniformly {expected_value}.")
    if not np.allclose(manifest["effective_output_spacing_x"] * output_h, target_fov):
        raise ValueError("Effective model spacing and field of view are inconsistent.")

    summary = json.loads(
        (processed_dir / "preprocessing_summary.json").read_text(encoding="utf-8")
    )
    fingerprint_config = json.loads(json.dumps(config))
    fingerprint_config.pop("project_root", None)
    fingerprint_config["split"].pop("regenerate", None)
    current_config_sha256 = hashlib.sha256(
        json.dumps(fingerprint_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if summary["config_sha256"] != current_config_sha256:
        raise ValueError("Current configuration does not match the processed dataset.")
    for relative, expected_hash in summary["source_file_sha256"].items():
        path = project_root / relative
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Source code changed after preprocessing: {relative}")
    current_source_files = {
        path.relative_to(project_root).as_posix()
        for path in sorted((project_root / "src").rglob("*.py"))
    } | {"requirements.txt"}
    if current_source_files != set(summary["source_file_sha256"]):
        raise ValueError("The source-file set changed after preprocessing.")
    patient_summary = pd.DataFrame(summary["patient_summary"])
    if not np.array_equal(
        patient_summary["raw_slices"],
        patient_summary["kept_slices"]
        + patient_summary["rejected_slice_indices"].map(len),
    ):
        raise ValueError("Raw/retained/rejected slice reconciliation failed.")
    kept_by_patient = manifest.groupby("patient_id").size()
    summary_kept = patient_summary.set_index("patient_id")["kept_slices"]
    if not kept_by_patient.sort_index().equals(summary_kept.sort_index()):
        raise ValueError("Manifest slice counts differ from preprocessing summary.")

    files_to_check = (
        manifest
        if check_all_files
        else manifest.iloc[:: max(len(manifest) // 100, 1)]
    )
    invalid_files = []
    image_size = tuple(config["preprocessing"]["image_size"])
    for row in files_to_check.itertuples(index=False):
        path = processed_dir / row.filename
        if not path.exists():
            invalid_files.append(f"missing:{row.filename}")
            continue
        image = np.load(path, mmap_mode="r")
        if image.shape != image_size:
            invalid_files.append(f"shape:{row.filename}:{image.shape}")
        elif image.dtype != np.float32:
            invalid_files.append(f"dtype:{row.filename}:{image.dtype}")
        elif not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            invalid_files.append(f"range:{row.filename}")
        elif hashlib.sha256(np.asarray(image).tobytes()).hexdigest() != row.processed_sha256:
            invalid_files.append(f"hash:{row.filename}")
    if invalid_files:
        raise ValueError(f"Invalid processed slices: {invalid_files[:10]}")

    raw_hash_failures = []
    for row in patient_df.itertuples(index=False):
        raw_relative = Path(row.raw_filename)
        if raw_relative.is_absolute() or ".." in raw_relative.parts:
            raw_hash_failures.append(f"unsafe:{row.raw_filename}")
            continue
        raw_path = (project_root / raw_relative).resolve()
        if project_root.resolve() not in raw_path.parents:
            raw_hash_failures.append(f"escape:{row.raw_filename}")
        elif not raw_path.exists() or _sha256_file(raw_path) != row.source_sha256:
            raw_hash_failures.append(f"hash:{row.raw_filename}")
    if raw_hash_failures:
        raise ValueError(f"Raw provenance failures: {raw_hash_failures[:10]}")

    train_weights = patient_normalized_weights(frames["train"])
    weighted = frames["train"][["patient_id", "label"]].copy()
    weighted["weight"] = train_weights
    patient_totals = weighted.groupby(["label", "patient_id"])["weight"].sum()
    class_totals = weighted.groupby("label")["weight"].sum()
    for _, totals in patient_totals.groupby(level=0):
        if not np.allclose(totals, totals.iloc[0], rtol=1e-5, atol=1e-5):
            raise ValueError("Patient-normalized weights are not equal within class.")
    if not np.allclose(class_totals, class_totals.iloc[0], rtol=1e-5, atol=1e-5):
        raise ValueError("Aggregate training weight differs between classes.")

    geometry = {
        "original": _geometry_baseline(patient_df, ORIGINAL_GEOMETRY_FEATURES),
        "harmonized_model_visible": _geometry_baseline(
            patient_df, MODEL_VISIBLE_GEOMETRY_FEATURES
        ),
    }
    (processed_dir / "geometry_baseline.json").write_text(
        json.dumps(geometry, indent=2), encoding="utf-8"
    )
    montage_path = _write_qc_montage(manifest, processed_dir)
    boundary_path = _write_boundary_montage(summary, manifest, config)

    if boundary_path is not None:
        boundary_hash = _sha256_file(boundary_path)
        review_path = Path(config["data"]["qc_review_file"])
        if not review_path.exists():
            raise ValueError(
                f"Technical QC review is required. Inspect {boundary_path} and "
                f"record approval in {review_path}."
            )
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if (
            review.get("approved") is not True
            or review.get("config_sha256") != summary["config_sha256"]
            or review.get("qc_boundary_montage_sha256") != boundary_hash
        ):
            raise ValueError("Technical QC approval is missing, stale, or mismatched.")
    else:
        boundary_hash = None
        review = {"approved": True, "review_scope": "No slices were rejected."}

    split_summary = {}
    for split, frame in frames.items():
        patients = frame.drop_duplicates("patient_id")
        split_summary[split] = {
            "patients": int(len(patients)),
            "positive_patients": int(patients["label"].sum()),
            "proxy_comparator_patients": int((patients["label"] == 0).sum()),
            "source_site_patients": patients["source_site"].value_counts().to_dict(),
            "slices": int(len(frame)),
        }

    patient_summary["rejected_slices"] = patient_summary[
        "rejected_slice_indices"
    ].map(len)
    rejection_by_site = (
        patient_summary.groupby("source_site")[["raw_slices", "kept_slices", "rejected_slices"]]
        .sum()
        .astype(int)
        .to_dict(orient="index")
    )
    report = {
        "ready_for_training": True,
        "patients": int(len(patient_df)),
        "slices": int(len(manifest)),
        "source_site_patients": actual,
        "excluded_exact_duplicate_volumes": int(len(deduplication)),
        "splits": split_summary,
        "checked_slice_files": int(len(files_to_check)),
        "verified_raw_volume_hashes": int(len(patient_df)),
        "patient_weight_total_min": float(patient_totals.min()),
        "patient_weight_total_max": float(patient_totals.max()),
        "effective_output_spacing_mm": [effective_x, effective_y],
        "geometry_baseline": geometry,
        "rejection_by_source": rejection_by_site,
        "qc_montage": montage_path.as_posix(),
        "qc_boundary_montage": boundary_path.as_posix() if boundary_path else None,
        "preprocessing_config_sha256": summary["config_sha256"],
        "validation_config_sha256": current_config_sha256,
        "technical_qc_review": review,
        "source_cohort_confounded_with_label": True,
    }
    release_inputs = {
        name: _sha256_file(processed_dir / name)
        for name in (
            "manifest.csv",
            "patient_splits.csv",
            "deduplication_report.csv",
            "preprocessing_summary.json",
            "geometry_baseline.json",
            "qc_montage.png",
            "qc_boundary_montage.png",
            "qc_review.json",
        )
        if (processed_dir / name).exists()
    }
    release_manifest = {
        "config_sha256": current_config_sha256,
        "source_file_sha256": summary["source_file_sha256"],
        "artifact_sha256": release_inputs,
    }
    release_path = processed_dir / "release_manifest.json"
    release_path.write_text(json.dumps(release_manifest, indent=2), encoding="utf-8")
    report["release_manifest_sha256"] = _sha256_file(release_path)
    (processed_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = validate_dataset(load_config(), check_all_files=True)
    print(json.dumps(result, indent=2))
