from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import keras
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import build_dataset, load_manifest
from src.evaluation import aggregate_patient_predictions
from src.modeling import MODEL_NAMES, soft_vote
from src.preprocess import load_config
from src.validate_dataset import validate_dataset


THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 2027
ENSEMBLE_NAME = "SoftVotingEnsemble"
EXPECTED_SPLITS = {
    "validation": {
        "slices": 1_063,
        "patients": 22,
        "positive_patients": 13,
        "proxy_comparator_patients": 9,
    },
    "test": {
        "slices": 1_287,
        "patients": 23,
        "positive_patients": 14,
        "proxy_comparator_patients": 9,
    },
}
METRIC_KEYS = (
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "brier_score",
)
VALIDATION_LOCK_KEYS = (
    "threshold",
    "bootstrap_iterations",
    "bootstrap_seed",
    "model_names",
    "voting",
    "voting_weights",
    "git_commit",
    "package_versions",
    "evaluation_script_sha256",
    "training_config_sha256",
    "manifest_sha256",
    "release_manifest_sha256",
    "checkpoint_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_request(split: str, confirm_final_test: bool) -> None:
    if split not in EXPECTED_SPLITS:
        raise ValueError(f"Evaluation split must be one of {sorted(EXPECTED_SPLITS)}.")
    if split == "test" and not confirm_final_test:
        raise ValueError(
            "Test evaluation is locked. Re-run with --confirm-final-test only after "
            "the validation evaluation and methodology are finalized."
        )


def validate_manifest_counts(manifest: pd.DataFrame, split: str) -> None:
    expected = EXPECTED_SPLITS[split]
    patients = manifest.drop_duplicates("patient_id")
    actual = {
        "slices": int(len(manifest)),
        "patients": int(len(patients)),
        "positive_patients": int(patients["label"].sum()),
        "proxy_comparator_patients": int((patients["label"] == 0).sum()),
    }
    if actual != expected:
        raise ValueError(
            f"Unexpected {split} cohort counts. Expected {expected}, found {actual}."
        )


def align_patient_predictions(
    predictions_by_model: dict[str, pd.DataFrame], threshold: float = THRESHOLD
) -> pd.DataFrame:
    if tuple(predictions_by_model) != tuple(MODEL_NAMES):
        raise ValueError(
            f"Expected predictions in model order {MODEL_NAMES}, "
            f"found {tuple(predictions_by_model)}."
        )

    aligned: pd.DataFrame | None = None
    for model_name in MODEL_NAMES:
        frame = predictions_by_model[model_name].copy()
        required = {"patient_id", "label", "source_site", "probability", "slice_count"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{model_name} patient predictions are missing {sorted(missing)}."
            )
        if frame["patient_id"].duplicated().any():
            raise ValueError(f"{model_name} contains duplicate patient predictions.")

        current = frame[list(required)].rename(
            columns={
                "label": f"{model_name}_label",
                "source_site": f"{model_name}_source_site",
                "probability": f"{model_name}_probability",
                "slice_count": f"{model_name}_slice_count",
            }
        )
        aligned = (
            current
            if aligned is None
            else aligned.merge(current, on="patient_id", how="outer", validate="one_to_one")
        )

    assert aligned is not None
    if aligned.isna().any().any():
        raise ValueError("Models did not produce predictions for identical patient IDs.")

    first = MODEL_NAMES[0]
    aligned["label"] = aligned[f"{first}_label"].astype(int)
    aligned["source_site"] = aligned[f"{first}_source_site"].astype(str)
    aligned["slice_count"] = aligned[f"{first}_slice_count"].astype(int)
    for model_name in MODEL_NAMES[1:]:
        if not np.array_equal(
            aligned["label"].to_numpy(),
            aligned[f"{model_name}_label"].to_numpy(dtype=int),
        ):
            raise ValueError(f"Patient labels do not align for {model_name}.")
        if not np.array_equal(
            aligned["source_site"].to_numpy(),
            aligned[f"{model_name}_source_site"].astype(str).to_numpy(),
        ):
            raise ValueError(f"Patient source sites do not align for {model_name}.")
        if not np.array_equal(
            aligned["slice_count"].to_numpy(),
            aligned[f"{model_name}_slice_count"].to_numpy(dtype=int),
        ):
            raise ValueError(f"Patient slice counts do not align for {model_name}.")

    probability_columns = [f"{name}_probability" for name in MODEL_NAMES]
    probabilities = aligned[probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("Patient probabilities contain non-finite values.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Patient probabilities must be in [0, 1].")

    ensemble = np.asarray(
        soft_vote([aligned[column].to_numpy() for column in probability_columns])
    ).reshape(-1)
    aligned[f"{ENSEMBLE_NAME}_probability"] = ensemble
    for name in (*MODEL_NAMES, ENSEMBLE_NAME):
        aligned[f"{name}_prediction"] = (
            aligned[f"{name}_probability"] >= threshold
        ).astype(int)

    internal_columns = [
        column
        for column in aligned.columns
        if column.endswith(("_label", "_source_site", "_slice_count"))
    ]
    return (
        aligned.drop(columns=internal_columns)
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def compute_binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = THRESHOLD
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("Labels and scores must be non-empty arrays of equal length.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both binary classes are required for evaluation.")
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Scores must be finite probabilities in [0, 1].")

    predicted = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp))
    return {
        "patients": int(labels.size),
        "positive_patients": int(labels.sum()),
        "proxy_comparator_patients": int((labels == 0).sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": specificity,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def stratified_bootstrap_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = THRESHOLD,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive.")
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Both classes are required for stratified bootstrap intervals.")

    rng = np.random.default_rng(seed)
    samples = {key: [] for key in METRIC_KEYS}
    for _ in range(iterations):
        sampled = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        metrics = compute_binary_metrics(labels[sampled], scores[sampled], threshold)
        for key in METRIC_KEYS:
            samples[key].append(float(metrics[key]))

    return {
        key: {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
        }
        for key, values in samples.items()
    }


def evaluate_patient_predictions(
    patients: pd.DataFrame,
    threshold: float = THRESHOLD,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, dict]:
    labels = patients["label"].to_numpy(dtype=np.int64)
    results = {}
    for name in (*MODEL_NAMES, ENSEMBLE_NAME):
        scores = patients[f"{name}_probability"].to_numpy(dtype=np.float64)
        results[name] = {
            "point_estimate": compute_binary_metrics(labels, scores, threshold),
            "bootstrap_95_ci": stratified_bootstrap_intervals(
                labels,
                scores,
                threshold=threshold,
                iterations=bootstrap_iterations,
                seed=BOOTSTRAP_SEED,
            ),
        }
    return results


def ensure_output_available(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Evaluation output directory is not empty: {path}. "
            "Preserve the existing results rather than overwriting them."
        )


def validate_validation_lock(prior: dict, current: dict) -> None:
    if prior.get("split") != "validation":
        raise ValueError("The prerequisite provenance is not a validation run.")
    mismatched = [
        key for key in VALIDATION_LOCK_KEYS if prior.get(key) != current.get(key)
    ]
    if mismatched:
        raise ValueError(
            "The finalized validation methodology or artifacts changed before test "
            f"evaluation: {mismatched}. Re-run and review validation first."
        )


def _plot_roc_curves(patients: pd.DataFrame, output_path: Path) -> None:
    labels = patients["label"].to_numpy(dtype=np.int64)
    figure, axis = plt.subplots(figsize=(7, 6))
    for name in (*MODEL_NAMES, ENSEMBLE_NAME):
        scores = patients[f"{name}_probability"].to_numpy(dtype=np.float64)
        false_positive, true_positive, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        axis.plot(false_positive, true_positive, label=f"{name} (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Patient-level ROC curves",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_confusion_matrices(patients: pd.DataFrame, output_path: Path) -> None:
    labels = patients["label"].to_numpy(dtype=np.int64)
    figure, axes = plt.subplots(2, 2, figsize=(9, 8))
    for axis, name in zip(axes.flat, (*MODEL_NAMES, ENSEMBLE_NAME)):
        predicted = patients[f"{name}_prediction"].to_numpy(dtype=np.int64)
        matrix = confusion_matrix(labels, predicted, labels=[0, 1])
        axis.imshow(matrix, cmap="Blues", vmin=0)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(
            title=name,
            xlabel="Predicted label",
            ylabel="True label",
            xticks=[0, 1],
            yticks=[0, 1],
        )
    figure.suptitle(f"Patient-level confusion matrices (threshold={THRESHOLD:.1f})")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "tensorflow", "keras"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _git_value(arguments: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _provenance(
    split: str,
    config: dict,
    checkpoints: dict[str, Path],
    manifest_path: Path,
    training_config_path: Path,
) -> dict:
    release_manifest_path = Path(config["data"]["processed_dir"]) / "release_manifest.json"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "threshold": THRESHOLD,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "model_names": list(MODEL_NAMES),
        "voting": "equal_weight_patient_probability_mean",
        "voting_weights": {name: 1.0 / len(MODEL_NAMES) for name in MODEL_NAMES},
        "source_cohort_confounded_with_label": True,
        "proxy_comparator_is_not_confirmed_endometriosis_negative": True,
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_status": _git_value(["status", "--porcelain"]),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "training_config_sha256": sha256_file(training_config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "release_manifest_sha256": (
            sha256_file(release_manifest_path)
            if release_manifest_path.exists()
            else "missing"
        ),
        "checkpoint_sha256": {
            name: sha256_file(path) for name, path in checkpoints.items()
        },
    }


def _load_training_contract(config: dict) -> tuple[Path, dict[str, Path]]:
    training_dir = Path(config["training"]["output_dir"])
    training_config_path = training_dir / "training_config.json"
    if not training_config_path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {training_config_path}")
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))

    configured_models = tuple(training_config["training"]["model_names"])
    if configured_models != MODEL_NAMES:
        raise ValueError(
            f"Training configuration model order must be {MODEL_NAMES}, "
            f"found {configured_models}."
        )
    if tuple(config["training"]["model_names"]) != MODEL_NAMES:
        raise ValueError("Current configuration does not contain the required model order.")

    checkpoints = {
        name: training_dir / name / "best.keras" for name in MODEL_NAMES
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing trained checkpoint(s): {missing}")
    if any(path.stat().st_size == 0 for path in checkpoints.values()):
        raise ValueError("A trained checkpoint is empty.")
    return training_config_path, checkpoints


def run_evaluation(config_path: str | Path, split: str, confirm_final_test: bool) -> Path:
    validate_split_request(split, confirm_final_test)
    config = load_config(config_path)
    validate_dataset(config, check_all_files=True)

    output_root = PROJECT_ROOT / "outputs" / "evaluation"
    output_dir = output_root / split
    ensure_output_available(output_dir)

    training_config_path, checkpoints = _load_training_contract(config)
    processed_dir = Path(config["data"]["processed_dir"])
    manifest_path = processed_dir / "manifest.csv"
    provenance = _provenance(
        split, config, checkpoints, manifest_path, training_config_path
    )
    if split == "test":
        validation_dir = output_root / "validation"
        validation_metrics_path = validation_dir / "metrics.json"
        validation_provenance_path = validation_dir / "provenance.json"
        if not validation_metrics_path.is_file() or not validation_provenance_path.is_file():
            raise FileNotFoundError(
                "A completed validation evaluation is required before test evaluation."
            )
        prior = json.loads(validation_provenance_path.read_text(encoding="utf-8"))
        validate_validation_lock(prior, provenance)

    manifest = load_manifest(processed_dir, split=split)
    validate_manifest_counts(manifest, split)
    dataset = build_dataset(config, manifest, augment=False, shuffle=False)

    slice_predictions = manifest.copy()
    patient_predictions = {}
    for model_name in MODEL_NAMES:
        print(f"Loading {model_name} checkpoint...")
        keras.backend.clear_session()
        model = keras.models.load_model(checkpoints[model_name], compile=False)
        probabilities = np.asarray(model.predict(dataset, verbose=1)).reshape(-1)
        if len(probabilities) != len(manifest):
            raise ValueError(
                f"{model_name} produced {len(probabilities)} probabilities for "
                f"{len(manifest)} slices."
            )
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0.0) | (probabilities > 1.0)
        ).any():
            raise ValueError(f"{model_name} produced invalid probabilities.")
        slice_predictions[f"{model_name}_probability"] = probabilities
        patient_predictions[model_name] = aggregate_patient_predictions(
            manifest, probabilities
        )
        del model
        keras.backend.clear_session()
        gc.collect()

    probability_columns = [f"{name}_probability" for name in MODEL_NAMES]
    slice_predictions[f"{ENSEMBLE_NAME}_probability"] = np.asarray(
        soft_vote(
            [slice_predictions[column].to_numpy() for column in probability_columns]
        )
    ).reshape(-1)
    patients = align_patient_predictions(patient_predictions, threshold=THRESHOLD)
    pooled_ensemble = aggregate_patient_predictions(
        manifest, slice_predictions[f"{ENSEMBLE_NAME}_probability"].to_numpy()
    )
    expected_ensemble = patients.set_index("patient_id")[
        f"{ENSEMBLE_NAME}_probability"
    ].sort_index()
    actual_ensemble = pooled_ensemble.set_index("patient_id")["probability"].sort_index()
    if not np.allclose(expected_ensemble, actual_ensemble, rtol=1e-7, atol=1e-9):
        raise ValueError("Slice-level and patient-level equal soft voting disagree.")

    for name in (*MODEL_NAMES, ENSEMBLE_NAME):
        slice_predictions[f"{name}_prediction"] = (
            slice_predictions[f"{name}_probability"] >= THRESHOLD
        ).astype(int)

    metrics = {
        "split": split,
        "threshold": THRESHOLD,
        "patients": int(len(patients)),
        "positive_patients": int(patients["label"].sum()),
        "proxy_comparator_patients": int((patients["label"] == 0).sum()),
        "models": evaluate_patient_predictions(patients),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    patients.to_csv(output_dir / "patient_predictions.csv", index=False)
    slice_predictions.to_csv(output_dir / "slice_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    _plot_roc_curves(patients, output_dir / "roc_curve.png")
    _plot_confusion_matrices(patients, output_dir / "confusion_matrices.png")

    ensemble = metrics["models"][ENSEMBLE_NAME]["point_estimate"]
    print(
        f"{split.title()} evaluation complete: {len(patients)} patients, "
        f"ensemble AUC={ensemble['roc_auc']:.4f}, "
        f"accuracy={ensemble['accuracy']:.4f}."
    )
    print(f"Artifacts saved to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models at the patient level with guarded test access."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--split", choices=tuple(EXPECTED_SPLITS), default="validation"
    )
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required to unlock the one-time internal test evaluation.",
    )
    args = parser.parse_args()
    run_evaluation(args.config, args.split, args.confirm_final_test)


if __name__ == "__main__":
    main()
