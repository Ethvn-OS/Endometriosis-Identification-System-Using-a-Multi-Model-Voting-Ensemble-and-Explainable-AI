from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import keras
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_models import (
    ENSEMBLE_NAME,
    THRESHOLD,
    ensure_output_available,
    sha256_file,
)
from src.modeling import MODEL_NAMES
from src.preprocess import load_config
from src.preprocessing.slices import to_three_channels
from src.validate_dataset import validate_dataset


EXPECTED_TARGET_LAYERS = {
    "ResNet50": "conv5_block3_out",
    "EfficientNetV2S": "top_activation",
    "DenseNet121": "relu",
}
EXPECTED_TEST = {
    "slices": 1_287,
    "patients": 23,
    "positive_patients": 14,
    "proxy_comparator_patients": 9,
}
CONFUSION_ORDER = {"TP": 0, "TN": 1, "FP": 2, "FN": 3}
MAX_PROBABILITY_DRIFT = 0.005


def validate_request(split: str, confirm_final_test_explanations: bool) -> None:
    if split != "test":
        raise ValueError("Trained-model Grad-CAM is locked to the finalized test split.")
    if not confirm_final_test_explanations:
        raise ValueError(
            "Final test explanations are locked. Re-run with "
            "--confirm-final-test-explanations after preserving final evaluation outputs."
        )


def confusion_category(label: int, prediction: int) -> str:
    categories = {
        (1, 1): "TP",
        (0, 0): "TN",
        (0, 1): "FP",
        (1, 0): "FN",
    }
    try:
        return categories[(int(label), int(prediction))]
    except KeyError as error:
        raise ValueError("Labels and predictions must be binary.") from error


def select_representative_slices(
    patient_predictions: pd.DataFrame, slice_predictions: pd.DataFrame
) -> pd.DataFrame:
    patient_required = {
        "patient_id",
        "label",
        "source_site",
        "slice_count",
        f"{ENSEMBLE_NAME}_probability",
        f"{ENSEMBLE_NAME}_prediction",
    }
    slice_required = {
        "filename",
        "patient_id",
        "label",
        "source_site",
        f"{ENSEMBLE_NAME}_probability",
        f"{ENSEMBLE_NAME}_prediction",
    }
    for name in MODEL_NAMES:
        patient_required.update({f"{name}_probability", f"{name}_prediction"})
        slice_required.update({f"{name}_probability", f"{name}_prediction"})
    patient_missing = patient_required - set(patient_predictions.columns)
    slice_missing = slice_required - set(slice_predictions.columns)
    if patient_missing:
        raise ValueError(
            f"Patient predictions are missing columns: {sorted(patient_missing)}"
        )
    if slice_missing:
        raise ValueError(
            f"Slice predictions are missing columns: {sorted(slice_missing)}"
        )
    if patient_predictions["patient_id"].duplicated().any():
        raise ValueError("Patient predictions contain duplicate patient IDs.")
    if slice_predictions["filename"].duplicated().any():
        raise ValueError("Slice predictions contain duplicate filenames.")
    if set(patient_predictions["patient_id"]) != set(slice_predictions["patient_id"]):
        raise ValueError("Patient and slice prediction files contain different patients.")

    records = []
    for patient in patient_predictions.sort_values("patient_id").itertuples(index=False):
        patient_id = str(patient.patient_id)
        slices = slice_predictions[
            slice_predictions["patient_id"].astype(str) == patient_id
        ].copy()
        if len(slices) != int(patient.slice_count):
            raise ValueError(
                f"{patient_id} has {len(slices)} slice predictions but the patient "
                f"table records {int(patient.slice_count)}."
            )
        patient_probability = float(
            getattr(patient, f"{ENSEMBLE_NAME}_probability")
        )
        slices["selection_distance"] = (
            slices[f"{ENSEMBLE_NAME}_probability"].astype(float)
            - patient_probability
        ).abs()
        selected = slices.sort_values(
            ["selection_distance", "filename"], kind="stable"
        ).iloc[0]
        prediction = int(getattr(patient, f"{ENSEMBLE_NAME}_prediction"))
        record = {
            "patient_id": patient_id,
            "label": int(patient.label),
            "source_site": str(patient.source_site),
            "category": confusion_category(int(patient.label), prediction),
            "slice_count": int(patient.slice_count),
            "ensemble_patient_probability": patient_probability,
            "ensemble_patient_prediction": prediction,
            "selected_filename": str(selected["filename"]),
            "ensemble_slice_probability": float(
                selected[f"{ENSEMBLE_NAME}_probability"]
            ),
            "ensemble_slice_prediction": int(
                selected[f"{ENSEMBLE_NAME}_prediction"]
            ),
            "selection_distance": float(selected["selection_distance"]),
            "slice_selection_rule": "closest_to_patient_mean_probability",
        }
        for name in MODEL_NAMES:
            record[f"{name}_patient_probability"] = float(
                getattr(patient, f"{name}_probability")
            )
            record[f"{name}_patient_prediction"] = int(
                getattr(patient, f"{name}_prediction")
            )
            record[f"{name}_slice_probability"] = float(
                selected[f"{name}_probability"]
            )
            record[f"{name}_slice_prediction"] = int(
                selected[f"{name}_prediction"]
            )
        records.append(record)

    cases = pd.DataFrame(records)
    cases["featured"] = False
    cases["featured_reason"] = ""
    for category, reason in (
        ("TP", "median_confidence_true_positive"),
        ("TN", "median_confidence_true_negative"),
    ):
        group = cases[cases["category"] == category]
        if group.empty:
            continue
        median = float(group["ensemble_patient_probability"].median())
        selected_index = (
            group.assign(
                distance_to_category_median=(
                    group["ensemble_patient_probability"] - median
                ).abs()
            )
            .sort_values(
                ["distance_to_category_median", "patient_id"], kind="stable"
            )
            .index[0]
        )
        cases.loc[selected_index, ["featured", "featured_reason"]] = [True, reason]
    for category, reason in (
        ("FP", "all_false_positives"),
        ("FN", "all_false_negatives"),
    ):
        mask = cases["category"] == category
        cases.loc[mask, ["featured", "featured_reason"]] = [True, reason]

    cases["category_order"] = cases["category"].map(CONFUSION_ORDER)
    return (
        cases.sort_values(["category_order", "patient_id"])
        .drop(columns="category_order")
        .reset_index(drop=True)
    )


def validate_case_counts(cases: pd.DataFrame, slice_predictions: pd.DataFrame) -> None:
    actual = {
        "slices": int(len(slice_predictions)),
        "patients": int(len(cases)),
        "positive_patients": int(cases["label"].sum()),
        "proxy_comparator_patients": int((cases["label"] == 0).sum()),
    }
    if actual != EXPECTED_TEST:
        raise ValueError(f"Unexpected test counts. Expected {EXPECTED_TEST}, found {actual}.")


def find_backbone(model: keras.Model) -> keras.Model:
    nested = [layer for layer in model.layers if isinstance(layer, keras.Model)]
    if len(nested) != 1:
        names = [layer.name for layer in nested]
        raise ValueError(f"Expected exactly one nested backbone, found {names}.")
    return nested[0]


def build_gradcam_models(
    model: keras.Model, model_name: str
) -> tuple[keras.Model, keras.Model, str]:
    backbone = find_backbone(model)
    expected_layer = EXPECTED_TARGET_LAYERS[model_name]
    try:
        target = backbone.get_layer(expected_layer)
    except ValueError as error:
        raise ValueError(
            f"{model_name} backbone does not contain Grad-CAM layer '{expected_layer}'."
        ) from error
    if target.output.ndim != 4:
        raise ValueError(f"Grad-CAM target '{target.name}' is not a spatial feature map.")

    feature_model = keras.Model(backbone.input, target.output)
    pooled = keras.layers.GlobalAveragePooling2D()(feature_model.output)
    trained_head = model.get_layer("head")
    linear_head = keras.layers.Dense(
        1,
        activation=None,
        use_bias=trained_head.use_bias,
        name=f"{model_name}_gradcam_linear_head",
    )
    logits = linear_head(pooled)
    linear_head.set_weights(trained_head.get_weights())
    gradcam_model = keras.Model(
        feature_model.input, [logits, feature_model.output]
    )
    preprocessing_model = keras.Model(
        model.input, model.get_layer("imagenet_preprocessing").output
    )
    return preprocessing_model, gradcam_model, target.name


def compute_gradcam(
    gradcam_model: keras.Model, prepared_image: np.ndarray, target_label: int
) -> np.ndarray:
    if target_label not in (0, 1):
        raise ValueError("Grad-CAM target label must be binary.")
    image = tf.convert_to_tensor(prepared_image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        logits, features = gradcam_model(image, training=False)
        score = logits[0, 0] if target_label == 1 else -logits[0, 0]
    gradients = tape.gradient(score, features)
    if gradients is None:
        raise ValueError("Grad-CAM gradients are disconnected from the target feature map.")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(
        weights[:, None, None, :] * features, axis=-1
    )[0]
    heatmap = tf.nn.relu(heatmap)
    maximum = float(tf.reduce_max(heatmap).numpy())
    if maximum > keras.backend.epsilon():
        heatmap = heatmap / maximum
    else:
        heatmap = tf.zeros_like(heatmap)
    result = heatmap.numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Grad-CAM produced non-finite values.")
    return result


def resize_heatmap(heatmap: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def validate_recomputed_probability(
    model_name: str,
    patient_id: str,
    expected_probability: float,
    actual_probability: float,
    expected_prediction: int,
    maximum_drift: float = MAX_PROBABILITY_DRIFT,
) -> float:
    values = np.asarray([expected_probability, actual_probability], dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Saved and recomputed probabilities must be finite and in [0, 1].")
    if expected_prediction not in (0, 1):
        raise ValueError("Saved slice prediction must be binary.")
    if int(expected_probability >= THRESHOLD) != expected_prediction:
        raise ValueError(
            f"Saved {model_name} probability and prediction disagree for {patient_id}."
        )
    actual_prediction = int(actual_probability >= THRESHOLD)
    if actual_prediction != expected_prediction:
        raise ValueError(
            f"{model_name} classification for {patient_id} changed from "
            f"{expected_prediction} to {actual_prediction}."
        )
    drift = abs(actual_probability - expected_probability)
    if drift > maximum_drift:
        raise ValueError(
            f"{model_name} probability drift for {patient_id} is {drift:.8f}, "
            f"exceeding the allowed {maximum_drift:.8f}."
        )
    return drift


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not safe:
        raise ValueError("Cannot create an artifact name from an empty patient ID.")
    return safe


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "opencv-python",
        "scikit-learn",
        "tensorflow",
        "keras",
        "matplotlib",
    ):
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


def _load_final_evaluation(config: dict) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
    dict[str, Path],
    dict[str, Path],
]:
    evaluation_dir = PROJECT_ROOT / "outputs" / "evaluation" / "test"
    artifacts = {
        "patient_predictions": evaluation_dir / "patient_predictions.csv",
        "slice_predictions": evaluation_dir / "slice_predictions.csv",
        "metrics": evaluation_dir / "metrics.json",
        "provenance": evaluation_dir / "provenance.json",
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing finalized test artifact(s): {missing}")

    provenance = json.loads(artifacts["provenance"].read_text(encoding="utf-8"))
    if provenance.get("split") != "test":
        raise ValueError("Evaluation provenance is not from the test split.")
    if provenance.get("threshold") != THRESHOLD:
        raise ValueError("Evaluation threshold does not match the fixed threshold.")
    if tuple(provenance.get("model_names", ())) != MODEL_NAMES:
        raise ValueError("Evaluation provenance does not contain the required models.")

    training_dir = Path(config["training"]["output_dir"])
    checkpoints = {
        name: training_dir / name / "best.keras" for name in MODEL_NAMES
    }
    missing_checkpoints = [
        str(path) for path in checkpoints.values() if not path.is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(f"Missing trained checkpoint(s): {missing_checkpoints}")
    current_checkpoint_hashes = {
        name: sha256_file(path) for name, path in checkpoints.items()
    }
    if provenance.get("checkpoint_sha256") != current_checkpoint_hashes:
        raise ValueError("Checkpoint hashes no longer match final test provenance.")

    processed_dir = Path(config["data"]["processed_dir"])
    current_artifacts = {
        "evaluation_script_sha256": sha256_file(
            PROJECT_ROOT / "scripts" / "evaluate_models.py"
        ),
        "training_config_sha256": sha256_file(
            training_dir / "training_config.json"
        ),
        "manifest_sha256": sha256_file(processed_dir / "manifest.csv"),
        "release_manifest_sha256": sha256_file(
            processed_dir / "release_manifest.json"
        ),
    }
    mismatched = [
        key
        for key, value in current_artifacts.items()
        if provenance.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            f"Final test evaluation artifacts no longer match: {mismatched}."
        )
    if provenance.get("package_versions") != {
        key: value
        for key, value in _package_versions().items()
        if key in provenance.get("package_versions", {})
    }:
        raise ValueError("Package versions changed after final test evaluation.")

    metrics = json.loads(artifacts["metrics"].read_text(encoding="utf-8"))
    if (
        metrics.get("split") != "test"
        or metrics.get("patients") != EXPECTED_TEST["patients"]
        or metrics.get("positive_patients") != EXPECTED_TEST["positive_patients"]
        or metrics.get("proxy_comparator_patients")
        != EXPECTED_TEST["proxy_comparator_patients"]
    ):
        raise ValueError("Final test metrics do not match the frozen test cohort.")

    patients = pd.read_csv(artifacts["patient_predictions"])
    slices = pd.read_csv(artifacts["slice_predictions"])
    return patients, slices, provenance, checkpoints, artifacts


def _load_selected_image(processed_dir: Path, filename: str) -> np.ndarray:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npy":
        raise ValueError(f"Unsafe selected slice filename: {filename}")
    path = (processed_dir / relative).resolve()
    if processed_dir.resolve() not in path.parents:
        raise ValueError(f"Selected slice escapes processed directory: {filename}")
    image = to_three_channels(np.load(path)).astype(np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Selected slice has unexpected shape: {image.shape}")
    if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("Selected slice must contain finite values in [0, 1].")
    return image


def _write_patient_panel(
    case: pd.Series,
    image: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    target_layers: dict[str, str],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    base = image[..., 0]
    axes[0].imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original selected slice")
    axes[0].axis("off")
    for axis, name in zip(axes[1:], MODEL_NAMES):
        axis.imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
        axis.imshow(heatmaps[name], cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
        axis.set_title(
            f"{name}\n"
            f"slice p={case[f'{name}_slice_probability']:.3f}, "
            f"target={case[f'{name}_slice_prediction']}\n"
            f"layer={target_layers[name]}",
            fontsize=8,
        )
        axis.axis("off")
    figure.suptitle(
        f"{case['patient_id']} | {case['category']} | true={case['label']} | "
        f"ensemble patient p={case['ensemble_patient_probability']:.3f}\n"
        f"representative slice: {Path(case['selected_filename']).name}",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_gradcam(
    config_path: str | Path, split: str, confirm_final_test_explanations: bool
) -> Path:
    validate_request(split, confirm_final_test_explanations)
    config = load_config(config_path)
    validate_dataset(config, check_all_files=True)
    output_dir = PROJECT_ROOT / "outputs" / "explainability" / split
    ensure_output_available(output_dir)

    patients, slices, evaluation_provenance, checkpoints, artifacts = (
        _load_final_evaluation(config)
    )
    cases = select_representative_slices(patients, slices)
    validate_case_counts(cases, slices)
    processed_dir = Path(config["data"]["processed_dir"])
    images = {
        row.patient_id: _load_selected_image(processed_dir, row.selected_filename)
        for row in cases.itertuples(index=False)
    }

    heatmaps: dict[str, dict[str, np.ndarray]] = {
        patient_id: {} for patient_id in cases["patient_id"]
    }
    probability_audit: dict[tuple[str, str], tuple[float, float]] = {}
    target_layers = {}
    for model_name in MODEL_NAMES:
        print(f"Loading {model_name} for Grad-CAM...")
        keras.backend.clear_session()
        model = keras.models.load_model(checkpoints[model_name], compile=False)
        preprocessing_model, gradcam_model, target_layer = build_gradcam_models(
            model, model_name
        )
        target_layers[model_name] = target_layer
        for case in cases.itertuples(index=False):
            image = images[case.patient_id]
            actual_probability = float(
                np.asarray(model(image[None, ...], training=False)).reshape(-1)[0]
            )
            expected_probability = float(
                getattr(case, f"{model_name}_slice_probability")
            )
            drift = validate_recomputed_probability(
                model_name,
                case.patient_id,
                expected_probability,
                actual_probability,
                int(getattr(case, f"{model_name}_slice_prediction")),
            )
            probability_audit[(case.patient_id, model_name)] = (
                actual_probability,
                drift,
            )
            prepared = np.asarray(
                preprocessing_model(image[None, ...], training=False)
            )[0]
            raw_heatmap = compute_gradcam(
                gradcam_model,
                prepared,
                int(getattr(case, f"{model_name}_slice_prediction")),
            )
            heatmaps[case.patient_id][model_name] = resize_heatmap(
                raw_heatmap, image.shape[:2]
            )
        del gradcam_model, preprocessing_model, model
        keras.backend.clear_session()
        gc.collect()

    for model_name in MODEL_NAMES:
        cases[f"{model_name}_recomputed_slice_probability"] = cases[
            "patient_id"
        ].map(lambda patient_id: probability_audit[(patient_id, model_name)][0])
        cases[f"{model_name}_absolute_probability_drift"] = cases[
            "patient_id"
        ].map(lambda patient_id: probability_audit[(patient_id, model_name)][1])

    patient_panels_dir = output_dir / "patient_panels"
    featured_dir = output_dir / "featured_cases"
    heatmaps_dir = output_dir / "heatmaps"
    patient_panels_dir.mkdir(parents=True, exist_ok=True)
    featured_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    panel_paths = {}
    for _, case in cases.iterrows():
        safe_patient = _safe_name(str(case["patient_id"]))
        patient_heatmaps_dir = heatmaps_dir / safe_patient
        patient_heatmaps_dir.mkdir(parents=True, exist_ok=True)
        for model_name in MODEL_NAMES:
            np.save(
                patient_heatmaps_dir / f"{model_name}.npy",
                heatmaps[str(case["patient_id"])][model_name],
            )
        panel_name = f"{case['category']}_{safe_patient}.png"
        panel_path = patient_panels_dir / panel_name
        _write_patient_panel(
            case,
            images[str(case["patient_id"])],
            heatmaps[str(case["patient_id"])],
            target_layers,
            panel_path,
        )
        panel_paths[str(case["patient_id"])] = panel_path.relative_to(
            output_dir
        ).as_posix()
        if bool(case["featured"]):
            shutil.copy2(panel_path, featured_dir / panel_name)

    cases["panel_file"] = cases["patient_id"].map(panel_paths)
    case_index_path = output_dir / "case_index.csv"
    cases.to_csv(case_index_path, index=False)
    present_categories = set(cases["category"])
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "patients": int(len(cases)),
        "representative_slices": int(len(cases)),
        "featured_cases": int(cases["featured"].sum()),
        "present_confusion_categories": sorted(
            present_categories, key=CONFUSION_ORDER.get
        ),
        "absent_confusion_categories": [
            category for category in CONFUSION_ORDER if category not in present_categories
        ],
        "patient_selection_rule": "all_test_patients",
        "slice_selection_rule": "slice_probability_closest_to_patient_mean_probability",
        "featured_selection_rule": (
            "median-confidence TP and TN plus every FP and FN; ties by patient ID"
        ),
        "gradcam_target_rule": "selected_slice_predicted_class_logit",
        "negative_class_score": "negative_of_positive_class_logit",
        "maximum_allowed_probability_drift": MAX_PROBABILITY_DRIFT,
        "maximum_observed_probability_drift": {
            name: float(cases[f"{name}_absolute_probability_drift"].max())
            for name in MODEL_NAMES
        },
        "target_layers": target_layers,
        "explanation_scope_warning": (
            "Each panel explains one deterministic representative slice. The patient "
            "prediction is a mean over all retained slices, and the soft-voting ensemble "
            "has no single convolutional feature map."
        ),
        "source_cohort_confounded_with_label": True,
        "proxy_comparator_is_not_confirmed_endometriosis_negative": True,
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_status": _git_value(["status", "--porcelain"]),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "gradcam_script_sha256": sha256_file(Path(__file__).resolve()),
        "case_index_sha256": sha256_file(case_index_path),
        "evaluation_provenance_sha256": sha256_file(artifacts["provenance"]),
        "evaluation_artifact_sha256": {
            name: sha256_file(path) for name, path in artifacts.items()
        },
        "checkpoint_sha256": evaluation_provenance["checkpoint_sha256"],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(
        f"Grad-CAM complete: {len(cases)} patient panels, "
        f"{int(cases['featured'].sum())} featured cases."
    )
    print(f"Artifacts saved to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic trained-model Grad-CAM explanations."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument(
        "--confirm-final-test-explanations",
        action="store_true",
        help="Required to explain the finalized internal test predictions.",
    )
    args = parser.parse_args()
    run_gradcam(args.config, args.split, args.confirm_final_test_explanations)


if __name__ == "__main__":
    main()
