from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

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
MODEL_NAMES = ("ResNet50", "EfficientNetV2S", "DenseNet121")
ENSEMBLE_NAME = "SoftVotingEnsemble"
ALL_MODELS = (*MODEL_NAMES, ENSEMBLE_NAME)
THRESHOLD = 0.5
WILSON_Z = 1.959963984540054
PRIMARY_BORDER_WIDTH = 32
SECONDARY_BORDER_WIDTH = 22
# Slice probabilities were serialized from float32 inference outputs.
PROBABILITY_ATOL = 1.1e-7
MAX_GRADCAM_PROBABILITY_DRIFT = 0.005
EXPECTED_SPLITS = {
    "validation": {"patients": 22, "positive": 13, "proxy": 9, "slices": 1063},
    "test": {"patients": 23, "positive": 14, "proxy": 9, "slices": 1287},
}
EXPECTED_SOURCE_LABELS = {"UT-D1": 1, "UT-D2": 1, "MOGaMBO": 0}
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
WILSON_METRICS = ("accuracy", "sensitivity", "specificity", "precision")
BOOTSTRAP_PRIMARY_METRICS = (
    "roc_auc",
    "balanced_accuracy",
    "f1",
    "brier_score",
)
MODEL_COLORS = {
    "ResNet50": "#355070",
    "EfficientNetV2S": "#6D597A",
    "DenseNet121": "#B56576",
    "SoftVotingEnsemble": "#EAAC8B",
}
EXPECTED_TARGET_LAYERS = {
    "ResNet50": "conv5_block3_out",
    "EfficientNetV2S": "top_activation",
    "DenseNet121": "relu",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_available(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Reporting output directory is not empty: {path}. "
            "Preserve the existing report rather than overwriting it."
        )


def wilson_interval(successes: int, total: int, z: float = WILSON_Z) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval denominator must be positive.")
    if successes < 0 or successes > total:
        raise ValueError("Wilson interval successes must be between zero and total.")
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        / denominator
        * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total))
    )
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    if lower < 1e-15:
        lower = 0.0
    if 1.0 - upper < 1e-15:
        upper = 1.0
    return lower, upper


def compute_binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("Labels and scores must be non-empty arrays of equal length.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both operational classes are required for reporting.")
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Probabilities must be finite and bounded by [0, 1].")
    predicted = (scores >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def stratified_bootstrap_intervals(
    labels: np.ndarray, scores: np.ndarray, iterations: int, seed: int
) -> dict[str, dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive.")
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Both classes are required for stratified bootstrap intervals.")
    samples = {key: [] for key in METRIC_KEYS}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sampled = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        metrics = compute_binary_metrics(labels[sampled], scores[sampled])
        for key in METRIC_KEYS:
            samples[key].append(float(metrics[key]))
    return {
        key: {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
        }
        for key, values in samples.items()
    }


def make_border_mask(shape: tuple[int, int], width: int) -> np.ndarray:
    height, image_width = shape
    if width <= 0 or width * 2 >= min(shape):
        raise ValueError("Border width must leave a non-border image interior.")
    mask = np.zeros((height, image_width), dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask


def edge_connected_zero_mask(image: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("Padding proxy requires a two-dimensional image.")
    candidates = image <= tolerance
    visited = np.zeros_like(candidates, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    height, width = candidates.shape
    for row in range(height):
        for column in (0, width - 1):
            if candidates[row, column] and not visited[row, column]:
                visited[row, column] = True
                queue.append((row, column))
    for column in range(width):
        for row in (0, height - 1):
            if candidates[row, column] and not visited[row, column]:
                visited[row, column] = True
                queue.append((row, column))
    while queue:
        row, column = queue.popleft()
        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and candidates[next_row, next_column]
                and not visited[next_row, next_column]
            ):
                visited[next_row, next_column] = True
                queue.append((next_row, next_column))
    return visited


def foreground_intensity_proxy(image: np.ndarray, padding_mask: np.ndarray) -> np.ndarray | None:
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [2.0, 98.0])
    if not np.isfinite([low, high]).all() or high <= low:
        return None
    threshold = low + 0.01 * (high - low)
    return (image > threshold) & ~padding_mask


def _top_fraction_mask(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    indices = np.argpartition(flat, flat.size - count)[-count:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[indices] = True
    return mask.reshape(values.shape)


def region_attention_metrics(
    heatmap: np.ndarray, region: np.ndarray | None, top_mask: np.ndarray
) -> dict[str, float | bool | None]:
    if region is None:
        return {
            "area_fraction": None,
            "activation_mass_fraction": None,
            "activation_enrichment": None,
            "mean_inside": None,
            "mean_outside": None,
            "inside_outside_ratio": None,
            "argmax_inside": None,
            "top10_fraction_inside": None,
        }
    region = np.asarray(region, dtype=bool)
    if region.shape != heatmap.shape:
        raise ValueError("Attention region shape does not match heatmap shape.")
    area_fraction = float(region.mean())
    total_mass = float(heatmap.sum())
    mass_fraction = float(heatmap[region].sum() / total_mass) if total_mass > 0 else None
    enrichment = (
        float(mass_fraction / area_fraction)
        if mass_fraction is not None and area_fraction > 0
        else None
    )
    mean_inside = float(heatmap[region].mean()) if region.any() else None
    outside = ~region
    mean_outside = float(heatmap[outside].mean()) if outside.any() else None
    ratio = (
        float(mean_inside / mean_outside)
        if mean_inside is not None and mean_outside is not None and mean_outside > 0
        else None
    )
    argmax = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape) if total_mass > 0 else None
    return {
        "area_fraction": area_fraction,
        "activation_mass_fraction": mass_fraction,
        "activation_enrichment": enrichment,
        "mean_inside": mean_inside,
        "mean_outside": mean_outside,
        "inside_outside_ratio": ratio,
        "argmax_inside": bool(region[argmax]) if argmax is not None else None,
        "top10_fraction_inside": float(region[top_mask].mean()) if total_mass > 0 else None,
    }


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    points = np.argwhere(mask)
    if not len(points):
        return None
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def heatmap_center_of_mass(heatmap: np.ndarray) -> tuple[float, float] | None:
    total = float(heatmap.sum())
    if total <= 0:
        return None
    rows, columns = np.indices(heatmap.shape)
    return (
        float((rows * heatmap).sum() / total),
        float((columns * heatmap).sum() / total),
    )


def _safe_resolve(root: Path, relative_value: str, suffix: str | None = None) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative artifact path: {relative_value}")
    if suffix is not None and relative.suffix.lower() != suffix.lower():
        raise ValueError(f"Unexpected artifact extension: {relative_value}")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError(f"Artifact path escapes expected root: {relative_value}")
    return resolved


def _require_files(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen reporting artifact(s): {missing}")


def _assert_close(actual: float, expected: float, message: str, atol: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=1e-9, atol=atol):
        raise ValueError(f"{message}: expected {expected}, found {actual}.")


def _validate_binary_values(values, message: str) -> None:
    numeric = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{message} must contain only exact binary values 0 and 1.")


def _validate_prediction_columns(frame: pd.DataFrame, split: str, level: str) -> None:
    base = {"patient_id", "label", "source_site"}
    if level == "patient":
        base.add("slice_count")
    else:
        base.update({"filename", "split", "processed_sha256"})
    for model in ALL_MODELS:
        base.update({f"{model}_probability", f"{model}_prediction"})
    missing = base - set(frame.columns)
    if missing:
        raise ValueError(f"{split} {level} predictions are missing {sorted(missing)}.")
    if frame[list(base)].isna().any().any():
        raise ValueError(f"{split} {level} predictions contain missing required values.")
    _validate_binary_values(frame["label"], f"{split} {level} labels")
    for model in ALL_MODELS:
        _validate_binary_values(
            frame[f"{model}_prediction"], f"{split} {level} {model} predictions"
        )
    if level == "patient":
        slice_counts = pd.to_numeric(frame["slice_count"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if (
            not np.isfinite(slice_counts).all()
            or (slice_counts <= 0).any()
            or not np.equal(slice_counts, np.floor(slice_counts)).all()
        ):
            raise ValueError(f"{split} patient slice counts must be positive integers.")


def _validate_split_artifacts(evaluation_dir: Path, split: str) -> dict:
    split_dir = evaluation_dir / split
    paths = {
        "patient_predictions": split_dir / "patient_predictions.csv",
        "slice_predictions": split_dir / "slice_predictions.csv",
        "metrics": split_dir / "metrics.json",
        "provenance": split_dir / "provenance.json",
    }
    _require_files(paths)
    patients = pd.read_csv(paths["patient_predictions"])
    slices = pd.read_csv(paths["slice_predictions"])
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    _validate_prediction_columns(patients, split, "patient")
    _validate_prediction_columns(slices, split, "slice")
    expected = EXPECTED_SPLITS[split]
    if len(patients) != expected["patients"] or len(slices) != expected["slices"]:
        raise ValueError(f"Unexpected frozen {split} patient or slice counts.")
    if patients["patient_id"].duplicated().any() or slices["filename"].duplicated().any():
        raise ValueError(f"Duplicate patient or slice identifiers found in {split}.")
    if set(slices["split"].astype(str)) != {split}:
        raise ValueError(f"Slice prediction split values disagree with {split}.")
    if int(patients["label"].sum()) != expected["positive"]:
        raise ValueError(f"Unexpected positive-patient count in {split}.")
    if int((patients["label"] == 0).sum()) != expected["proxy"]:
        raise ValueError(f"Unexpected proxy-comparator count in {split}.")
    source_labels = patients[["source_site", "label"]].drop_duplicates()
    if any(
        EXPECTED_SOURCE_LABELS.get(str(row.source_site)) != int(row.label)
        for row in source_labels.itertuples(index=False)
    ):
        raise ValueError(f"Frozen {split} source-site and operational-label mapping changed.")
    expected_metric_metadata = {
        "split": split,
        "threshold": THRESHOLD,
        "patients": expected["patients"],
        "positive_patients": expected["positive"],
        "proxy_comparator_patients": expected["proxy"],
    }
    for key, value in expected_metric_metadata.items():
        if metrics.get(key) != value:
            raise ValueError(f"Frozen {split} metrics have unexpected {key} metadata.")
    if set(metrics.get("models", {})) != set(ALL_MODELS):
        raise ValueError(f"Frozen {split} metrics have an unexpected model set.")
    if set(patients["patient_id"].astype(str)) != set(slices["patient_id"].astype(str)):
        raise ValueError(f"Patient and slice tables disagree for {split}.")
    if provenance.get("split") != split or float(provenance.get("threshold")) != THRESHOLD:
        raise ValueError(f"Frozen {split} provenance has unexpected methodology values.")
    if tuple(provenance.get("model_names", ())) != MODEL_NAMES:
        raise ValueError(f"Frozen {split} provenance has unexpected model order.")
    if provenance.get("source_cohort_confounded_with_label") is not True:
        raise ValueError("Source-cohort confounding warning is missing from provenance.")
    if provenance.get("proxy_comparator_is_not_confirmed_endometriosis_negative") is not True:
        raise ValueError("Proxy-comparator warning is missing from provenance.")
    weights = provenance.get("voting_weights", {})
    expected_weight = 1.0 / len(MODEL_NAMES)
    if (
        provenance.get("voting") != "equal_weight_patient_probability_mean"
        or set(weights) != set(MODEL_NAMES)
        or not all(np.isclose(float(weights[name]), expected_weight) for name in MODEL_NAMES)
    ):
        raise ValueError(f"Frozen {split} voting weights are not the expected equal weights.")

    labels = patients["label"].to_numpy(dtype=np.int64)
    recomputed_metrics: dict[str, dict[str, float | int]] = {}
    for frame_name, frame in (("patient", patients), ("slice", slices)):
        probabilities = frame[[f"{name}_probability" for name in MODEL_NAMES]].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"{split} {frame_name} probabilities are invalid.")
        expected_ensemble = probabilities.mean(axis=1)
        if not np.allclose(
            expected_ensemble,
            frame[f"{ENSEMBLE_NAME}_probability"].to_numpy(dtype=np.float64),
            rtol=1e-7,
            atol=PROBABILITY_ATOL,
        ):
            raise ValueError(f"{split} {frame_name} ensemble probabilities changed.")
        for model in ALL_MODELS:
            scores = frame[f"{model}_probability"].to_numpy(dtype=np.float64)
            predictions = frame[f"{model}_prediction"].to_numpy(dtype=np.int64)
            if not np.array_equal(predictions, (scores >= THRESHOLD).astype(np.int64)):
                raise ValueError(f"{split} {frame_name} {model} predictions changed.")

    patient_groups = slices.groupby("patient_id", sort=False)
    slice_counts = patient_groups.size()
    for patient in patients.itertuples(index=False):
        patient_id = str(patient.patient_id)
        if int(slice_counts[patient_id]) != int(patient.slice_count):
            raise ValueError(f"{split} slice count changed for {patient_id}.")
        patient_slices = patient_groups.get_group(patient_id)
        if set(patient_slices["label"].astype(int)) != {int(patient.label)}:
            raise ValueError(f"{split} slice labels changed for {patient_id}.")
        if set(patient_slices["source_site"].astype(str)) != {str(patient.source_site)}:
            raise ValueError(f"{split} slice source sites changed for {patient_id}.")
    for model in ALL_MODELS:
        grouped = patient_groups[f"{model}_probability"].mean()
        expected_scores = patients["patient_id"].astype(str).map(grouped).to_numpy(dtype=np.float64)
        actual_scores = patients[f"{model}_probability"].to_numpy(dtype=np.float64)
        if not np.allclose(expected_scores, actual_scores, rtol=1e-7, atol=PROBABILITY_ATOL):
            raise ValueError(f"{split} patient pooling changed for {model}.")
        recomputed = compute_binary_metrics(labels, actual_scores)
        recomputed_metrics[model] = recomputed
        stored = metrics.get("models", {}).get(model, {}).get("point_estimate", {})
        expected_point_metadata = {
            "patients": expected["patients"],
            "positive_patients": expected["positive"],
            "proxy_comparator_patients": expected["proxy"],
            "threshold": THRESHOLD,
        }
        for key, value in expected_point_metadata.items():
            if stored.get(key) != value:
                raise ValueError(f"Frozen {split} {model} has unexpected {key} metadata.")
        for key, value in recomputed.items():
            if key not in stored:
                raise ValueError(f"Frozen {split} metrics omit {model} {key}.")
            if isinstance(value, int):
                if int(stored[key]) != value:
                    raise ValueError(f"Frozen {split} {model} {key} changed.")
            else:
                _assert_close(float(value), float(stored[key]), f"Frozen {split} {model} {key}")
        intervals = metrics["models"][model].get("bootstrap_95_ci", {})
        reproduced_intervals = stratified_bootstrap_intervals(
            labels,
            actual_scores,
            iterations=int(provenance["bootstrap_iterations"]),
            seed=int(provenance["bootstrap_seed"]),
        )
        for key in METRIC_KEYS:
            interval = intervals.get(key, {})
            lower = float(interval.get("lower", math.nan))
            upper = float(interval.get("upper", math.nan))
            if not np.isfinite([lower, upper]).all() or not 0 <= lower <= upper <= 1:
                raise ValueError(f"Frozen {split} bootstrap interval is invalid for {model} {key}.")
            _assert_close(
                lower,
                reproduced_intervals[key]["lower"],
                f"Frozen {split} {model} {key} bootstrap lower bound",
            )
            _assert_close(
                upper,
                reproduced_intervals[key]["upper"],
                f"Frozen {split} {model} {key} bootstrap upper bound",
            )
    return {
        "paths": paths,
        "patients": patients.sort_values("patient_id").reset_index(drop=True),
        "slices": slices,
        "metrics": metrics,
        "provenance": provenance,
        "recomputed_metrics": recomputed_metrics,
    }


def _wilson_counts(metrics: dict[str, float | int], metric: str) -> tuple[int, int]:
    tn = int(metrics["true_negative"])
    fp = int(metrics["false_positive"])
    fn = int(metrics["false_negative"])
    tp = int(metrics["true_positive"])
    counts = {
        "accuracy": (tp + tn, tp + tn + fp + fn),
        "sensitivity": (tp, tp + fn),
        "specificity": (tn, tn + fp),
        "precision": (tp, tp + fp),
    }
    return counts[metric]


def build_statistical_tables(split_data: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    wide_rows = []
    for split, data in split_data.items():
        for model in ALL_MODELS:
            metrics = data["recomputed_metrics"][model]
            bootstrap = data["metrics"]["models"][model]["bootstrap_95_ci"]
            wide: dict[str, str | float] = {"split": split, "model": model}
            for metric in METRIC_KEYS:
                estimate = float(metrics[metric])
                if metric in WILSON_METRICS:
                    numerator, denominator = _wilson_counts(metrics, metric)
                    lower, upper = wilson_interval(numerator, denominator)
                    method = "wilson_score_95"
                else:
                    numerator = denominator = None
                    lower = float(bootstrap[metric]["lower"])
                    upper = float(bootstrap[metric]["upper"])
                    method = "stratified_patient_bootstrap_percentile_95"
                long_rows.append(
                    {
                        "split": split,
                        "model": model,
                        "metric": metric,
                        "estimate": estimate,
                        "primary_interval_method": method,
                        "primary_lower": lower,
                        "primary_upper": upper,
                        "numerator": numerator,
                        "denominator": denominator,
                        "retained_bootstrap_lower": float(bootstrap[metric]["lower"]),
                        "retained_bootstrap_upper": float(bootstrap[metric]["upper"]),
                        "bootstrap_iterations": int(data["provenance"]["bootstrap_iterations"]),
                        "bootstrap_seed": int(data["provenance"]["bootstrap_seed"]),
                    }
                )
                wide[f"{metric}_estimate"] = estimate
                wide[f"{metric}_lower"] = lower
                wide[f"{metric}_upper"] = upper
                wide[f"{metric}_interval_method"] = method
            wide_rows.append(wide)
    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


def _cohen_kappa(first: np.ndarray, second: np.ndarray) -> float | None:
    observed = float(np.mean(first == second))
    first_positive = float(np.mean(first == 1))
    second_positive = float(np.mean(second == 1))
    expected = first_positive * second_positive + (1 - first_positive) * (1 - second_positive)
    if np.isclose(expected, 1.0):
        return None
    return float((observed - expected) / (1.0 - expected))


def build_agreement_tables(split_data: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    overview_rows = []
    for split, data in split_data.items():
        patients = data["patients"]
        for index, first_name in enumerate(ALL_MODELS):
            for second_name in ALL_MODELS[index + 1 :]:
                first = patients[f"{first_name}_prediction"].to_numpy(dtype=np.int64)
                second = patients[f"{second_name}_prediction"].to_numpy(dtype=np.int64)
                n00 = int(np.sum((first == 0) & (second == 0)))
                n01 = int(np.sum((first == 0) & (second == 1)))
                n10 = int(np.sum((first == 1) & (second == 0)))
                n11 = int(np.sum((first == 1) & (second == 1)))
                discordant = patients.loc[first != second, "patient_id"].astype(str).tolist()
                first_scores = patients[f"{first_name}_probability"].astype(float)
                second_scores = patients[f"{second_name}_probability"].astype(float)
                difference = (first_scores - second_scores).abs()
                pair_rows.append(
                    {
                        "split": split,
                        "model_a": first_name,
                        "model_b": second_name,
                        "n00": n00,
                        "n01": n01,
                        "n10": n10,
                        "n11": n11,
                        "overall_agreement": float((n00 + n11) / len(patients)),
                        "positive_agreement": float(2 * n11 / (2 * n11 + n10 + n01))
                        if 2 * n11 + n10 + n01
                        else None,
                        "negative_agreement": float(2 * n00 / (2 * n00 + n10 + n01))
                        if 2 * n00 + n10 + n01
                        else None,
                        "cohen_kappa": _cohen_kappa(first, second),
                        "discordant_count": len(discordant),
                        "discordant_patient_ids": ";".join(discordant),
                        "pearson_probability": float(first_scores.corr(second_scores, method="pearson")),
                        "spearman_probability": float(first_scores.corr(second_scores, method="spearman")),
                        "mean_absolute_probability_difference": float(difference.mean()),
                        "maximum_absolute_probability_difference": float(difference.max()),
                    }
                )
        base_votes = patients[[f"{name}_prediction" for name in MODEL_NAMES]].to_numpy(dtype=int)
        positive_votes = base_votes.sum(axis=1)
        hard_majority = (positive_votes >= 2).astype(int)
        ensemble = patients[f"{ENSEMBLE_NAME}_prediction"].to_numpy(dtype=int)
        patterns = pd.Series(["".join(map(str, row)) for row in base_votes]).value_counts().sort_index()
        overview_rows.append(
            {
                "split": split,
                "patients": len(patients),
                "base_models_unanimous": int(np.sum((positive_votes == 0) | (positive_votes == 3))),
                "base_models_two_to_one": int(np.sum((positive_votes == 1) | (positive_votes == 2))),
                "soft_vote_hard_majority_disagreements": int(np.sum(ensemble != hard_majority)),
                "vote_pattern_counts": ";".join(f"{key}:{value}" for key, value in patterns.items()),
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(overview_rows)


def _flatten_region(prefix: str, values: dict[str, float | bool | None]) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def audit_gradcam(
    explainability_dir: Path,
    processed_dir: Path,
    test_data: dict,
    consumed_paths: set[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    paths = {
        "case_index": explainability_dir / "case_index.csv",
        "provenance": explainability_dir / "provenance.json",
    }
    _require_files(paths)
    cases = pd.read_csv(paths["case_index"])
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    consumed_paths.update(paths.values())
    if sha256_file(paths["case_index"]) != provenance.get("case_index_sha256"):
        raise ValueError("Grad-CAM case index no longer matches its provenance.")
    expected_provenance = {
        "split": "test",
        "patients": EXPECTED_SPLITS["test"]["patients"],
        "representative_slices": EXPECTED_SPLITS["test"]["patients"],
        "featured_cases": 3,
        "patient_selection_rule": "all_test_patients",
        "slice_selection_rule": "slice_probability_closest_to_patient_mean_probability",
        "featured_selection_rule": (
            "median-confidence TP and TN plus every FP and FN; ties by patient ID"
        ),
        "gradcam_target_rule": "selected_slice_predicted_class_logit",
        "negative_class_score": "negative_of_positive_class_logit",
        "maximum_allowed_probability_drift": MAX_GRADCAM_PROBABILITY_DRIFT,
        "target_layers": EXPECTED_TARGET_LAYERS,
        "source_cohort_confounded_with_label": True,
        "proxy_comparator_is_not_confirmed_endometriosis_negative": True,
    }
    mismatched_provenance = [
        key for key, value in expected_provenance.items() if provenance.get(key) != value
    ]
    if mismatched_provenance:
        raise ValueError(f"Grad-CAM provenance changed: {mismatched_provenance}")
    if provenance.get("checkpoint_sha256") != test_data["provenance"].get("checkpoint_sha256"):
        raise ValueError("Grad-CAM checkpoint identities differ from final evaluation provenance.")
    if sha256_file(test_data["paths"]["provenance"]) != provenance.get(
        "evaluation_provenance_sha256"
    ):
        raise ValueError("Grad-CAM evaluation provenance link changed.")
    evaluation_hashes = provenance.get("evaluation_artifact_sha256", {})
    for name, path in test_data["paths"].items():
        if sha256_file(path) != evaluation_hashes.get(name):
            raise ValueError(f"Grad-CAM provenance no longer matches test {name}.")
    if len(cases) != EXPECTED_SPLITS["test"]["patients"]:
        raise ValueError("Grad-CAM case index does not contain all test patients.")
    required_case_columns = {
        "patient_id",
        "label",
        "source_site",
        "category",
        "slice_count",
        "selected_filename",
        "slice_selection_rule",
        "featured",
        "featured_reason",
        "panel_file",
        "ensemble_patient_prediction",
        "ensemble_slice_prediction",
    }
    for model in MODEL_NAMES:
        required_case_columns.update(
            {
                f"{model}_patient_prediction",
                f"{model}_slice_prediction",
                f"{model}_recomputed_slice_probability",
                f"{model}_absolute_probability_drift",
            }
        )
    missing_case_columns = required_case_columns - set(cases.columns)
    if missing_case_columns:
        raise ValueError(f"Grad-CAM case index is missing {sorted(missing_case_columns)}.")
    _validate_binary_values(cases["label"], "Grad-CAM case labels")
    for column in (
        "ensemble_patient_prediction",
        "ensemble_slice_prediction",
        *(f"{model}_{level}_prediction" for model in MODEL_NAMES for level in ("patient", "slice")),
    ):
        _validate_binary_values(cases[column], f"Grad-CAM {column}")
    if cases["patient_id"].astype(str).duplicated().any():
        raise ValueError("Grad-CAM case index contains duplicate patient identifiers.")
    patients = test_data["patients"].set_index("patient_id")
    slices = test_data["slices"].set_index("filename")
    if set(cases["patient_id"].astype(str)) != set(patients.index.astype(str)):
        raise ValueError("Grad-CAM case index does not contain the exact frozen test cohort.")
    if cases["featured"].isna().any() or not cases["featured"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("Grad-CAM featured flags are not strict booleans.")
    if int(cases["featured"].sum()) != int(provenance["featured_cases"]):
        raise ValueError("Grad-CAM featured-case count changed.")
    expected_featured = pd.Series(False, index=cases.index)
    expected_featured_reason = pd.Series("", index=cases.index, dtype=object)
    for category, reason in (
        ("TP", "median_confidence_true_positive"),
        ("TN", "median_confidence_true_negative"),
    ):
        group = cases[cases["category"] == category]
        if len(group):
            median = float(group["ensemble_patient_probability"].median())
            selected_index = (
                group.assign(
                    distance_to_category_median=(
                        group["ensemble_patient_probability"] - median
                    ).abs()
                )
                .sort_values(["distance_to_category_median", "patient_id"], kind="stable")
                .index[0]
            )
            expected_featured.loc[selected_index] = True
            expected_featured_reason.loc[selected_index] = reason
    for category, reason in (("FP", "all_false_positives"), ("FN", "all_false_negatives")):
        mask = cases["category"] == category
        expected_featured.loc[mask] = True
        expected_featured_reason.loc[mask] = reason
    actual_featured_reason = cases["featured_reason"].fillna("").astype(str)
    if not cases["featured"].equals(expected_featured) or not actual_featured_reason.equals(
        expected_featured_reason
    ):
        raise ValueError("Grad-CAM featured-case selection changed.")
    rows = []
    integrity = []
    observed_drifts = {model: [] for model in MODEL_NAMES}
    for case in cases.itertuples(index=False):
        patient_id = str(case.patient_id)
        if patient_id not in patients.index:
            raise ValueError(f"Unknown Grad-CAM patient: {patient_id}")
        patient = patients.loc[patient_id]
        if int(case.label) != int(patient["label"]):
            raise ValueError(f"Grad-CAM label changed for {patient_id}.")
        if str(case.source_site) != str(patient["source_site"]):
            raise ValueError(f"Grad-CAM source site changed for {patient_id}.")
        if int(case.slice_count) != int(patient["slice_count"]):
            raise ValueError(f"Grad-CAM slice count changed for {patient_id}.")
        if str(case.slice_selection_rule) != "closest_to_patient_mean_probability":
            raise ValueError(f"Grad-CAM slice-selection rule changed for {patient_id}.")
        selected_filename = str(case.selected_filename)
        if selected_filename not in slices.index:
            raise ValueError(f"Unknown selected slice for {patient_id}: {selected_filename}")
        selected = slices.loc[selected_filename]
        if str(selected["patient_id"]) != patient_id:
            raise ValueError(f"Selected slice belongs to a different patient: {patient_id}")
        patient_slices = test_data["slices"][
            test_data["slices"]["patient_id"].astype(str) == patient_id
        ].copy()
        patient_probability = float(patient[f"{ENSEMBLE_NAME}_probability"])
        patient_slices["selection_distance"] = (
            patient_slices[f"{ENSEMBLE_NAME}_probability"].astype(float)
            - patient_probability
        ).abs()
        expected_selected = patient_slices.sort_values(
            ["selection_distance", "filename"], kind="stable"
        ).iloc[0]
        if str(expected_selected["filename"]) != selected_filename:
            raise ValueError(f"Representative-slice selection changed for {patient_id}.")
        expected_category = {
            (1, 1): "TP",
            (0, 0): "TN",
            (0, 1): "FP",
            (1, 0): "FN",
        }[(int(patient["label"]), int(patient[f"{ENSEMBLE_NAME}_prediction"]))]
        if str(case.category) != expected_category:
            raise ValueError(f"Grad-CAM confusion category changed for {patient_id}.")
        expected_panel = f"patient_panels/{expected_category}_{patient_id}.png"
        if str(case.panel_file) != expected_panel:
            raise ValueError(f"Grad-CAM panel path changed for {patient_id}.")
        _assert_close(
            float(case.ensemble_patient_probability),
            patient_probability,
            f"Grad-CAM ensemble patient probability for {patient_id}",
        )
        if int(case.ensemble_patient_prediction) != int(patient[f"{ENSEMBLE_NAME}_prediction"]):
            raise ValueError(f"Grad-CAM ensemble patient prediction changed for {patient_id}.")
        _assert_close(
            float(case.ensemble_slice_probability),
            float(selected[f"{ENSEMBLE_NAME}_probability"]),
            f"Grad-CAM ensemble slice probability for {patient_id}",
            atol=PROBABILITY_ATOL,
        )
        _assert_close(
            float(case.selection_distance),
            float(expected_selected["selection_distance"]),
            f"Grad-CAM selection distance for {patient_id}",
            atol=PROBABILITY_ATOL,
        )
        if int(case.ensemble_slice_prediction) != int(selected[f"{ENSEMBLE_NAME}_prediction"]):
            raise ValueError(f"Grad-CAM ensemble slice prediction changed for {patient_id}.")
        maximum_drift = MAX_GRADCAM_PROBABILITY_DRIFT
        for model in MODEL_NAMES:
            _assert_close(
                float(getattr(case, f"{model}_patient_probability")),
                float(patient[f"{model}_probability"]),
                f"Grad-CAM {model} patient probability for {patient_id}",
            )
            if int(getattr(case, f"{model}_patient_prediction")) != int(
                patient[f"{model}_prediction"]
            ):
                raise ValueError(f"Grad-CAM {model} patient prediction changed for {patient_id}.")
            saved_slice_probability = float(selected[f"{model}_probability"])
            _assert_close(
                float(getattr(case, f"{model}_slice_probability")),
                saved_slice_probability,
                f"Grad-CAM {model} slice probability for {patient_id}",
                atol=PROBABILITY_ATOL,
            )
            saved_slice_prediction = int(selected[f"{model}_prediction"])
            if int(getattr(case, f"{model}_slice_prediction")) != saved_slice_prediction:
                raise ValueError(f"Grad-CAM {model} target class changed for {patient_id}.")
            recomputed_probability = float(
                getattr(case, f"{model}_recomputed_slice_probability")
            )
            drift = float(getattr(case, f"{model}_absolute_probability_drift"))
            if not np.isfinite([recomputed_probability, drift]).all() or not (
                0 <= recomputed_probability <= 1 and drift >= 0
            ):
                raise ValueError(f"Grad-CAM {model} probability audit is invalid for {patient_id}.")
            _assert_close(
                drift,
                abs(recomputed_probability - saved_slice_probability),
                f"Grad-CAM {model} probability drift for {patient_id}",
                atol=PROBABILITY_ATOL,
            )
            if drift > maximum_drift or int(recomputed_probability >= THRESHOLD) != saved_slice_prediction:
                raise ValueError(f"Grad-CAM {model} probability audit failed for {patient_id}.")
            observed_drifts[model].append(drift)
        processed_path = _safe_resolve(processed_dir, selected_filename, suffix=".npy")
        if not processed_path.is_file():
            raise FileNotFoundError(f"Missing selected processed slice: {processed_path}")
        consumed_paths.add(processed_path)
        image = np.load(processed_path, allow_pickle=False)
        if image.shape != (224, 224) or image.dtype != np.float32:
            raise ValueError(f"Selected slice schema changed for {patient_id}: {image.shape}, {image.dtype}")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"Selected slice values are invalid for {patient_id}.")
        array_sha256 = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        if array_sha256 != str(selected["processed_sha256"]):
            raise ValueError(f"Selected processed slice hash changed for {patient_id}.")
        padding = edge_connected_zero_mask(image)
        foreground = foreground_intensity_proxy(image, padding)
        border32 = make_border_mask(image.shape, PRIMARY_BORDER_WIDTH)
        border22 = make_border_mask(image.shape, SECONDARY_BORDER_WIDTH)
        foreground_center = _centroid(foreground) if foreground is not None else None
        for model in MODEL_NAMES:
            heatmap_path = _safe_resolve(
                explainability_dir,
                f"heatmaps/{patient_id}/{model}.npy",
                suffix=".npy",
            )
            if not heatmap_path.is_file():
                raise FileNotFoundError(f"Missing Grad-CAM heatmap: {heatmap_path}")
            consumed_paths.add(heatmap_path)
            heatmap = np.load(heatmap_path, allow_pickle=False)
            if heatmap.shape != image.shape or heatmap.dtype != np.float32:
                raise ValueError(f"Grad-CAM schema changed: {heatmap_path}")
            if not np.isfinite(heatmap).all() or heatmap.min() < 0 or heatmap.max() > 1:
                raise ValueError(f"Grad-CAM values are invalid: {heatmap_path}")
            top_mask = _top_fraction_mask(heatmap)
            center = heatmap_center_of_mass(heatmap)
            distance = None
            if center is not None and foreground_center is not None:
                distance = float(math.dist(center, foreground_center))
            row = {
                "patient_id": patient_id,
                "category": str(case.category),
                "label": int(case.label),
                "source_site": str(patient["source_site"]),
                "model": model,
                "model_slice_target_class": int(getattr(case, f"{model}_slice_prediction")),
                "selected_filename": selected_filename,
                "heatmap_file": heatmap_path.relative_to(explainability_dir).as_posix(),
                "selected_array_sha256": array_sha256,
                "selected_file_sha256": sha256_file(processed_path),
                "heatmap_sha256": sha256_file(heatmap_path),
                "zero_heatmap": bool(float(heatmap.sum()) <= 0),
                "heatmap_center_row": center[0] if center else None,
                "heatmap_center_column": center[1] if center else None,
                "foreground_center_row": foreground_center[0] if foreground_center else None,
                "foreground_center_column": foreground_center[1] if foreground_center else None,
                "heatmap_foreground_center_distance_pixels": distance,
            }
            row.update(_flatten_region("border32", region_attention_metrics(heatmap, border32, top_mask)))
            row.update(_flatten_region("border22", region_attention_metrics(heatmap, border22, top_mask)))
            row.update(_flatten_region("edge_zero", region_attention_metrics(heatmap, padding, top_mask)))
            row.update(_flatten_region("foreground", region_attention_metrics(heatmap, foreground, top_mask)))
            row.update(
                _flatten_region(
                    "border32_edge_zero",
                    region_attention_metrics(heatmap, border32 & padding, top_mask),
                )
            )
            rows.append(row)
        panel_path = _safe_resolve(explainability_dir, str(case.panel_file), suffix=".png")
        if not panel_path.is_file():
            raise FileNotFoundError(f"Missing patient panel: {panel_path}")
        consumed_paths.add(panel_path)
        if bool(case.featured):
            featured_path = explainability_dir / "featured_cases" / panel_path.name
            if not featured_path.is_file():
                raise FileNotFoundError(f"Missing featured case copy: {featured_path}")
            consumed_paths.add(featured_path)
            if sha256_file(featured_path) != sha256_file(panel_path):
                raise ValueError(f"Featured case copy differs from patient panel: {patient_id}")
    stored_maximum_drifts = provenance.get("maximum_observed_probability_drift", {})
    if set(stored_maximum_drifts) != set(MODEL_NAMES):
        raise ValueError("Grad-CAM maximum observed drift model set changed.")
    for model, values in observed_drifts.items():
        _assert_close(
            float(stored_maximum_drifts[model]),
            max(values),
            f"Grad-CAM {model} maximum observed probability drift",
            atol=PROBABILITY_ATOL,
        )
    per_map = pd.DataFrame(rows)
    summary_metrics = (
        "border32_activation_mass_fraction",
        "border22_activation_mass_fraction",
        "edge_zero_activation_mass_fraction",
        "foreground_activation_mass_fraction",
        "border32_activation_enrichment",
        "edge_zero_activation_enrichment",
        "foreground_activation_enrichment",
        "heatmap_foreground_center_distance_pixels",
    )
    summary_rows = []
    for model, group in per_map.groupby("model", sort=False):
        for metric in summary_metrics:
            values = group[metric].dropna().astype(float)
            summary_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "count": int(len(values)),
                    "median": float(values.median()) if len(values) else None,
                    "q1": float(values.quantile(0.25)) if len(values) else None,
                    "q3": float(values.quantile(0.75)) if len(values) else None,
                    "minimum": float(values.min()) if len(values) else None,
                    "maximum": float(values.max()) if len(values) else None,
                }
            )
    integrity.append(
        {
            "check": "gradcam_artifact_counts",
            "status": "pass",
            "details": f"{len(cases)} patients, {len(per_map)} heatmaps, {int(cases['featured'].sum())} featured cases",
        }
    )
    integrity.append(
        {
            "check": "gradcam_heatmap_authenticity",
            "status": "warning",
            "details": (
                "The producer provenance contains no heatmap hash manifest. The report records "
                "current heatmap hashes for future checks, but cannot cryptographically establish "
                "that transferred heatmaps are the originally generated files."
            ),
        }
    )
    return per_map, pd.DataFrame(summary_rows), provenance, integrity


def _plot_roc(patients: pd.DataFrame, split: str, figures_dir: Path) -> None:
    labels = patients["label"].to_numpy(dtype=int)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    for model in ALL_MODELS:
        scores = patients[f"{model}_probability"].to_numpy(dtype=float)
        false_positive, true_positive, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        axis.plot(
            false_positive,
            true_positive,
            linewidth=2,
            color=MODEL_COLORS[model],
            label=f"{model} (AUC={auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", alpha=0.7)
    axis.set(
        xlabel="False-positive rate (proxy-comparator records)",
        ylabel="True-positive rate (UT-EndoMRI records)",
        title=f"{split.title()} patient-level ROC curves",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figures_dir / f"{split}_roc_curves.{suffix}", dpi=300)
    plt.close(figure)


def _plot_confusions(patients: pd.DataFrame, split: str, figures_dir: Path) -> None:
    labels = patients["label"].to_numpy(dtype=int)
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    for axis, model in zip(axes.flat, ALL_MODELS):
        predicted = patients[f"{model}_prediction"].to_numpy(dtype=int)
        matrix = confusion_matrix(labels, predicted, labels=[0, 1])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14)
        axis.set(
            title=model,
            xlabel="Predicted operational label",
            ylabel="Observed operational label",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Proxy (0)", "UT (1)"],
            yticklabels=["Proxy (0)", "UT (1)"],
        )
    figure.suptitle(f"{split.title()} patient-level confusion matrices (threshold=0.5)")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figures_dir / f"{split}_confusion_matrices.{suffix}", dpi=300)
    plt.close(figure)


def _plot_metric_forest(statistics: pd.DataFrame, figures_dir: Path) -> None:
    test = statistics[statistics["split"] == "test"].copy()
    metric_labels = {
        "roc_auc": "ROC-AUC",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "precision": "Precision",
        "f1": "F1",
        "brier_score": "Brier score",
    }
    figure, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)
    for axis, metric in zip(axes.flat, METRIC_KEYS):
        subset = test[test["metric"] == metric].set_index("model").loc[list(ALL_MODELS)]
        positions = np.arange(len(ALL_MODELS))
        estimates = subset["estimate"].to_numpy(dtype=float)
        lower = subset["primary_lower"].to_numpy(dtype=float)
        upper = subset["primary_upper"].to_numpy(dtype=float)
        axis.errorbar(
            estimates,
            positions,
            xerr=np.vstack([estimates - lower, upper - estimates]),
            fmt="o",
            capsize=3,
            color="#355070",
        )
        axis.set_yticks(positions, ALL_MODELS if metric in ("roc_auc", "specificity") else [""] * 4)
        axis.set_xlim(0, 1.02)
        axis.set_title(metric_labels[metric])
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    figure.suptitle("Internal test metrics with primary 95% intervals")
    figure.text(
        0.5,
        0.01,
        "Wilson: accuracy, sensitivity, specificity, precision. Retained stratified bootstrap: ROC-AUC, balanced accuracy, F1, Brier.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    for suffix in ("png", "pdf"):
        figure.savefig(figures_dir / f"test_metrics_forest.{suffix}", dpi=300)
    plt.close(figure)


def _plot_agreement(agreement: pd.DataFrame, figures_dir: Path) -> None:
    test = agreement[agreement["split"] == "test"]
    matrix = np.eye(len(ALL_MODELS), dtype=float)
    for row in test.itertuples(index=False):
        first = ALL_MODELS.index(row.model_a)
        second = ALL_MODELS.index(row.model_b)
        value = float(row.cohen_kappa) if pd.notna(row.cohen_kappa) else math.nan
        matrix[first, second] = matrix[second, first] = value
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    for row in range(len(ALL_MODELS)):
        for column in range(len(ALL_MODELS)):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                "NA" if np.isnan(value) else f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if not np.isnan(value) and value >= 0.6 else "black",
            )
    axis.set_xticks(range(len(ALL_MODELS)), ALL_MODELS, rotation=25, ha="right")
    axis.set_yticks(range(len(ALL_MODELS)), ALL_MODELS)
    axis.set_title("Internal test pairwise prediction agreement (Cohen's kappa)")
    figure.colorbar(image, ax=axis, label="Kappa")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figures_dir / f"model_agreement_matrix.{suffix}", dpi=300)
    plt.close(figure)


def _plot_gradcam(per_map: pd.DataFrame, figures_dir: Path) -> None:
    metrics = (
        ("border32_activation_enrichment", "32-pixel border"),
        ("edge_zero_activation_enrichment", "Edge-connected zero/background proxy"),
        ("foreground_activation_enrichment", "Foreground-intensity proxy"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    for axis, (metric, title) in zip(axes, metrics):
        values = [
            per_map.loc[per_map["model"] == model, metric].dropna().to_numpy(dtype=float)
            for model in MODEL_NAMES
        ]
        axis.boxplot(values, tick_labels=MODEL_NAMES, showfliers=False)
        for index, series in enumerate(values, start=1):
            jitter = np.linspace(-0.08, 0.08, len(series)) if len(series) else np.array([])
            axis.scatter(np.full(len(series), index) + jitter, series, s=12, alpha=0.55, color="#B56576")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.axhline(1.0, linestyle="--", color="#555555", linewidth=1)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Activation enrichment (1.0 = proportional to region area)")
    figure.suptitle("Exploratory Grad-CAM artifact-attention audit (23 representative slices)")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figures_dir / f"gradcam_artifact_attention.{suffix}", dpi=300)
    plt.close(figure)


def _git_value(arguments: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "matplotlib", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _write_report(
    path: Path,
    statistics: pd.DataFrame,
    agreement_overview: pd.DataFrame,
    gradcam_summary: pd.DataFrame,
    integrity: pd.DataFrame,
) -> None:
    test_ensemble = statistics[
        (statistics["split"] == "test") & (statistics["model"] == ENSEMBLE_NAME)
    ].set_index("metric")
    overview = agreement_overview.set_index("split").loc["test"]
    warning_rows = integrity[integrity["status"] == "warning"]
    lines = [
        "# Final Reporting and Statistical Audit",
        "",
        "## Scope",
        "",
        "This report reads frozen validation, internal-test, and Grad-CAM artifacts. It does not load checkpoints, rerun inference, alter predictions, optimize the threshold, or pool validation and test records.",
        "",
        "## Required Interpretation Warnings",
        "",
        "- MOGaMBO is a cervical-cancer proxy comparator, not a healthy or confirmed endometriosis-negative cohort.",
        "- Source cohort is confounded with operational label; these results measure internal cohort discrimination and do not establish clinical endometriosis diagnosis.",
        "- The internal test cohort contains 23 records (14 operational positives and 9 proxy comparators).",
        "- Grad-CAM explains one representative slice per patient and one component model at a time. There is no single ensemble convolutional feature map.",
        "- Spatial Grad-CAM measurements use maps enlarged from native 7 x 7 activations to 224 x 224 and must not be interpreted as pixel-precise localization.",
        "- Border, zero/background, and foreground analyses are exploratory artifact-attention audits, not lesion-localization validation.",
        "- External hospital evaluation and expert clinical review remain pending.",
    ]
    for row in warning_rows.itertuples(index=False):
        lines.append(f"- {row.details}")
    lines.extend(
        [
            "",
            "## Internal Test Ensemble",
            "",
            "| Metric | Estimate | Primary 95% interval | Method |",
            "|---|---:|---:|---|",
        ]
    )
    display_names = {
        "roc_auc": "ROC-AUC",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "precision": "Precision",
        "f1": "F1",
        "brier_score": "Brier score",
    }
    for metric in METRIC_KEYS:
        row = test_ensemble.loc[metric]
        lines.append(
            f"| {display_names[metric]} | {row['estimate']:.4f} | "
            f"[{row['primary_lower']:.4f}, {row['primary_upper']:.4f}] | "
            f"{row['primary_interval_method']} |"
        )
    lines.extend(
        [
            "",
            "## Model Agreement",
            "",
            f"The three base models were unanimous for {int(overview['base_models_unanimous'])} of {int(overview['patients'])} internal-test records. Soft voting and hard majority voting differed for {int(overview['soft_vote_hard_majority_disagreements'])} records.",
            "",
            "## Grad-CAM Audit",
            "",
            f"The audit analyzed {EXPECTED_SPLITS['test']['patients']} heatmap files per model using a 32-pixel primary border and a 22-pixel sensitivity border. The edge-connected zero/background mask and foreground-intensity mask are proxies rather than anatomical annotations.",
            "",
            "## Artifact Integrity",
            "",
        ]
    )
    for row in integrity.itertuples(index=False):
        lines.append(f"- **{row.status.upper()}** `{row.check}`: {row.details}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_manifest(paths: set[Path]) -> list[dict]:
    rows = []
    for path in sorted({path.resolve() for path in paths}, key=str):
        rows.append(
            {
                "path": path.relative_to(PROJECT_ROOT.resolve()).as_posix()
                if PROJECT_ROOT.resolve() in path.parents
                else str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _discover_preflight_paths(
    evaluation_dir: Path, explainability_dir: Path, processed_dir: Path
) -> set[Path]:
    paths = {
        evaluation_dir / split / name
        for split in ("validation", "test")
        for name in (
            "patient_predictions.csv",
            "slice_predictions.csv",
            "metrics.json",
            "provenance.json",
        )
    }
    case_index_path = explainability_dir / "case_index.csv"
    paths.update({case_index_path, explainability_dir / "provenance.json"})
    for directory in ("heatmaps", "patient_panels", "featured_cases"):
        paths.update(path for path in (explainability_dir / directory).rglob("*") if path.is_file())
    expected_counts = {
        "heatmaps": ("*.npy", 69),
        "patient_panels": ("*.png", 23),
        "featured_cases": ("*.png", 3),
    }
    for directory, (pattern, expected) in expected_counts.items():
        actual = len(list((explainability_dir / directory).rglob(pattern)))
        if actual != expected:
            raise ValueError(
                f"Unexpected {directory} artifact count. Expected {expected}, found {actual}."
            )
    for name in ("manifest.csv", "release_manifest.json"):
        path = processed_dir / name
        if path.is_file():
            paths.add(path)
    _require_files({str(index): path for index, path in enumerate(paths)})
    case_index_hash = sha256_file(case_index_path)
    cases = pd.read_csv(case_index_path)
    if sha256_file(case_index_path) != case_index_hash:
        raise RuntimeError("Grad-CAM case index changed while discovering source artifacts.")
    for filename in cases["selected_filename"].astype(str):
        paths.add(_safe_resolve(processed_dir, filename, suffix=".npy"))
    _require_files({str(index): path for index, path in enumerate(paths)})
    return paths


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256"):
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def generate_final_report(
    evaluation_dir: Path,
    explainability_dir: Path,
    processed_dir: Path,
    output_dir: Path,
) -> Path:
    evaluation_dir = evaluation_dir.resolve()
    explainability_dir = explainability_dir.resolve()
    processed_dir = processed_dir.resolve()
    output_dir = output_dir.resolve()
    for input_dir in (evaluation_dir, explainability_dir, processed_dir):
        if (
            output_dir == input_dir
            or output_dir in input_dir.parents
            or input_dir in output_dir.parents
        ):
            raise ValueError(
                f"Reporting output must not overlap a frozen input tree: {output_dir} and {input_dir}."
            )
    ensure_output_available(output_dir)
    script_initial_hash = sha256_file(Path(__file__).resolve())
    consumed_paths = _discover_preflight_paths(
        evaluation_dir, explainability_dir, processed_dir
    )
    input_manifest = _file_manifest(consumed_paths)
    initial_hashes = {row["path"]: row["sha256"] for row in input_manifest}
    split_data = {
        split: _validate_split_artifacts(evaluation_dir, split)
        for split in ("validation", "test")
    }
    overlap = set(split_data["validation"]["patients"]["patient_id"]) & set(
        split_data["test"]["patients"]["patient_id"]
    )
    if overlap:
        raise ValueError(f"Validation and test patient identifiers overlap: {sorted(overlap)}")
    integrity_rows = [
        {
            "check": "frozen_evaluation_reproduction",
            "status": "pass",
            "details": "All validation and test point estimates, bootstrap intervals, predictions, patient means, and equal-weight ensemble probabilities reproduced from frozen CSV files.",
        },
        {
            "check": "validation_test_disjointness",
            "status": "pass",
            "details": "Validation and internal-test patient identifiers are disjoint.",
        },
    ]
    validation_provenance = split_data["validation"]["provenance"]
    test_provenance = split_data["test"]["provenance"]
    lock_keys = (
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
    mismatched_lock = [
        key for key in lock_keys if validation_provenance.get(key) != test_provenance.get(key)
    ]
    if mismatched_lock:
        raise ValueError(f"Validation-to-test methodology lock changed: {mismatched_lock}")

    current_global = {
        "manifest_sha256": processed_dir / "manifest.csv",
        "release_manifest_sha256": processed_dir / "release_manifest.json",
    }
    global_mismatch = []
    for key, path in current_global.items():
        if path.is_file():
            consumed_paths.add(path)
            if sha256_file(path) != test_provenance.get(key):
                global_mismatch.append(key)
    if global_mismatch:
        integrity_rows.append(
            {
                "check": "current_global_preprocessing_release",
                "status": "warning",
                "details": (
                    "The current local preprocessing manifest/release files are not byte-identical "
                    f"to the partner-run evaluation provenance ({', '.join(global_mismatch)}). "
                    "Frozen evaluation CSVs remain authoritative for this report."
                ),
            }
        )
    else:
        integrity_rows.append(
            {
                "check": "current_global_preprocessing_release",
                "status": "pass",
                "details": "Current local preprocessing manifests match partner-run evaluation provenance.",
            }
        )

    statistics, comparison = build_statistical_tables(split_data)
    agreement, agreement_overview = build_agreement_tables(split_data)
    gradcam_per_map, gradcam_summary, gradcam_provenance, gradcam_integrity = audit_gradcam(
        explainability_dir,
        processed_dir,
        split_data["test"],
        consumed_paths,
    )
    integrity_rows.extend(gradcam_integrity)
    integrity = pd.DataFrame(integrity_rows)
    cohort_summary = pd.DataFrame(
        [
            {
                "split": split,
                "patients": len(data["patients"]),
                "positive_records": int(data["patients"]["label"].sum()),
                "proxy_comparator_records": int((data["patients"]["label"] == 0).sum()),
                "slices": len(data["slices"]),
            }
            for split, data in split_data.items()
        ]
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        figures_dir = temporary / "figures"
        figures_dir.mkdir(parents=True)
        cohort_summary.to_csv(temporary / "cohort_summary.csv", index=False)
        statistics.to_csv(temporary / "statistical_results.csv", index=False)
        comparison.to_csv(temporary / "thesis_model_comparison.csv", index=False)
        agreement.to_csv(temporary / "agreement_summary.csv", index=False)
        agreement_overview.to_csv(temporary / "agreement_overview.csv", index=False)
        discordant = agreement[agreement["discordant_count"] > 0][
            ["split", "model_a", "model_b", "discordant_count", "discordant_patient_ids"]
        ]
        discordant.to_csv(temporary / "discordant_cases.csv", index=False)
        gradcam_per_map.to_csv(temporary / "gradcam_per_map.csv", index=False)
        gradcam_summary.to_csv(temporary / "gradcam_summary.csv", index=False)
        integrity.to_csv(temporary / "integrity_checks.csv", index=False)
        (temporary / "input_manifest.json").write_text(
            json.dumps({"files": input_manifest}, indent=2), encoding="utf-8"
        )
        for split, data in split_data.items():
            _plot_roc(data["patients"], split, figures_dir)
            _plot_confusions(data["patients"], split, figures_dir)
        _plot_metric_forest(statistics, figures_dir)
        _plot_agreement(agreement, figures_dir)
        _plot_gradcam(gradcam_per_map, figures_dir)
        _write_report(
            temporary / "report.md",
            statistics,
            agreement_overview,
            gradcam_summary,
            integrity,
        )
        run_provenance = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reporting_script_sha256": script_initial_hash,
            "python_version": platform.python_version(),
            "package_versions": _package_versions(),
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_status": _git_value(["status", "--porcelain"]),
            "threshold": THRESHOLD,
            "model_names": list(MODEL_NAMES),
            "ensemble": ENSEMBLE_NAME,
            "patient_aggregation": "arithmetic_mean_of_retained_slice_probabilities",
            "ensemble_method": "equal_mean_of_three_patient_probabilities",
            "wilson_z": WILSON_Z,
            "wilson_metrics": list(WILSON_METRICS),
            "retained_bootstrap_primary_metrics": list(BOOTSTRAP_PRIMARY_METRICS),
            "bootstrap_iterations": int(test_provenance["bootstrap_iterations"]),
            "bootstrap_seed": int(test_provenance["bootstrap_seed"]),
            "primary_border_width_pixels": PRIMARY_BORDER_WIDTH,
            "secondary_border_width_pixels": SECONDARY_BORDER_WIDTH,
            "padding_proxy": "four-connected edge pixels with intensity <= 1e-6",
            "foreground_proxy": "P2 + 0.01*(P98-P2), excluding edge-connected zero proxy",
            "gradcam_native_map_shape": [7, 7],
            "gradcam_audit_map_shape": [224, 224],
            "gradcam_spatial_precision_warning": (
                "All spatial measures are exploratory summaries of cubic-upsampled 7 x 7 maps, "
                "not independent 224-pixel-resolution localization measurements."
            ),
            "manifest_mismatch_policy": "prominent_warning",
            "source_cohort_confounded_with_label": True,
            "proxy_comparator_is_not_confirmed_endometriosis_negative": True,
            "gradcam_scope_warning": gradcam_provenance.get("explanation_scope_warning"),
            "gradcam_heatmap_authentication": "unavailable_no_producer_hash_manifest",
        }
        (temporary / "run_provenance.json").write_text(
            json.dumps(run_provenance, indent=2), encoding="utf-8"
        )
        final_manifest = _file_manifest(consumed_paths)
        final_hashes = {row["path"]: row["sha256"] for row in final_manifest}
        if final_hashes != initial_hashes:
            raise RuntimeError("A frozen source artifact changed during report generation.")
        if sha256_file(Path(__file__).resolve()) != script_initial_hash:
            raise RuntimeError("The reporting script changed during report generation.")
        _write_checksums(temporary)
        if output_dir.exists():
            output_dir.rmdir()
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Final reporting audit complete: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a final report from frozen patient predictions and Grad-CAM artifacts."
    )
    parser.add_argument("--evaluation-dir", default="outputs/evaluation")
    parser.add_argument("--explainability-dir", default="outputs/explainability/test")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/reporting/final")
    args = parser.parse_args()
    generate_final_report(
        Path(args.evaluation_dir),
        Path(args.explainability_dir),
        Path(args.processed_dir),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
