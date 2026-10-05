from __future__ import annotations

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def aggregate_patient_predictions(
    manifest_df: pd.DataFrame, probabilities: np.ndarray
) -> pd.DataFrame:
    """Mean-pool ordered slice probabilities into one score per patient."""

    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(scores) != len(manifest_df):
        raise ValueError(
            f"Received {len(scores)} predictions for {len(manifest_df)} manifest rows."
        )
    frame = manifest_df[["patient_id", "label", "source_site"]].copy()
    frame["slice_probability"] = scores
    if frame.groupby("patient_id")["label"].nunique().max() != 1:
        raise ValueError("A patient has inconsistent labels during aggregation.")
    return (
        frame.groupby("patient_id", as_index=False)
        .agg(
            label=("label", "first"),
            source_site=("source_site", "first"),
            probability=("slice_probability", "mean"),
            slice_count=("slice_probability", "size"),
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


class PatientValidationMetrics(keras.callbacks.Callback):
    """Add mean-pooled patient validation loss/AUC to epoch logs."""

    def __init__(self, validation_dataset, validation_manifest: pd.DataFrame):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.validation_manifest = validation_manifest.reset_index(drop=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probabilities = self.model.predict(self.validation_dataset, verbose=0)
        patients = aggregate_patient_predictions(
            self.validation_manifest, probabilities
        )
        labels = patients["label"].to_numpy(dtype=np.float64)
        scores = np.clip(
            patients["probability"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7
        )
        loss = float(
            -np.mean(labels * np.log(scores) + (1.0 - labels) * np.log(1.0 - scores))
        )
        auc = float(roc_auc_score(labels, scores))
        logs["val_patient_loss"] = loss
        logs["val_patient_auc"] = auc
        print(f" - val_patient_loss: {loss:.4f} - val_patient_auc: {auc:.4f}")
