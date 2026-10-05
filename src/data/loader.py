from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.preprocessing.augmentation import build_augmentation_layer
from src.preprocessing.slices import to_three_channels


MANIFEST_REQUIRED_COLUMNS = {
    "filename",
    "patient_id",
    "cohort",
    "source_site",
    "sequence",
    "split",
    "label",
}
VALID_SPLITS = ("train", "validation", "test")


def load_manifest(processed_dir: str | Path, split: str | None = None) -> pd.DataFrame:
    """Load and validate the slice manifest, optionally filtering one split."""

    processed_dir = Path(processed_dir)
    csv_path = processed_dir / "manifest.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = MANIFEST_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required column(s): {sorted(missing)}")

    df["filename"] = df["filename"].astype(str).str.replace("\\", "/", regex=False)
    if df["filename"].duplicated().any():
        raise ValueError("Manifest contains duplicate processed filenames.")
    for filename in df["filename"]:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npy":
            raise ValueError(f"Unsafe manifest filename: {filename}")
        resolved = (processed_dir / relative).resolve()
        if processed_dir.resolve() not in resolved.parents:
            raise ValueError(f"Manifest filename escapes processed directory: {filename}")
    unknown_splits = set(df["split"]) - set(VALID_SPLITS)
    if unknown_splits:
        raise ValueError(f"Manifest contains unknown splits: {sorted(unknown_splits)}")
    if not set(df["label"]).issubset({0, 1}):
        raise ValueError("Manifest labels must be binary values 0 and 1.")
    if split is not None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Expected one of {VALID_SPLITS}.")
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Split '{split}' has no manifest rows.")
    return df


def count_per_split(manifest_df: pd.DataFrame) -> dict[str, int]:
    """Return the number of slices per split."""

    return manifest_df.groupby("split")["filename"].count().to_dict()


def patient_normalized_weights(manifest_df: pd.DataFrame) -> np.ndarray:
    """Give every patient equal aggregate loss weight within each class."""

    label_counts = manifest_df.groupby("patient_id")["label"].nunique()
    if (label_counts != 1).any():
        invalid = label_counts[label_counts != 1].index.tolist()
        raise ValueError(f"Patients with inconsistent labels: {invalid[:10]}")

    slices_per_patient = manifest_df.groupby("patient_id").size()
    patient_labels = manifest_df.groupby("patient_id")["label"].first()
    patients_per_class = patient_labels.value_counts()

    raw_weights = np.array(
        [
            1.0
            / (
                float(slices_per_patient[patient_id])
                * float(patients_per_class[label])
            )
            for patient_id, label in zip(
                manifest_df["patient_id"], manifest_df["label"]
            )
        ],
        dtype=np.float32,
    )
    return raw_weights / raw_weights.mean()


def _load_and_expand(path) -> np.ndarray:
    if hasattr(path, "numpy"):
        path = path.numpy()
    if isinstance(path, (bytes, np.bytes_)):
        path = path.decode()
    elif isinstance(path, (str, np.str_)):
        path = str(path)

    array = np.load(path)
    return to_three_channels(array).astype(np.float32)


def build_dataset(
    config: dict,
    manifest_df: pd.DataFrame,
    augment: bool = False,
    shuffle: bool = True,
    include_patient_weights: bool = False,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Build a TensorFlow dataset for one manifest split."""

    if manifest_df.empty:
        raise ValueError("Manifest contains no rows for the requested split.")

    image_h, image_w = config["preprocessing"]["image_size"]
    loader_cfg = config.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 32))
    shuffle_buffer = int(loader_cfg.get("shuffle_buffer", 4096))
    seed = seed if seed is not None else int(loader_cfg.get("seed", 42))

    processed_dir = Path(config["data"]["processed_dir"])
    files = [str(processed_dir / filename) for filename in manifest_df["filename"]]
    labels = manifest_df["label"].to_numpy(dtype=np.float32)
    weights = (
        patient_normalized_weights(manifest_df)
        if include_patient_weights
        else np.ones(len(manifest_df), dtype=np.float32)
    )

    def _decode_map(path: tf.Tensor, label: tf.Tensor, weight: tf.Tensor):
        image = tf.py_function(_load_and_expand, [path], Tout=tf.float32)
        image.set_shape((image_h, image_w, 3))
        return image, tf.cast(label, tf.float32), tf.cast(weight, tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((files, labels, weights))
    dataset = dataset.map(_decode_map, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(
            min(shuffle_buffer, len(manifest_df)),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(batch_size, drop_remainder=False)
    augmentation = build_augmentation_layer(config) if augment else None
    if augmentation is not None:
        dataset = dataset.map(
            lambda image, label, weight: (
                augmentation(image, training=True),
                label,
                weight,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if not include_patient_weights:
        dataset = dataset.map(
            lambda image, label, weight: (image, label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.prefetch(tf.data.AUTOTUNE)


def assert_disjoint_patients(split_frames: dict[str, pd.DataFrame]) -> None:
    """Fail if any patient appears in more than one split."""

    patient_sets = {
        split_name: set(frame["patient_id"].astype(str))
        for split_name, frame in split_frames.items()
    }
    split_names = list(patient_sets)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            leaked = patient_sets[left_name] & patient_sets[right_name]
            if leaked:
                shown = ", ".join(sorted(leaked)[:10])
                raise ValueError(
                    f"Patient leakage between {left_name} and {right_name}: {shown}"
                )


def create_datasets(config: dict) -> dict[str, tf.data.Dataset]:
    """Create weighted training and deterministic validation/test datasets."""

    processed_dir = config["data"]["processed_dir"]
    frames = {
        split: load_manifest(processed_dir, split=split) for split in VALID_SPLITS
    }
    assert_disjoint_patients(frames)
    return {
        "train": build_dataset(
            config,
            frames["train"],
            augment=True,
            shuffle=True,
            include_patient_weights=True,
        ),
        "validation": build_dataset(
            config, frames["validation"], augment=False, shuffle=False
        ),
        "test": build_dataset(
            config, frames["test"], augment=False, shuffle=False
        ),
    }


def create_train_val_datasets(config: dict) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Compatibility wrapper for the dataset-loader demonstration notebook."""

    datasets = create_datasets(config)
    return datasets["train"], datasets["validation"]
