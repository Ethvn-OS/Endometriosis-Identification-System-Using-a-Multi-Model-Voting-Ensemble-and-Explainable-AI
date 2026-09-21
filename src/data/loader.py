from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.preprocessing.augmentation import build_augmentation_layer
from src.preprocessing.slices import to_three_channels


MANIFEST_REQUIRED_COLUMNS = {"filename", "patient_id", "sequence", "split", "label"}


def load_manifest(processed_dir: str | Path, split: str | None = None) -> pd.DataFrame:
    """
    Load the preprocessing manifest produced by ``src/preprocess.py``.

    Normalizes manifest paths to POSIX style (the current manifest was written
    on Windows with backslashes) so the loader works cross-platform.

    Args:
        processed_dir: Directory containing ``manifest.csv`` and the ``.npy`` files.
        split: Optional split filter: ``"train"`` or ``"validation"``.

    Returns:
        Manifest ``DataFrame`` (optionally reduced to one split).
    """

    processed_dir = Path(processed_dir)
    csv_path = processed_dir / "manifest.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = MANIFEST_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest missing required column(s): {sorted(missing)}"
        )

    df["filename"] = df["filename"].astype(str).str.replace("\\", "/", regex=False)

    if split is not None:
        available = sorted(df["split"].unique())
        if split not in available:
            raise ValueError(
                f"Split '{split}' not present in manifest. Available: {available}"
            )
        df = df[df["split"] == split].reset_index(drop=True)

    return df


def count_per_split(manifest_df: pd.DataFrame) -> dict[str, int]:
    """
    Return the number of slices per split in a manifest.
    """

    return manifest_df.groupby("split")["filename"].count().to_dict()


def _load_and_expand(path) -> np.ndarray:
    """
    Load a single preprocessed ``.npy`` slice and expand it to three channels.

    Runs inside ``tf.py_function``, whose argument may arrive as a Python
    ``bytes`` object or as a scalar ``EagerTensor`` depending on mode.
    """

    if hasattr(path, "numpy"):
        path = path.numpy()

    if isinstance(path, (bytes, np.bytes_)):
        path = path.decode()
    elif isinstance(path, (str, np.str_)):
        path = str(path)

    arr = np.load(path)
    return to_three_channels(arr).astype(np.float32)


def build_dataset(
    config: dict,
    manifest_df: pd.DataFrame,
    augment: bool = False,
    shuffle: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    """
    Build a ``tf.data.Dataset`` from a (single-split) manifest.

    Pipeline: load ``.npy`` -> expand to ``(H, W, 3)`` -> optional shuffle
    -> batch -> optional augmentation (train only) -> prefetch.

    Yields ``(x, y)`` pairs where ``x`` has shape ``(B, H, W, 3)`` (float32,
    values in ``[0, 1]``) and ``y`` has shape ``(B,)`` (int32 binary labels).

    Augmentation is applied per batch with ``training=True`` and is never
    applied to validation.
    """

    if manifest_df.empty:
        raise ValueError("Manifest contains no rows for the requested split.")

    image_h, image_w = config["preprocessing"]["image_size"]
    loader_cfg = config.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 32))
    shuffle_buffer = int(loader_cfg.get("shuffle_buffer", 4096))
    seed = seed if seed is not None else int(loader_cfg.get("seed", 42))

    processed_dir = Path(config["data"]["processed_dir"])
    files = [str(processed_dir / f) for f in manifest_df["filename"].tolist()]
    labels = manifest_df["label"].to_numpy(dtype=np.int32)

    def _decode_map(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.py_function(_load_and_expand, [path], Tout=tf.float32)
        x.set_shape((image_h, image_w, 3))
        return x, tf.cast(label, tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((files, labels))

    ds = ds.map(_decode_map, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    aug_layer = build_augmentation_layer(config) if augment else None
    if aug_layer is not None:
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_train_val_datasets(
    config: dict,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Convenience wrapper: build the train dataset (augmented, shuffled) and the
    validation dataset (deterministic, no augmentation) from ``config``.
    """

    processed_dir = config["data"]["processed_dir"]

    train_df = load_manifest(processed_dir, split="train")
    val_df = load_manifest(processed_dir, split="validation")

    train_ds = build_dataset(config, train_df, augment=True, shuffle=True)
    val_ds = build_dataset(config, val_df, augment=False, shuffle=False)

    return train_ds, val_ds