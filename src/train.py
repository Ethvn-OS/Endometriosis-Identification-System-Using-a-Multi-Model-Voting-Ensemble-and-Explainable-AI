from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import create_datasets
from src.data.loader import load_manifest
from src.evaluation import PatientValidationMetrics
from src.modeling import MODEL_NAMES, build_classifier, compile_classifier
from src.preprocess import load_config
from src.validate_dataset import validate_dataset


def _callbacks(
    output_dir: Path,
    model_name: str,
    config: dict,
    validation_dataset,
    validation_manifest,
):
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    return [
        PatientValidationMetrics(validation_dataset, validation_manifest),
        keras.callbacks.ModelCheckpoint(
            model_dir / "best.keras",
            monitor="val_patient_auc",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_patient_auc",
            mode="max",
            patience=int(training["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_patient_auc",
            mode="max",
            patience=int(training["reduce_lr_patience"]),
            factor=0.2,
            min_lr=1e-7,
        ),
        keras.callbacks.CSVLogger(model_dir / "history.csv"),
    ]


def run(config: dict, train: bool) -> None:
    """Validate readiness; fit models only when ``train`` is explicitly true."""

    report = validate_dataset(config, check_all_files=True)
    print(
        f"Dataset ready: {report['patients']} patients, {report['slices']} slices."
    )
    seed = int(config["loader"]["seed"])
    keras.utils.set_random_seed(seed)
    datasets = create_datasets(config)

    if not train:
        images, labels, weights = next(iter(datasets["train"]))
        print(
            f"Training batch: images={tuple(images.shape)}, labels={tuple(labels.shape)}, "
            f"weights={tuple(weights.shape)}"
        )
        print("Readiness check complete. No model training was started.")
        return

    output_dir = Path(config["training"]["output_dir"])
    configured_models = tuple(config["training"]["model_names"])
    if not configured_models or len(configured_models) != len(set(configured_models)):
        raise ValueError("training.model_names must be non-empty and unique.")
    if any(name not in MODEL_NAMES for name in configured_models):
        raise ValueError(f"Configured model names must be selected from {MODEL_NAMES}.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Training output directory is not empty: {output_dir}. "
            "Move or remove it deliberately before a new run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    validation_manifest = load_manifest(
        config["data"]["processed_dir"], split="validation"
    )
    for model_name in configured_models:
        keras.backend.clear_session()
        model = build_classifier(model_name, weights="imagenet")
        compile_classifier(model, float(config["training"]["learning_rate"]))
        model.fit(
            datasets["train"],
            validation_data=datasets["validation"],
            epochs=int(config["training"]["epochs"]),
            callbacks=_callbacks(
                output_dir,
                model_name,
                config,
                datasets["validation"],
                validation_manifest,
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate training readiness; model fitting requires --train."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fit all configured models. Omit this flag for readiness checks only.",
    )
    args = parser.parse_args()
    run(load_config(args.config), train=args.train)


if __name__ == "__main__":
    main()
