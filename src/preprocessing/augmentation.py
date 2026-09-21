from __future__ import annotations


def build_augmentation_layer(config: dict) -> "keras.Sequential | None":
    """
    Build a Keras preprocessing layer stack from the augmentation config.

    Maps ``configs/config.yaml`` under ``augmentation`` onto Keras 3 layers,
    applied only to the training split:

    - ``rotation_range`` (degrees) -> ``RandomRotation``
      (Keras 3 factor is a fraction of 2*pi, hence ``degrees / 360``)
    - ``zoom_range`` ``[lo, hi]`` -> ``RandomZoom``
      (Keras 3 factor is an offset from 1, hence ``[lo - 1, hi - 1]``)
    - ``translation_shift`` (fraction) -> ``RandomTranslation``

    Returns ``None`` when augmentation is disabled or all transforms are
    neutral, so callers can simply branch on the returned layer.
    """

    aug_cfg = config.get("augmentation", {})

    if not aug_cfg.get("enabled", False):
        return None

    import keras

    rotation_range = float(aug_cfg.get("rotation_range", 0.0))
    zoom_lower, zoom_upper = aug_cfg.get("zoom_range", [1.0, 1.0])
    translation_shift = float(aug_cfg.get("translation_shift", 0.0))

    layers: list[keras.layers.Layer] = []

    if rotation_range > 0.0:
        layers.append(
            keras.layers.RandomRotation(factor=rotation_range / 360.0)
        )

    if zoom_lower != 1.0 or zoom_upper != 1.0:
        layers.append(
            keras.layers.RandomZoom(
                height_factor=(zoom_lower - 1.0, zoom_upper - 1.0),
                width_factor=(zoom_lower - 1.0, zoom_upper - 1.0),
            )
        )

    if translation_shift > 0.0:
        layers.append(
            keras.layers.RandomTranslation(
                height_factor=translation_shift,
                width_factor=translation_shift,
            )
        )

    if not layers:
        return None

    layer = keras.Sequential(layers)
    layer._name = "augmentation_pipeline"  # type: ignore[attr-defined]
    return layer