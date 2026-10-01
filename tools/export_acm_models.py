"""Export the two completed ACM Kaggle runs to LiteRT models for Streamlit.

Run in a Kaggle notebook after attaching both saved training notebook outputs.
The exported models use exactly the final-epoch weights and class order recorded
by the training notebooks. Conversion changes the storage/runtime format only.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


CLASS_NAMES = ["healthy", "black_pod_rot", "pod_borer"]
BLOCK_SPECS = ((64, 2), (128, 2), (256, 2), (512, 3), (512, 2))
EXPECTED_COUNTS = {
    "healthy": {"train": 2340, "val": 669, "test": 335},
    "black_pod_rot": {"train": 660, "val": 188, "test": 95},
    "pod_borer": {"train": 72, "val": 20, "test": 11},
}


@tf.keras.utils.register_keras_serializable(package="Cacao")
class SpatialChannelPool(tf.keras.layers.Layer):
    def call(self, inputs):
        mean_map = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_map = tf.reduce_max(inputs, axis=-1, keepdims=True)
        return tf.concat([mean_map, max_map], axis=-1)


def cbam_block(x, name, ratio=16):
    channels = int(x.shape[-1])
    hidden = max(1, channels // ratio)
    shared_down = tf.keras.layers.Dense(
        hidden, activation="relu", name=f"{name}_channel_down"
    )
    shared_up = tf.keras.layers.Dense(channels, name=f"{name}_channel_up")
    avg = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_avg")(x)
    maximum = tf.keras.layers.GlobalMaxPooling2D(keepdims=True, name=f"{name}_max")(x)
    channel_logits = tf.keras.layers.Add(name=f"{name}_channel_add")([
        shared_up(shared_down(avg)), shared_up(shared_down(maximum))
    ])
    channel_mask = tf.keras.layers.Activation("sigmoid", name=f"{name}_channel_mask")(
        channel_logits
    )
    refined = tf.keras.layers.Multiply(name=f"{name}_channel_refine")([x, channel_mask])
    pooled = SpatialChannelPool(name=f"{name}_spatial_pool")(refined)
    spatial_mask = tf.keras.layers.Conv2D(
        1, kernel_size=7, padding="same", activation="sigmoid", use_bias=False,
        name=f"{name}_spatial_mask"
    )(pooled)
    return tf.keras.layers.Multiply(name=f"{name}_spatial_refine")([
        refined, spatial_mask
    ])


def build_model(use_cbam: bool, seed: int = 42) -> tf.keras.Model:
    """Reconstruct the exact layer names and shapes of the ACM training pair."""
    inputs = tf.keras.Input(shape=(224, 224, 3), name="cacao_rgb_input")
    x = inputs
    for block_index, (filters, count) in enumerate(BLOCK_SPECS, start=1):
        for convolution_index in range(1, count + 1):
            x = tf.keras.layers.Conv2D(
                filters, kernel_size=3, strides=1, padding="same", activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotUniform(
                    seed=seed + 100 * block_index + convolution_index
                ),
                name=f"block{block_index}_conv{convolution_index}",
            )(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=2, strides=2, name=f"block{block_index}_pool"
        )(x)
        if use_cbam:
            x = cbam_block(x, name=f"cbam_after_block{block_index}_pool", ratio=16)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(
        4096, activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 10001),
        name="fc1",
    )(x)
    x = tf.keras.layers.Dense(
        4096, activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 10002),
        name="fc2",
    )(x)
    outputs = tf.keras.layers.Dense(
        3, activation="softmax", dtype="float32",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 10003),
        name="class_probs",
    )(x)
    model = tf.keras.Model(
        inputs, outputs, name="soh_table_cbam" if use_cbam else "soh_table_baseline"
    )
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 3)
    assert sum(isinstance(layer, tf.keras.layers.MaxPooling2D) for layer in model.layers) == 5
    assert sum(isinstance(layer, tf.keras.layers.Conv2D) and layer.name.startswith("block")
               for layer in model.layers) == 11
    assert sum(layer.name.endswith("_spatial_mask") for layer in model.layers) == (5 if use_cbam else 0)
    return model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_completed_runs(input_dir: Path) -> dict:
    found = {"baseline": [], "cbam": []}
    for config_path in input_dir.rglob("run_config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if config.get("method_source") != "ACM paper sections 2.2-2.5":
            continue
        arm = "cbam" if config.get("use_cbam") is True else "baseline"
        if config.get("use_cbam") is not False and arm == "baseline":
            continue
        found[arm].append(config_path.parent)
    for arm, paths in found.items():
        if len(paths) != 1:
            listing = "\n".join(str(path) for path in paths) or "(none)"
            raise ValueError(
                f"Expected exactly one completed ACM {arm} run under {input_dir}; "
                f"found {len(paths)}:\n{listing}\nAttach the correct saved Kaggle notebook output."
            )
    return {arm: paths[0] for arm, paths in found.items()}


def validate_run(run_dir: Path, arm: str) -> tuple[dict, dict]:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    expected_cbam = arm == "cbam"
    checks = {
        "use_cbam": expected_cbam,
        "class_order": CLASS_NAMES,
        "image_shape": [224, 224, 3],
        "model_input_transform": "rgb_float32_divide_by_255",
        "label_encoding": "integer",
        "loss_name": "sparse_categorical_crossentropy",
        "seed": 42,
        "split_seed": 42,
        "initial_lr": 0.1,
        "batch_size": 64,
        "epochs": 100,
        "early_stopping": False,
        "checkpoint_selection": "final_epoch",
        "optimizer": "Adamax",
        "augmentation": False,
        "class_weights": False,
        "dropout": False,
        "clahe": False,
        "l2_regularization": False,
        "cbam_reduction_ratio": 16 if expected_cbam else None,
        "cbam_spatial_kernel": 7 if expected_cbam else None,
    }
    for key, expected in checks.items():
        if config.get(key) != expected:
            raise ValueError(f"{arm} run_config {key} does not match: {config.get(key)!r}")
    if config.get("actual_split_counts") != EXPECTED_COUNTS:
        # Pandas dict orientation in run_config is split -> label, so compare after transposing.
        actual = config.get("actual_split_counts", {})
        rebuilt = {label: {split: int(actual.get(split, {}).get(label, -1))
                           for split in ("train", "val", "test")}
                   for label in CLASS_NAMES}
        if rebuilt != EXPECTED_COUNTS:
            raise ValueError(f"{arm} split counts do not match the ACM paper.")
    if metrics.get("completed_epochs") != 100 or metrics.get("checkpoint_selection") != "final_epoch":
        raise ValueError(f"{arm} did not report a complete final-epoch evaluation.")
    if metrics.get("split_signature") != config.get("split_signature"):
        raise ValueError(f"{arm} run config and test metrics disagree on the split.")
    if not (run_dir / "final.weights.h5").is_file():
        raise FileNotFoundError(f"{arm} final.weights.h5 is missing from {run_dir}.")
    return config, metrics


def compare_keras_and_litert(model: tf.keras.Model, model_path: Path) -> float:
    """Check numerical parity on deterministic inputs before handing out the export."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=2)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    if (tuple(input_info["shape"]) != (1, 224, 224, 3)
            or tuple(output_info["shape"]) != (1, 3)
            or input_info["dtype"] != np.float32):
        raise ValueError("Exported model has unexpected input/output tensors.")
    maximum_difference = 0.0
    for value in (0.0, 0.5, 1.0):
        image = np.full((1, 224, 224, 3), value, dtype=np.float32)
        expected = model(image, training=False).numpy().reshape(3)
        interpreter.set_tensor(input_info["index"], image)
        interpreter.invoke()
        actual = interpreter.get_tensor(output_info["index"]).reshape(3)
        if not np.isfinite(actual).all() or not np.isclose(actual.sum(), 1.0, atol=0.02):
            raise ValueError("Exported model did not return finite three-class probabilities.")
        maximum_difference = max(maximum_difference, float(np.max(np.abs(expected - actual))))
    if maximum_difference > 0.05:
        raise ValueError(f"Export changed model probabilities by {maximum_difference:.4f}.")
    return maximum_difference


def export_pair(input_dir: Path, output_dir: Path) -> Path:
    runs = find_completed_runs(input_dir)
    validated = {arm: validate_run(path, arm) for arm, path in runs.items()}
    signatures = {config["split_signature"] for config, _ in validated.values()}
    if len(signatures) != 1:
        raise ValueError("Baseline and CBAM outputs used different Preparation splits.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment": "ACM paper paired cacao classification",
        "class_order": CLASS_NAMES,
        "input": {"shape": [1, 224, 224, 3], "orientation": "exif_transpose",
                  "color": "RGB", "resize": "bilinear", "scale": "divide_by_255"},
        "models": {},
    }
    for arm in ("baseline", "cbam"):
        config, metrics = validated[arm]
        print(f"Exporting {arm} from {runs[arm]}", flush=True)
        tf.keras.backend.clear_session()
        gc.collect()
        model = build_model(arm == "cbam", seed=42)
        model.load_weights(str(runs[arm] / "final.weights.h5"))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converted = converter.convert()
        path = output_dir / f"{arm}.tflite"
        path.write_bytes(converted)
        del converted
        export_format = "LiteRT/TFLite float16 weight compression, float32 input/output"
        try:
            max_difference = compare_keras_and_litert(model, path)
        except ValueError as exc:
            print(f"{arm}: compressed export failed parity ({exc}); retrying float32")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            path.write_bytes(converter.convert())
            max_difference = compare_keras_and_litert(model, path)
            export_format = "LiteRT/TFLite float32"
        print(f"{arm}: {path.stat().st_size / (1024**2):.1f} MiB; parity max diff {max_difference:.5f}")
        report = metrics["classification_report"]
        manifest["models"][arm] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "use_cbam": arm == "cbam",
            "seed": 42,
            "split_signature": config["split_signature"],
            "completed_epochs": 100,
            "checkpoint_selection": "final_epoch",
            "format": export_format,
            "conversion_max_abs_difference": max_difference,
            "test_metrics": {
                "test_loss": float(metrics["test_loss"]),
                "test_accuracy": float(metrics["test_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "pod_borer_recall": float(report["pod_borer"]["recall"]),
                "class_metrics": metrics["class_metrics"],
                "confusion_matrix": metrics["confusion_matrix"],
            },
        }
        del model
    manifest_path = output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Model manifest:", manifest_path)
    return manifest_path


# END DEFINITIONS: the companion Kaggle notebook calls export_pair below.

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/streamlit_export"))
    args = parser.parse_args()
    export_pair(args.input, args.output)
