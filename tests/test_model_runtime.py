"""Guard the inference boundary between the Kaggle notebooks and Streamlit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from model_runtime import LiteClassifier, ModelArtifactError, load_manifest, prepare_image


def manifest_fixture() -> dict:
    return {
        "schema_version": 1,
        "class_order": ["healthy", "black_pod_rot", "pod_borer"],
        "input": {"shape": [1, 224, 224, 3], "orientation": "exif_transpose",
                  "color": "RGB", "resize": "bilinear", "scale": "divide_by_255"},
        "models": {
            arm: {"file": f"{arm}.tflite", "sha256": "a" * 64,
                  "seed": 42, "completed_epochs": 100,
                  "checkpoint_selection": "final_epoch", "use_cbam": arm == "cbam",
                  "split_signature": "same-preparation-split"}
            for arm in ("baseline", "cbam")
        },
    }


class FakeInterpreter:
    def __init__(self, input_shape=(1, 224, 224, 3), scores=(0.2, 0.3, 0.5)):
        self.input_shape = np.asarray(input_shape)
        self.scores = np.asarray([scores], dtype=np.float32)
        self.received = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"shape": self.input_shape, "dtype": np.float32, "index": 0}]

    def get_output_details(self):
        return [{"shape": np.asarray((1, 3)), "dtype": np.float32, "index": 1}]

    def set_tensor(self, index, value):
        self.received = value

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self.scores


class RuntimeTests(unittest.TestCase):
    def test_preparation_produces_training_input(self):
        batch, preview = prepare_image(Image.new("RGBA", (40, 80), (255, 128, 0, 255)))
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertEqual(preview.mode, "RGB")
        np.testing.assert_allclose(batch[0, 0, 0], [1.0, 128 / 255.0, 0.0])

    def test_manifest_requires_same_split_and_final_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_manifest.json"
            manifest = manifest_fixture()
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIsNotNone(load_manifest(path))
            manifest["models"]["cbam"]["split_signature"] = "another-split"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ModelArtifactError):
                load_manifest(path)
            manifest["models"]["cbam"]["split_signature"] = "same-preparation-split"
            manifest["models"]["baseline"]["completed_epochs"] = 5
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ModelArtifactError):
                load_manifest(path)

    def test_classifier_rejects_old_detector_shape(self):
        with self.assertRaisesRegex(ModelArtifactError, "224"):
            LiteClassifier(FakeInterpreter(input_shape=(1, 640, 640, 3)))

    def test_classifier_uses_three_class_output(self):
        interpreter = FakeInterpreter()
        result = LiteClassifier(interpreter).predict(np.zeros((1, 224, 224, 3), np.float32))
        np.testing.assert_allclose(result, [0.2, 0.3, 0.5], atol=1e-6)
        self.assertEqual(interpreter.received.shape, (1, 224, 224, 3))

    def test_classifier_rejects_invalid_probabilities(self):
        with self.assertRaisesRegex(ModelArtifactError, "softmax"):
            LiteClassifier(FakeInterpreter(scores=(0.0, 0.0, 0.0))).predict(
                np.zeros((1, 224, 224, 3), np.float32)
            )


if __name__ == "__main__":
    unittest.main()
