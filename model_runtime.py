"""Training-matched image preparation and verified LiteRT inference."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CLASS_NAMES = ("healthy", "black_pod_rot", "pod_borer")
DISPLAY_NAMES = {
    "healthy": "Healthy pod",
    "black_pod_rot": "Black Pod Rot",
    "pod_borer": "Cacao Pod Borer",
}
INPUT_SHAPE = (1, 224, 224, 3)
MAX_MODEL_BYTES = 2_000_000_000


class ModelArtifactError(RuntimeError):
    """A model file or its metadata does not match the paired ACM experiment."""


def prepare_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """Apply the same EXIF/RGB/bilinear/255 transform as the Kaggle notebooks."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    batch = np.asarray(resized, dtype=np.float32)[None, ...] / 255.0
    if batch.shape != INPUT_SHAPE or not np.isfinite(batch).all():
        raise ValueError("Image did not produce a finite 224×224 RGB tensor.")
    return batch, resized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError("model_manifest.json is unreadable.") from exc
    if manifest.get("schema_version") != 1 or manifest.get("class_order") != list(CLASS_NAMES):
        raise ModelArtifactError("Unsupported manifest schema or class order.")
    transform = manifest.get("input", {})
    expected_transform = {
        "shape": list(INPUT_SHAPE), "orientation": "exif_transpose",
        "color": "RGB", "resize": "bilinear", "scale": "divide_by_255",
    }
    if any(transform.get(key) != value for key, value in expected_transform.items()):
        raise ModelArtifactError("Manifest preprocessing differs from training.")
    models = manifest.get("models", {})
    if set(models) != {"baseline", "cbam"}:
        raise ModelArtifactError("Both baseline and CBAM models are required.")
    signatures = set()
    for arm, spec in models.items():
        digest = spec.get("sha256", "")
        if (spec.get("file") != f"{arm}.tflite" or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ModelArtifactError(f"Invalid {arm} model file or SHA-256.")
        if (spec.get("seed") != 42 or spec.get("completed_epochs") != 100
                or spec.get("checkpoint_selection") != "final_epoch"
                or spec.get("use_cbam") is not (arm == "cbam")):
            raise ModelArtifactError(f"{arm} metadata differs from the ACM training method.")
        signatures.add(spec.get("split_signature"))
    if len(signatures) != 1 or None in signatures:
        raise ModelArtifactError("The two models must share one split signature.")
    return manifest


def _download_model(url: str, destination: Path, expected_sha256: str) -> None:
    if not url.startswith("https://"):
        raise ModelArtifactError("A model download URL must use HTTPS.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    request = urllib.request.Request(url, headers={"User-Agent": "Cacao-Prototype/1.0"})
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_MODEL_BYTES:
                    raise ModelArtifactError("Model download exceeds 2 GB.")
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ModelArtifactError("Downloaded model checksum does not match its manifest.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_model_file(app_dir: Path, arm: str, spec: dict) -> Path:
    """Locate a local model, or download a verified release asset on demand."""
    local = app_dir / "models" / spec["file"]
    if local.is_file():
        if sha256_file(local) != spec["sha256"]:
            raise ModelArtifactError(f"Local {arm} model checksum does not match.")
        return local
    cache_dir = Path(os.environ.get("CACAO_MODEL_CACHE", Path.home() / ".cache" / "cacao-prototype"))
    cached = cache_dir / spec["file"]
    if cached.is_file() and sha256_file(cached) == spec["sha256"]:
        return cached
    url = os.environ.get(f"{arm.upper()}_MODEL_URL")
    if not url:
        raise ModelArtifactError(
            f"{arm} export is missing. Put {spec['file']} in models/ or set {arm.upper()}_MODEL_URL."
        )
    _download_model(url, cached, spec["sha256"])
    return cached


class LiteClassifier:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.lock = threading.Lock()
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ModelArtifactError("Expected one image input and one class output.")
        self.input = inputs[0]
        self.output = outputs[0]
        if tuple(int(v) for v in self.input["shape"]) != INPUT_SHAPE:
            raise ModelArtifactError("Model input is not 1×224×224×3.")
        if tuple(int(v) for v in self.output["shape"]) != (1, 3):
            raise ModelArtifactError("Model output is not three class scores.")
        if self.input["dtype"] not in (np.float32, np.uint8, np.int8):
            raise ModelArtifactError("Unsupported image tensor type.")

    @staticmethod
    def _quantize(values: np.ndarray, detail: dict) -> np.ndarray:
        dtype = detail["dtype"]
        if dtype == np.float32:
            return values.astype(np.float32)
        scale, zero = detail.get("quantization", (0, 0))
        if not scale:
            raise ModelArtifactError("Quantized input has no scale.")
        limits = np.iinfo(dtype)
        return np.clip(np.rint(values / scale + zero), limits.min, limits.max).astype(dtype)

    @staticmethod
    def _dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
        if detail["dtype"] == np.float32:
            return values.astype(np.float32)
        scale, zero = detail.get("quantization", (0, 0))
        if not scale:
            raise ModelArtifactError("Quantized output has no scale.")
        return (values.astype(np.float32) - zero) * scale

    def predict(self, batch: np.ndarray) -> np.ndarray:
        if batch.shape != INPUT_SHAPE or batch.dtype != np.float32:
            raise ValueError("Expected a float32 1×224×224×3 image batch.")
        with self.lock:
            self.interpreter.set_tensor(self.input["index"], self._quantize(batch, self.input))
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output["index"])
        scores = self._dequantize(output, self.output).reshape(3)
        if (not np.isfinite(scores).all() or np.any(scores < -0.001)
                or np.any(scores > 1.001) or not np.isclose(scores.sum(), 1.0, atol=0.02)):
            raise ModelArtifactError("Model did not return a valid three-class softmax vector.")
        return np.clip(scores, 0.0, 1.0)


def load_runtime(path: Path, expected_sha256: str) -> LiteClassifier:
    if sha256_file(path) != expected_sha256:
        raise ModelArtifactError("Model checksum changed before loading.")
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:
        raise ModelArtifactError(
            "LiteRT could not load on this machine. Check its native runtime support "
            "or use the Streamlit Linux deployment."
        ) from exc

    return LiteClassifier(Interpreter(model_path=str(path), num_threads=2))
