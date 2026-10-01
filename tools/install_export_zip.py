"""Verify a Kaggle exporter ZIP and install its model artifacts locally.

Usage: python tools/install_export_zip.py C:\\path\\to\\results.zip
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_runtime import load_manifest  # noqa: E402


def install(zip_path: Path) -> None:
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    manifest_path = ROOT / "model_manifest.json"
    manifest_temp = ROOT / "model_manifest.json.download"
    pending = []
    try:
        with zipfile.ZipFile(zip_path) as archive:
            manifest_data = archive.read("streamlit_export/model_manifest.json")
            manifest_temp.write_bytes(manifest_data)
            manifest = load_manifest(manifest_temp)
            if manifest is None:
                raise ValueError("The ZIP has no model manifest.")

            for arm in ("baseline", "cbam"):
                spec = manifest["models"][arm]
                member = f"streamlit_export/{arm}.tflite"
                info = archive.getinfo(member)
                if info.file_size != spec["size_bytes"] or info.file_size > 2_000_000_000:
                    raise ValueError(f"{arm} size differs from the manifest.")
                destination = model_dir / f"{arm}.tflite"
                temporary = model_dir / f"{arm}.download"
                pending.append((temporary, destination))
                digest = hashlib.sha256()
                total = 0
                with archive.open(member) as source, temporary.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        total += len(chunk)
                        digest.update(chunk)
                        target.write(chunk)
                if total != info.file_size or digest.hexdigest() != spec["sha256"]:
                    raise ValueError(f"{arm} SHA-256 verification failed.")
                print(f"Verified {arm}: {total / 1024**2:.1f} MiB, SHA-256 {digest.hexdigest()}")

        for temporary, destination in pending:
            temporary.replace(destination)
        manifest_temp.replace(manifest_path)
        print(f"Installed {manifest_path}")
        print(f"Models installed in {model_dir} (ignored by Git).")
    finally:
        manifest_temp.unlink(missing_ok=True)
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/install_export_zip.py RESULTS_ZIP")
    install(Path(sys.argv[1]))
