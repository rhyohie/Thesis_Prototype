"""Generate the self-contained Kaggle export notebook from the checked-in script."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "tools" / "export_acm_models.py").read_text(encoding="utf-8")
definitions = source.split("# END DEFINITIONS:", 1)[0]


def cell(kind: str, content: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": content.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


notebook = {
    "cells": [
        cell("markdown", """# Export the completed ACM baseline and CBAM models

Run this **after both 100-epoch training notebooks finish**. In a new Kaggle notebook, use **Add Input → Notebook** to attach the saved outputs of the new baseline and CBAM training notebooks. You do not need the raw image dataset here.

Use a Kaggle GPU and **Run All**. This reads each run's `run_config.json`, `test_metrics.json`, and `final.weights.h5`; verifies their methods and shared split; then writes `baseline.tflite`, `cbam.tflite`, and `model_manifest.json` under `/kaggle/working/streamlit_export/`. The export uses the final trained weights. It does not train again or create new results.

Download all three files from the committed notebook output. Keep the two `.tflite` files out of ordinary Git commits; follow the repository README to place them in a public GitHub Release and connect the URLs to Streamlit. The manifest is committed with the application code.
"""),
        cell("code", definitions),
        cell("code", 'export_pair(Path("/kaggle/input"), Path("/kaggle/working/streamlit_export"))\n'),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = ROOT / "export_for_streamlit.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(destination)
