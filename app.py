"""Streamlit prototype for the paired three-class ACM cacao models."""

from __future__ import annotations

import hashlib
import html
import io
import time
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_runtime import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    ModelArtifactError,
    load_manifest,
    load_runtime,
    prepare_image,
    resolve_model_file,
)


APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ARM_LABELS = {"baseline": "VGG baseline", "cbam": "VGG + CBAM"}
ARM_COLORS = {"baseline": "#1ac8ed", "cbam": "#ff8058"}

st.set_page_config(
    page_title="Cacao Pod | Model Comparison", page_icon="🌱", layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500;600&family=DM+Serif+Display&display=swap');
html,body,[data-testid="stAppViewContainer"]{background:#071116;color:#e8f3f5}
[data-testid="stAppViewContainer"]{background-image:linear-gradient(rgba(83,129,143,.10) 1px,transparent 1px),linear-gradient(90deg,rgba(83,129,143,.10) 1px,transparent 1px);background-size:48px 48px}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent}
[data-testid="stMainBlockContainer"]{max-width:1300px;padding-top:2.2rem}
body,p,button,label{font-family:'DM Sans',sans-serif}
.eyebrow,.section-label,.small-label,.pill,.method-step,.card-kicker,.card-chip,.card-foot{font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.12em}
.eyebrow{color:#87a0a9;font-size:.68rem;margin-bottom:.7rem}
.hero-title{font-family:'DM Serif Display',Georgia,serif;color:#eef8f9;font-size:clamp(2.7rem,5vw,5.5rem);line-height:1.03;margin:0 0 .7rem}
.hero-title .accent{color:#1ac8ed}.hero-copy{color:#a6bac0;max-width:750px;font-size:1.05rem;line-height:1.55}
.hero-rule{height:1px;background:#25404a;margin:1.7rem 0 1.3rem}
.pills{display:flex;flex-wrap:wrap;gap:.55rem;margin:.95rem 0 1.55rem}
.pill{border:1px solid #27424b;color:#a9c1c7;border-radius:4px;padding:.45rem .65rem;background:#0b1a20;font-size:.62rem}
.section-label{color:#b3cbd0;font-size:.73rem;margin:1.45rem 0 .72rem}
.method-bar{display:flex;flex-wrap:wrap;gap:.45rem;margin:1.15rem 0 1.45rem}
.method-step{background:#0c1c22;border:1px solid #25404a;color:#9db7be;padding:.55rem .66rem;border-radius:4px;font-size:.62rem}
.method-arrow{color:#4a7380;align-self:center}
.prediction-card{background:#0b171d;border:1px solid #28424b;border-radius:9px;overflow:hidden;min-height:355px;margin-top:.55rem}
.prediction-card.baseline{border-top:2px solid #1ac8ed}.prediction-card.cbam{border-top:2px solid #ff8058}
.card-head{padding:1.02rem 1.2rem;border-bottom:1px solid #263a43;display:flex;justify-content:space-between;align-items:center}
.card-kicker{color:#afc7cd;font-size:.67rem}.card-chip{border-radius:3px;padding:.25rem .44rem;font-size:.64rem;border:1px solid currentColor}
.baseline .card-chip,.baseline .top-score{color:#1ac8ed}.cbam .card-chip,.cbam .top-score{color:#ff8058}
.card-body{padding:1.38rem 1.2rem 1.5rem}.small-label{color:#68828b;font-size:.64rem;margin-bottom:.45rem}
.top-class{font-family:'DM Serif Display',Georgia,serif;font-size:2.1rem;line-height:1.15;color:#f0f7f8;min-height:2.5rem}
.top-score{font-family:'DM Mono',monospace;font-size:1.48rem;font-weight:600;margin:.2rem 0 1.25rem}
.prob-row{display:grid;grid-template-columns:112px 1fr 55px;gap:.65rem;align-items:center;margin:.78rem 0;font-size:.75rem;color:#adbec3}
.prob-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.prob-track{height:7px;border-radius:10px;background:#21343c;overflow:hidden}.prob-fill{height:100%;border-radius:10px}.prob-value{font-family:'DM Mono',monospace;text-align:right}
.card-foot{border-top:1px solid #263a43;padding:.78rem 1.2rem;font-size:.61rem;color:#79939a}
.empty-card{color:#6e8a93;min-height:220px;display:flex;align-items:center;justify-content:center;font-family:'DM Mono',monospace;font-size:.76rem}
.comparison-note{padding:1rem 1.1rem;border-left:3px solid #2bc8e8;background:#0d2028;color:#c3d7db;border-radius:3px;margin:1.35rem 0}
.warning-note{padding:1rem 1.1rem;border-left:3px solid #ff8058;background:#211a1a;color:#e5c5bb;border-radius:3px;margin:1.2rem 0}
.footer-note{color:#748f98;font-size:.78rem;border-top:1px solid #25404a;padding-top:1rem;margin-top:2.2rem}
.table-scroll{overflow-x:auto;border:1px solid #28424b;border-radius:8px;background:#0b171d}
.class-table-block{padding-bottom:1.4rem}
.metrics-table{border-collapse:collapse;width:100%;min-width:650px;color:#cfdee2;font-size:.83rem}
.metrics-table th,.metrics-table td{padding:.82rem 1rem;border-bottom:1px solid #263a43;text-align:left;white-space:nowrap}
.metrics-table th{font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.09em;color:#80a6b0;font-size:.62rem;background:#11232b}
.metrics-table tr:last-child td{border-bottom:0}
[data-testid="stFileUploader"],[data-testid="stCameraInput"]{background:#0b171d;border:1px solid #28424b;border-radius:8px;padding:.65rem}
div.stButton>button[kind="primary"]{background:#17c5e8;color:#001016;border:0;font-weight:700;border-radius:5px;padding:.7rem 1.35rem}
div.stButton>button[kind="primary"]:hover{background:#5fe0f6;color:#001016}
@media(max-width:680px){.hero-title{font-size:2.8rem}.prob-row{grid-template-columns:93px 1fr 52px}}
</style>
""", unsafe_allow_html=True)


def read_upload(uploaded) -> Image.Image:
    payload = uploaded.getvalue()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose an image smaller than 20 MB.")
    try:
        with Image.open(io.BytesIO(payload)) as candidate:
            candidate.verify()
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("This file could not be read as an image.") from exc
    return image


def prediction_card(arm: str, result: dict | None) -> str:
    label = ARM_LABELS[arm]
    if result is None:
        body = '<div class="empty-card">AWAITING IMAGE + MODEL</div>'
        foot = "Same image · Same preprocessing · Three classes"
    else:
        index = int(result["best_index"])
        scores = result["probabilities"]
        color = ARM_COLORS[arm]
        bars = "".join(
            '<div class="prob-row">'
            f'<div class="prob-name">{html.escape(DISPLAY_NAMES[name])}</div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{float(score)*100:.2f}%;background:{color}"></div></div>'
            f'<div class="prob-value">{float(score)*100:.1f}%</div></div>'
            for name, score in zip(CLASS_NAMES, scores)
        )
        body = (
            '<div class="card-body"><div class="small-label">Predicted class</div>'
            f'<div class="top-class">{html.escape(DISPLAY_NAMES[CLASS_NAMES[index]])}</div>'
            f'<div class="top-score">{float(scores[index])*100:.1f}%</div>{bars}</div>'
        )
        foot = f"Inference {float(result['elapsed_ms']):.0f} ms · image-level prediction"
    return (
        f'<div class="prediction-card {arm}"><div class="card-head">'
        f'<span class="card-kicker">{html.escape(label)}</span>'
        f'<span class="card-chip">{"READY" if result else "PENDING"}</span></div>'
        f'{body}<div class="card-foot">{foot}</div></div>'
    )


def render_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    columns = list(rows[0])
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f'<div class="table-scroll"><table class="metrics-table"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


@st.cache_resource(show_spinner=False)
def cached_runtime(arm: str, path: str, digest: str):
    return load_runtime(Path(path), digest)


st.markdown('<div class="eyebrow">Research prototype / paired model comparison</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title"><span class="accent">Cacao</span> Pod Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-copy">One cacao pod photo, two trained models. Compare the baseline VGG reconstruction with its CBAM-modified counterpart on the same image.</p>', unsafe_allow_html=True)
st.markdown('<div class="pills"><span class="pill">3 image-level classes</span><span class="pill">224 × 224 RGB</span><span class="pill">matched preprocessing</span></div><div class="hero-rule"></div>', unsafe_allow_html=True)

try:
    manifest = load_manifest(APP_DIR / "model_manifest.json")
except ModelArtifactError as exc:
    st.error(f"Model metadata is invalid: {exc}")
    manifest = None

if manifest is not None:
    matrices = [manifest["models"][arm].get("test_metrics", {}).get("confusion_matrix")
                for arm in ("baseline", "cbam")]
    if all(isinstance(matrix, list) and len(matrix) == 3
           and all(len(row) == 3 and row[1] == 0 and row[2] == 0 for row in matrix)
           for matrix in matrices):
        st.warning(
            "Held-out test finding: both saved models predicted Healthy for every test image. "
            "Black Pod Rot and Pod Borer recall were 0%. Treat all photo scores as "
            "research outputs, not reliable disease identification."
        )

st.markdown('<div class="section-label">01 / Input photograph</div>', unsafe_allow_html=True)
input_col, preview_col = st.columns([1, 1], gap="large")
with input_col:
    source_mode = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    if source_mode == "Upload":
        uploaded = st.file_uploader("Choose a cacao pod photo", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"])
    else:
        uploaded = st.camera_input("Take a cacao pod photo")
        st.caption("Uses this device's camera. Allow camera access if your browser asks.")

image = prepared = None
if uploaded is not None:
    try:
        image = read_upload(uploaded)
        prepared, preview = prepare_image(image)
    except ValueError as exc:
        st.error(str(exc))
with preview_col:
    if image is None:
        st.info("Upload or capture one clear cacao pod image to preview the model input.")
    else:
        st.image(preview, caption="Image after the training-matched 224×224 RGB transform", width="stretch")
        st.caption(f"Original image: {image.width} × {image.height} pixels")

st.markdown('<div class="method-bar"><span class="method-step">EXIF orientation</span><span class="method-arrow">→</span><span class="method-step">RGB</span><span class="method-arrow">→</span><span class="method-step">bilinear 224 × 224</span><span class="method-arrow">→</span><span class="method-step">divide pixels by 255</span></div>', unsafe_allow_html=True)

if manifest is None:
    st.markdown('<div class="warning-note"><strong>Model exports pending.</strong> This classifier runs after the two new ACM notebook weights are exported and connected. No result is shown from the older detector models.</div>', unsafe_allow_html=True)

if st.button("Compare models", type="primary", disabled=prepared is None or manifest is None):
    fingerprint = hashlib.sha256(uploaded.getvalue()).hexdigest()
    try:
        with st.spinner("Loading the paired models and classifying the image…"):
            results = {}
            for arm in ("baseline", "cbam"):
                spec = manifest["models"][arm]
                model_file = resolve_model_file(APP_DIR, arm, spec)
                runtime = cached_runtime(arm, str(model_file), spec["sha256"])
                started = time.perf_counter()
                probabilities = runtime.predict(prepared)
                results[arm] = {
                    "probabilities": [float(value) for value in probabilities],
                    "best_index": int(probabilities.argmax()),
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
        st.session_state["comparison"] = {"fingerprint": fingerprint, "results": results}
    except (ModelArtifactError, RuntimeError, OSError, ValueError) as exc:
        st.error(f"Model inference could not complete: {exc}")

fingerprint = hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded is not None else None
saved = st.session_state.get("comparison")
results = saved["results"] if saved and saved["fingerprint"] == fingerprint else None
st.markdown('<div class="section-label">02 / Paired predictions</div>', unsafe_allow_html=True)
left, right = st.columns(2, gap="medium")
with left:
    st.markdown(prediction_card("baseline", results["baseline"] if results else None), unsafe_allow_html=True)
with right:
    st.markdown(prediction_card("cbam", results["cbam"] if results else None), unsafe_allow_html=True)

if results:
    baseline_name = CLASS_NAMES[results["baseline"]["best_index"]]
    cbam_name = CLASS_NAMES[results["cbam"]["best_index"]]
    if baseline_name == cbam_name:
        message = f"Both models selected {DISPLAY_NAMES[baseline_name]}. Compare their class scores above."
    else:
        message = f"The models disagree: baseline selected {DISPLAY_NAMES[baseline_name]}, while CBAM selected {DISPLAY_NAMES[cbam_name]}."
    st.markdown(f'<div class="comparison-note">{html.escape(message)}</div>', unsafe_allow_html=True)
    st.caption("Scores are softmax outputs for one photo. They do not locate symptoms or establish field accuracy.")

if manifest is not None:
    st.markdown('<div class="section-label">03 / Held-out study results</div>', unsafe_allow_html=True)
    metrics = [manifest["models"][arm].get("test_metrics") for arm in ("baseline", "cbam")]
    if all(isinstance(item, dict) for item in metrics):
        rows = [{
            "Model": ARM_LABELS[arm],
            "Test accuracy": f"{100 * float(item['test_accuracy']):.1f}%",
            "Test loss": f"{float(item['test_loss']):.4f}",
            "Macro F1": f"{float(item['macro_f1']):.3f}",
            "Pod Borer recall": f"{100 * float(item['pod_borer_recall']):.1f}%",
        } for arm, item in zip(("baseline", "cbam"), metrics)]
        st.markdown(render_table(rows), unsafe_allow_html=True)
        st.caption("These values come from the saved Kaggle hold-out evaluations, not this uploaded photo.")
        with st.expander("Class-level test results"):
            for arm, item in zip(("baseline", "cbam"), metrics):
                st.markdown(f"**{ARM_LABELS[arm]}**")
                class_rows = [{
                    "Class": DISPLAY_NAMES[row["class"]],
                    "Test images": int(row["support"]),
                    "Precision": f"{100 * float(row['precision']):.1f}%",
                    "Recall / sensitivity": f"{float(row['sensitivity_percent']):.1f}%",
                    "Specificity": f"{float(row['specificity_percent']):.1f}%",
                    "F1": f"{float(row['f1_score']):.3f}",
                } for row in item["class_metrics"]]
                st.markdown(f'<div class="class-table-block">{render_table(class_rows)}</div>', unsafe_allow_html=True)
    else:
        st.caption("Held-out metrics will appear after both Kaggle runs are exported.")

st.markdown('<div class="footer-note">Research prototype · Three-class image classification only · Not a field diagnosis or symptom-localization tool.</div>', unsafe_allow_html=True)
