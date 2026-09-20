"""
Cacao Pod Disease and Pest Detection - Prototype
================================================
Thesis: An Improved YOLOv8-CBAM Object Detection Model for Cacao Black Pod Diseases
        and Pest Identification

Demonstrates the Input -> Processing -> Output pipeline using the TensorFlow Lite
model exported in Notebook 4.

SETUP
-----
    pip install streamlit ultralytics pillow
    streamlit run app.py

Put the .tflite files from tflite_export/ in the same folder as this file.
The app finds them automatically.

WHY ULTRALYTICS AND NOT THE RAW TFLITE INTERPRETER
--------------------------------------------------
A YOLOv8 TFLite model outputs a raw tensor with no boxes in it: you have to transpose
it, threshold it, run non-maximum suppression and rescale the coordinates yourself.
Ultralytics' YOLO class loads the same .tflite file and does all of that correctly.
The model running here is still the exported TFLite artifact, which is the thing the
paper claims. Writing the post-processing by hand would add bugs, not credibility.
"""

import io
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).parent
IMGSZ = 640

CLASS_NAMES = ["HEALTHY", "BLACKPOD", "PODBORER", "MIRID"]

DISPLAY_NAMES = {
    "HEALTHY": "Healthy pod",
    "BLACKPOD": "Black Pod Rot",
    "PODBORER": "Cacao Pod Borer",
    "MIRID": "Mirid Bug damage",
}

# ---------------------------------------------------------------------------
# REPLACE THIS TEXT WITH RECOMMENDATIONS FROM A CITED SOURCE.
# The wording below reflects commonly published practice, but your paper must cite
# an authority your panel will accept: the Philippine Coconut Authority, the
# Department of Agriculture, or a peer-reviewed reference. Do not present these as
# findings of your study. Put the citation in the SOURCE field so it shows in the UI.
# ---------------------------------------------------------------------------
ADVICE = {
    "HEALTHY": {
        "status": "No disease or pest damage detected",
        "actions": [
            "Continue routine monitoring on a weekly schedule.",
            "Maintain canopy pruning and field sanitation.",
        ],
        "source": "[ADD CITATION]",
    },
    "BLACKPOD": {
        "status": "Black Pod Rot (Phytophthora palmivora) indicators detected",
        "actions": [
            "Remove infected pods and bury or burn them away from the plot.",
            "Prune to improve airflow and reduce canopy humidity.",
            "Harvest ripe pods frequently so inoculum does not build up.",
            "Apply a copper-based fungicide following label rates if incidence is high.",
        ],
        "source": "[ADD CITATION]",
    },
    "PODBORER": {
        "status": "Cacao Pod Borer (Conopomorpha cramerella) damage detected",
        "actions": [
            "Harvest completely and frequently to break the life cycle.",
            "Sleeve young pods with plastic sleeves where practical.",
            "Bury pod husks immediately after breaking.",
            "Prune to reduce shade and remove alternate hosts nearby.",
        ],
        "source": "[ADD CITATION]",
    },
    "MIRID": {
        "status": "Mirid Bug (Helopeltis spp.) damage detected",
        "actions": [
            "Maintain adequate shade, as mirid pressure rises in exposed canopies.",
            "Prune affected branches and remove chupons.",
            "Inspect flush growth regularly, since mirids feed on young tissue.",
            "Apply a recommended insecticide only if damage passes the action threshold.",
        ],
        "source": "[ADD CITATION]",
    },
}


# ===========================================================================
# Model loading
# ===========================================================================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def find_models():
    """Return {label: path} for every .tflite and .pt next to this file."""
    found = {}
    for p in sorted(APP_DIR.glob("*.tflite")) + sorted(APP_DIR.glob("*.pt")):
        name = p.stem
        tag = "CBAM" if "cbam" in name.lower() else "Baseline"
        fmt = "TFLite" if p.suffix == ".tflite" else "PyTorch"
        prec = ""
        if "float16" in name:
            prec = " FP16"
        elif "float32" in name:
            prec = " FP32"
        found[f"{tag} ({fmt}{prec})"] = str(p)
    return found


# ===========================================================================
# Processing
# ===========================================================================
def detect(model, pil_image, conf, iou):
    t0 = time.perf_counter()
    results = model.predict(pil_image, imgsz=IMGSZ, conf=conf, iou=iou, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r = results[0]

    detections = []
    for box in r.boxes:
        cid = int(box.cls[0])
        detections.append({
            "class_id": cid,
            "class": CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid),
            "confidence": float(box.conf[0]),
            "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    annotated = Image.fromarray(r.plot()[:, :, ::-1])   # BGR to RGB
    return annotated, detections, elapsed_ms


def summarise(detections):
    if not detections:
        return "No pods detected.", []
    counts = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    parts = []
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        best = max(d["confidence"] for d in detections if d["class"] == cls)
        parts.append(f"{DISPLAY_NAMES.get(cls, cls)}: {n} region(s), "
                     f"highest confidence {best:.0%}")
    return " | ".join(parts), sorted(counts, key=lambda c: -counts[c])


# ===========================================================================
# UI
# ===========================================================================
st.set_page_config(page_title="Cacao Pod Detection Prototype",
                   page_icon="*", layout="wide")

st.title("Cacao Pod Disease and Pest Detection")
st.caption("Prototype demonstrating the trained YOLOv8 detection model exported to "
           "TensorFlow Lite. Research prototype, not a diagnostic tool.")

models = find_models()
if not models:
    st.error(
        "No model files found. Put the exported `.tflite` files (or a `.pt` checkpoint) "
        "in the same folder as `app.py`, then reload this page."
    )
    st.stop()

with st.sidebar:
    st.header("Model")
    choice = st.selectbox("Detection model", list(models.keys()))
    model_path = models[choice]
    st.caption(f"`{Path(model_path).name}`")

    compare = False
    if len(models) > 1:
        compare = st.checkbox(
            "Compare two models side by side",
            help="Runs the same image through both so the difference is visible "
                 "rather than described.")
        if compare:
            other = st.selectbox(
                "Second model",
                [k for k in models if k != choice],
                key="second")

    st.header("Detection settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Minimum score for a detection to be shown.")
    iou = st.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.05,
                    help="Overlap above which two boxes are treated as the same pod.")

    st.divider()
    st.caption(f"Input size {IMGSZ}x{IMGSZ}. Classes: {', '.join(CLASS_NAMES)}.")

# ---------------- INPUT ----------------
st.subheader("1. Input")
tab_upload, tab_camera = st.tabs(["Upload an image", "Use the camera"])
with tab_upload:
    uploaded = st.file_uploader("Choose a cacao pod image",
                                type=["jpg", "jpeg", "png", "bmp", "webp"])
with tab_camera:
    captured = st.camera_input("Take a photo")

source = uploaded or captured
if source is None:
    st.info("Upload or capture an image to begin.")
    st.stop()

try:
    image = Image.open(io.BytesIO(source.getvalue())).convert("RGB")
except Exception as e:
    st.error(f"Could not read that file: {e}")
    st.stop()

st.write(f"Input accepted: {image.width} x {image.height} pixels, "
         f"{len(source.getvalue())/1024:.0f} KB")

# ---------------- PROCESSING ----------------
st.subheader("2. Processing")
with st.spinner("Running detection..."):
    try:
        model = load_model(model_path)
        annotated, detections, ms = detect(model, image, conf, iou)
    except Exception as e:
        st.error(f"Inference failed: {type(e).__name__}: {e}")
        st.stop()
st.write(f"Model `{Path(model_path).name}` ran in **{ms:.0f} ms** "
         f"and returned {len(detections)} detection(s) above {conf:.0%} confidence.")

# ---------------- OUTPUT ----------------
st.subheader("3. Output")
if compare:
    other_path = models[other]
    with st.spinner("Running the second model..."):
        annotated2, detections2, ms2 = detect(load_model(other_path), image, conf, iou)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{choice}**")
        st.image(annotated, use_container_width=True)
        st.caption(f"{len(detections)} detections, {ms:.0f} ms")
    with c2:
        st.markdown(f"**{other}**")
        st.image(annotated2, use_container_width=True)
        st.caption(f"{len(detections2)} detections, {ms2:.0f} ms")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original**")
        st.image(image, use_container_width=True)
    with c2:
        st.markdown("**Detected**")
        st.image(annotated, use_container_width=True)

summary, present = summarise(detections)
if detections:
    st.success(summary)
else:
    st.warning("No pods detected above the confidence threshold. Try lowering it in the "
               "sidebar, or use a clearer photo of the pod.")

if detections:
    with st.expander("Detection details", expanded=False):
        st.dataframe(
            [{"#": i + 1,
              "Class": DISPLAY_NAMES.get(d["class"], d["class"]),
              "Confidence": f"{d['confidence']:.1%}",
              "Box (x1, y1, x2, y2)": ", ".join(str(v) for v in d["xyxy"])}
             for i, d in enumerate(detections)],
            hide_index=True, use_container_width=True)

    st.markdown("### Recommended action")
    for cls in present:
        a = ADVICE.get(cls)
        if not a:
            continue
        with st.container(border=True):
            st.markdown(f"**{DISPLAY_NAMES.get(cls, cls)}** - {a['status']}")
            for step in a["actions"]:
                st.markdown(f"- {step}")
            st.caption(f"Source: {a['source']}")

    st.caption("This prototype supports field inspection and does not replace diagnosis "
               "by a qualified agriculturist.")

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history.insert(0, {
    "time": datetime.now().strftime("%H:%M:%S"),
    "model": choice,
    "detections": len(detections),
    "result": summary,
    "ms": round(ms),
})
st.session_state.history = st.session_state.history[:20]

with st.expander(f"Scan history ({len(st.session_state.history)})"):
    st.dataframe(st.session_state.history, hide_index=True, use_container_width=True)
    st.download_button(
        "Download history as JSON",
        json.dumps(st.session_state.history, indent=2),
        file_name="scan_history.json",
        mime="application/json")
