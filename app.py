import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
from collections import Counter

st.set_page_config(page_title="🧠 Leather Defect Detection (YOLO)", layout="wide")
st.title("🧠 Leather Defect Detection (YOLO)")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # make sure best.pt is in repo
    return model

model = load_model()

# -------------------------
# Detection function
# -------------------------
def detect_defects(image: Image.Image, conf_thresh):
    image_np = np.array(image)

    results = model(image_np, conf=conf_thresh)[0]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        detections.append((label, conf, x1, y1, x2, y2))

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

        # Draw label
        draw.text((x1, y1 - 10), f"{label} {conf:.2f}", fill="lime")

    return annotated, detections

# -------------------------
# UI
# -------------------------
option = st.radio("Choose Input", ["Upload Image", "Camera"])
conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

image = None

if option == "Upload Image":
    file = st.file_uploader("Upload leather image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
else:
    cam = st.camera_input("Capture")
    if cam:
        image = Image.open(cam).convert("RGB")

# -------------------------
# Run detection
# -------------------------
if image is not None:
    annotated, defects = detect_defects(image, conf_thresh)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_column_width=True)

    with col2:
        st.image(annotated, caption="Detected Defects", use_column_width=True)

    st.write(f"### 🧪 {len(defects)} defect(s) found")

    # Detailed results
    for i, (label, conf, x1, y1, x2, y2) in enumerate(defects, 1):
        st.write(
            f"Defect {i}: {label} ({conf:.2f}) → "
            f"Location ({x1},{y1}) Size {x2-x1}x{y2-y1}"
        )

    # -------------------------
    # Summary
    # -------------------------
    labels = [d[0] for d in defects]
    counts = Counter(labels)

    st.write("### 📊 Defect Summary")
    st.write(dict(counts))

    # -------------------------
    # Download result
    # -------------------------
    import io

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Result Image",
        data=byte_im,
        file_name="detected_defects.jpg",
        mime="image/jpeg"
    )
