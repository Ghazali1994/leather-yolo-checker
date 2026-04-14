import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="🧠 Leather Defect Detection (YOLO)", layout="wide")
st.title("🧠 Leather Defect Detection (YOLO)")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # make sure path is correct
    return model

model = load_model()

# -------------------------
# Detection function
# -------------------------
def detect_defects(image: Image.Image, conf_thresh):
    image_np = np.array(image)

    results = model(image_np, conf=conf_thresh)[0]

    annotated = image_np.copy()
    boxes = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        boxes.append((label, conf, x1, y1, x2, y2))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return annotated, boxes

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

    st.image(annotated, caption="Detected Defects", use_column_width=True)

    st.write(f"### 🧪 {len(defects)} defect(s) found")

    for i, (label, conf, x1, y1, x2, y2) in enumerate(defects, 1):
        st.write(
            f"Defect {i}: {label} ({conf:.2f}) → "
            f"Location ({x1},{y1}) Size {x2-x1}x{y2-y1}"
        )
