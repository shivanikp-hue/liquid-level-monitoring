# app.py
# AI Liquid Level Monitoring System
# - Upload Image Prediction
# - Live Webcam Capture
# - Actual Water Percentage Measurement using OpenCV
# - Attractive Streamlit Dashboard UI

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Liquid Level Monitoring",
    page_icon="💧",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
}

.main-title {
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    color: #38bdf8;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #cbd5e1;
    margin-bottom: 2rem;
}

.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}

.result-card {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    padding: 30px;
    border-radius: 24px;
    text-align: center;
    color: white;
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 50px;
    font-size: 14px;
}

[data-testid="stSidebar"] {
    background: #0b1220;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">💧 AI Liquid Level Monitoring</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Measure actual water percentage from images using OpenCV</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Navigation")
mode = st.sidebar.radio(
    "Select Mode",
    ["📤 Upload Image", "📷 Live Webcam"]
)



# ============================================================

st.markdown("---")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def classify_level(percentage):
    if percentage >= 100:
        return "Overflowing"
    elif percentage >= 80:
        return "Full Water Level"
    elif percentage >= 40:
        return "Half Water Level"
    elif percentage >= 5:
        return "Low Water Level"
    else:
        return "Empty Bottle"


def detect_water_level(image):
    """
    Detect approximate water level in a transparent bottle.
    Returns:
        label, percentage, confidence, annotated_image
    """

    # Convert PIL image to NumPy array
    img = np.array(image)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize large images for consistent processing
    max_width = 800
    if img_bgr.shape[1] > max_width:
        scale = max_width / img_bgr.shape[1]
        img_bgr = cv2.resize(
            img_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    output = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    # Region of interest (center area where bottle usually appears)
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    y1 = int(h * 0.05)
    y2 = int(h * 0.95)

    roi = img_bgr[y1:y2, x1:x2]

    if roi.size == 0:
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return "Unknown", 0, 0.0, output_rgb

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Smooth image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect horizontal edges (water line)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_y = np.abs(grad_y)

    # Average edge strength per row
    row_strength = np.mean(grad_y, axis=1)

    # Ignore top and bottom margins
    start = int(len(row_strength) * 0.10)
    end = int(len(row_strength) * 0.95)

    if end <= start:
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return "Unknown", 0, 0.0, output_rgb

    search_region = row_strength[start:end]

    # Find strongest horizontal edge
    water_row_local = np.argmax(search_region)
    water_row = water_row_local + start

    # Confidence based on edge strength ratio
    max_strength = search_region[water_row_local]
    mean_strength = np.mean(search_region) + 1e-6
    confidence = min(99.0, (max_strength / mean_strength) * 20)

    # Calculate fill percentage
    bottle_height = roi.shape[0]
    water_height = bottle_height - water_row
    percentage = int((water_height / bottle_height) * 100)

    # Clamp result
    percentage = max(0, min(110, percentage))

    # Classification label
    label = classify_level(percentage)

    # Draw bottle ROI
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw detected water line (blue)
    cv2.line(
        output,
        (x1, y1 + water_row),
        (x2, y1 + water_row),
        (255, 0, 0),
        3
    )

    # Convert back to RGB
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    return label, percentage, confidence, output_rgb


def analyze_image(image):
    return detect_water_level(image)

# ============================================================
# UPLOAD IMAGE MODE
# ============================================================
if mode == "📤 Upload Image":
    st.header("📤 Upload Bottle Image")

    uploaded_file = st.file_uploader(
        "Choose a transparent bottle image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("🔍 Measuring water level..."):
            label, percentage, confidence, annotated = analyze_image(image)

        col1, col2 = st.columns([1.1, 1])

        with col1:
            st.image(
                annotated,
                caption="Detected Water Line (Blue)",
                use_container_width=True
            )

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"## {label}")
            st.markdown(f"# 💧 {percentage}% Filled")
            st.markdown(f"### 🎯 Confidence: {confidence:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

            st.progress(min(percentage, 100) / 100)

# ============================================================
# LIVE WEBCAM MODE
# ============================================================
elif mode == "📷 Live Webcam":
    st.header("📷 Live Webcam Detection")

    st.info(
        "Capture a bottle image using your webcam and the system will "
        "measure the actual water level."
    )

    camera_image = st.camera_input("📸 Take a Picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

        with st.spinner("🔍 Measuring water level..."):
            label, percentage, confidence, annotated = analyze_image(image)

        col1, col2 = st.columns([1.1, 1])

        with col1:
            st.image(
                annotated,
                caption="Detected Water Line (Blue)",
                use_container_width=True
            )

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"## {label}")
            st.markdown(f"# 💧 {percentage}% Filled")
            st.markdown(f"### 🎯 Confidence: {confidence:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

            st.progress(min(percentage, 100) / 100)

# ============================================================


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        🚀 Built with OpenCV and Streamlit<br>
    
    </div>
    """,
    unsafe_allow_html=True
)