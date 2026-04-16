import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_image

# Constants
MODEL_PATH = "models/cnn_model.h5"
IMG_SIZE = 128

# Load model once
@st.cache_resource
def load_cnn():
    return load_model(MODEL_PATH)

model = load_cnn()

st.title("Lung Cancer Detection System 🫁")

uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg", "png", "jpeg"])

# 🔥 NEW: CT Scan Validation Function
def is_ct_scan(img):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check if image is mostly grayscale (low color variance)
        b, g, r = cv2.split(img)
        color_diff = np.mean(np.abs(b - g)) + np.mean(np.abs(g - r))

        # Check intensity distribution (CT scans have mid-range contrast)
        mean_intensity = np.mean(gray)

        # Heuristic rules
        if color_diff > 20:   # too colorful → reject
            return False

        if mean_intensity < 20 or mean_intensity > 230:  # too dark/light
            return False

        return True

    except:
        return False


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is None:
        st.error("Invalid image file ❌")
        st.stop()

    # 🔥 VALIDATION CHECK
    if not is_ct_scan(img):
        st.error("⚠️ Invalid Input: Please upload a lung CT scan image only.")
        st.image(img, caption="Rejected Image", use_column_width=True)
        st.stop()

    # Preprocess
    input_img = preprocess_image(img)

    # Prediction
    prediction = model.predict(input_img)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.error(f"Cancer Detected ❌ (Confidence: {confidence:.2f})")
    else:
        st.success(f"Normal ✅ (Confidence: {1-confidence:.2f})")

    # Show images
    st.image(img, caption="Uploaded CT Image", use_column_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Processed Grayscale Image", use_column_width=True)