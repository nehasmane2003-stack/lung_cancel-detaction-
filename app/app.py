import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/cnn_model.h5")

IMG_SIZE = 128

st.title("Lung Cancer Detection System 🫁")

uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Preprocess
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    prediction = model.predict(img_input)

    if prediction[0][0] > 0.5:
        st.error("Cancer Detected ❌")
    else:
        st.success("Normal ✅")

    st.image(img, caption="Uploaded Image", use_column_width=True)