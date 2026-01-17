import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model (ABSOLUTE PATH)
MODEL_PATH =r"D:\AI IMAGE DETECTOR\ai_vs_real_model.h5"
model = load_model(MODEL_PATH)

# Preprocessing for MobileNetV2 (160 × 160)
def preprocess(img):
    img = img.resize((160, 160))
    img_arr = image.img_to_array(img)
    img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# UI
st.title(" AI vs Real Image Detector")
st.write("Upload an image and I will tell whether it's **AI Generated** or **Real Human**.")

uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=250)


    img_arr = preprocess(img)

    prediction = model.predict(img_arr)[0][0]

    if prediction > 0.5:
        label = "AI Generated Image"
        confidence = prediction * 100
    else:
        label = " Real Human Image"
        confidence = (1 - prediction) * 100

    st.subheader(" Prediction Result")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}%")


   





