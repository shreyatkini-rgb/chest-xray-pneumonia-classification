import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🩺",
    layout="centered"
)

IMG_SIZE = (224, 224)
MODEL_PATH = "models/xray_mobilenetv2.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🩺 Chest X-Ray Pneumonia Classifier")
st.write(
    "Upload a chest X-ray and the trained deep-learning model will "
    "classify it as NORMAL or PNEUMONIA."
)

st.warning(
    "Educational demonstration only. This model is not a medical "
    "diagnostic device and must not be used for clinical decisions."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model not found. Run `python train.py --data data/chest_xray` first."
    )
    st.stop()

model = load_model()

uploaded = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    # MobileNetV2 preprocessing: scale pixels to [-1, 1]
    

    probability = float(model.predict(arr, verbose=0)[0][0])

    if probability >= 0.5:
        prediction = "PNEUMONIA"
        confidence = probability
    else:
        prediction = "NORMAL"
        confidence = 1 - probability

    st.subheader("Prediction")
    if prediction == "PNEUMONIA":
        st.error(f"🫁 {prediction}")
    else:
        st.success(f"✅ {prediction}")

    st.metric("Model confidence", f"{confidence * 100:.2f}%")

    st.caption(
        "The confidence shown is the model's probability estimate, "
        "not a clinically validated probability of disease."
    )
