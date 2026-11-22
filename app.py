import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 AI Plant Disease Detection System")
st.write("Upload a plant leaf image to identify its disease.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Leaf Image")

   img = image.resize((224, 224))
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    st.success(f"Predicted class: {predicted_class}")
