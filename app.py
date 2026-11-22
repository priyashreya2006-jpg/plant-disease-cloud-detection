
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("plant_disease_model.keras")

IMG_SIZE = 224

def predict(image):
    if image is None:
        return "Please upload a plant leaf image."

    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100

    disease = class_names[predicted_class]

    return f"""
    ## 🌿 Plant Disease Diagnosis

    **Prediction:** {disease}

    **Confidence:** {confidence:.2f}%

    *AI prediction — please verify with an agricultural expert.*
    """

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Plant Leaf Image"),
    outputs=gr.Markdown(),
    title="🌿 AI Plant Disease Detection System",
    description="Upload a plant leaf image to identify possible diseases."
)

if __name__ == "__main__":
    demo.launch()
