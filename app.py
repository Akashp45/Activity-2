import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = tf.keras.models.load_model('cnn_model.h5')

# Preprocess image
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image

st.title("Digit Classification with CNN")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)
    st.write(f"Predicted Class: {predicted_class[0]}")
