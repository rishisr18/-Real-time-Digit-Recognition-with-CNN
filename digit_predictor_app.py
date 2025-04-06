import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2

# Load pre-trained CNN model
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model('cnn_mnist_model.h5')
    return model

model = load_trained_model()

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.arange(10).reshape(-1, 1))

# Streamlit UI
st.title("🧠 Real-time Digit Recognition with CNN")
st.write("Draw a digit on the canvas below. The CNN will predict the digit as you draw!")

# Side-by-side layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎨 Draw Here")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=20,
        stroke_color="#ffffff",
        background_color="#000000",
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas",
    )

# Utility: Check if canvas has meaningful drawing
def is_canvas_empty(image_data):
    gray = np.array(Image.fromarray(image_data).convert('L'))
    return np.max(gray) < 10

# Utility: Check if drawing is valid (enough white pixels + centered)
def is_valid_digit(image_data):
    gray = np.array(Image.fromarray(image_data).convert('L'))
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours to detect the digit's area
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    x, y, w, h = cv2.boundingRect(contours[0])

    # Rule out tiny or edge drawings
    if w < 30 or h < 30:
        return False
    if x < 10 or y < 10 or x + w > 290 or y + h > 290:
        return False

    return True

# Preprocess and predict
def preprocess_and_predict(image_data):
    img = Image.fromarray(image_data).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Prediction logic
with col2:
    st.subheader("🔍 Prediction")

    if canvas_result.image_data is not None:
        image_data = canvas_result.image_data.astype(np.uint8)

        if is_canvas_empty(image_data):
            st.info("✏️ Draw a digit to see prediction.")
        elif not is_valid_digit(image_data):
            st.warning("⚠️ Please draw a **valid digit** in the **center of the canvas**.")
        else:
            predicted_digit = preprocess_and_predict(image_data)
            st.success(f"Predicted Digit: **{predicted_digit}**")
