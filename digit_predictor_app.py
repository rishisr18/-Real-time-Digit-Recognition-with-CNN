import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load pre-trained CNN model
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model('cnn_mnist_model.h5')  # Use your trained CNN model path
    return model

model = load_trained_model()

# Dummy encoder (optional)
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.arange(10).reshape(-1, 1))

# Streamlit UI
st.title("🧠 Real-time Digit Recognition with CNN")
st.write("Draw a digit on the canvas below. The CNN will predict the digit as you draw!")

# Create two columns: one for canvas, one for prediction
col1, col2 = st.columns([2, 1])  # Adjust width ratio as needed

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

# Function to check if canvas is empty
def is_canvas_empty(image_data):
    gray_img = np.array(Image.fromarray(image_data).convert('L'))
    return np.max(gray_img) < 10  # Low intensity = empty canvas

# Preprocess and predict
def preprocess_and_predict(image_data):
    img = Image.fromarray(image_data)
    img = img.convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Show prediction in right column
with col2:
    st.subheader("🔍 Prediction")
    if canvas_result.image_data is not None:
        image_data = canvas_result.image_data.astype(np.uint8)

        if is_canvas_empty(image_data):
            st.info("✏️ Draw a digit to see prediction.")
        else:
            predicted_digit = preprocess_and_predict(image_data)
            st.success(f"Predicted Digit: **{predicted_digit}**")
