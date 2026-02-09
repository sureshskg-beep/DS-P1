import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("models/digit_cnn_model.h5")

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload a digit image (28x28)", type=["png","jpg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28,28))
    img_arr = np.array(img)/255.0
    img_arr = img_arr.reshape(1,28,28,1)

    pred = model.predict(img_arr)
    st.write("Predicted Digit:", np.argmax(pred))
