import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("clean_model.keras")

st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image and the model will tell if it's a cat or a dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
