# Output Images
import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\vn573\\Desktop\\ai-waste-based-management-system-main\\ai-waste-based-management-system-main\\AI_Based_Waste_Management_project-main\\model\\waste_model.h5')

# Define class labels in the same order as in training
class_labels =['cardboard','glass','metal', 'plastic'] 

# Image size used during training
img_size = 150

# Streamlit UI
st.title("Waste Classification App")
st.write("Upload an image of waste to classify it into one of the following categories:")
st.write(", ".join(class_labels))

uploaded_file = st.file_uploader("C:\\Users\\vn573\Desktop\\ai-waste-based-management-system-main\\ai-waste-based-management-system-main\\AI_Based_Waste_Management_project-main\\plasticsample.jpg", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Successfully', width="stretch")

    # Preprocess the image
    image = image.resize((img_size, img_size))
    image_array = np.array(image) / 255.0  # Normalize
    if image_array.shape[-1] == 4:  # Remove alpha channel if present
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.write("### Prediction:")
    st.success(f"This looks like **{predicted_class}**.")

    st.write("### Confidence:")
    for i, label in enumerate(class_labels):
        st.write(f"{label}: {prediction[0][i]*100:.2f}%")
