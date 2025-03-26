import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load models
@st.cache_resource
def load_models():
    plant_model = tf.keras.models.load_model("models/plant_id_model.h5", compile=False)
    banana_model = tf.keras.models.load_model("models/Banana_model.h5", compile=False)
    potato_model = tf.keras.models.load_model("models/Potato_model.h5", compile=False)
    rice_model = tf.keras.models.load_model("models/Rice_model.h5", compile=False)
    tomato_model = tf.keras.models.load_model("models/Tomato_model.h5", compile=False)
    return plant_model, banana_model, potato_model, rice_model, tomato_model

plant_model, banana_model, potato_model, rice_model, tomato_model = load_models()

# Class labels
plant_classes = ['Banana', 'Potato', 'Rice', 'Tomato']
banana_classes = ['boron', 'calcium', 'healthy', 'iron', 'magnesium', 'manganese', 'potassium', 'sulphur', 'zinc']
potato_classes = ['Bacteria', 'Fungi', 'Healthy']
rice_classes = ['BrownSpot', 'Healthy', 'LeafBlast', 'Nitrogen', 'Phosphorus', 'Potassium']
tomato_classes = ['Healthy', 'Yellow_Leaf_Curl_Virus', 'mosaic_virus']

def predict(model, img, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Streamlit UI
st.title("Plant Disease Detection App")
st.write("Upload an image of a plant leaf to detect its disease or deficiency.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=200)
    
    # Preprocess the image
    image = image.resize((224, 224))
    image = np.array(image).astype("uint8")  # Convert to uint8 format
    
    # Step 1: Predict Plant Type
    plant_type, plant_confidence = predict(plant_model, image, plant_classes)
    st.write(f"**Detected Plant:** {plant_type} (Confidence: {plant_confidence}%)")
    
    # Step 2: Predict Disease based on Plant Type
    if plant_type == "Banana":
        disease, disease_confidence = predict(banana_model, image, banana_classes)
    elif plant_type == "Potato":
        disease, disease_confidence = predict(potato_model, image, potato_classes)
    elif plant_type == "Rice":
        disease, disease_confidence = predict(rice_model, image, rice_classes)
    elif plant_type == "Tomato":
        disease, disease_confidence = predict(tomato_model, image, tomato_classes)
    else:
        disease, disease_confidence = "Unknown", 0
    
    st.write(f"**Predicted Disease/Deficiency:** {disease} (Confidence: {disease_confidence}%)")
    
