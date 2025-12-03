import streamlit as st
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle

# Load model and label encoder
model = load_model("scratch_detector_cnn.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("Scratch Detector")

# Folder input
folder = st.text_input("Enter folder path of test images:")

if folder and os.path.isdir(folder):
    results = {"file_name": [], "decision": []}
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(('.png','.jpg','.jpeg')):
            img_path = os.path.join(folder, file_name)
            img = load_img(img_path, target_size=(128,128), color_mode='grayscale')
            img_array = img_to_array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)[0][0]
            label = "defect" if pred > 0.5 else "no_defect"
            results["file_name"].append(file_name)
            results["decision"].append(label)
    
    df_results = pd.DataFrame(results)
    st.write(df_results)
else:
    st.write("Please enter a valid folder path.")