import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Video Tracker", layout="wide")

st.title("🎬 AI Object Detection & Tracking Tool")
st.markdown("Developed by **Rukshan Weerasekara** | Creative Technologist")
st.markdown("---")

# --- 2. Load Model ---
# We use st.cache_resource so the model only loads once
@st.cache_resource
def load_model():
    return YOLO('yolov8m.pt') 

model = load_model()

# --- 3. Sidebar Configuration ---
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
st.sidebar.info("This tool uses YOLOv8m for high-accuracy tracking in video production workflows.")

# --- 4. File Uploader ---
uploaded_file = st.file_uploader("Upload an Image or Video", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # --- IMAGE PROCESSING ---
    if file_extension in ['jpg', 'jpeg', 'png']:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original Image", use_container_width=True)
        
        if st.button("Run Detection"):
            with st.spinner("Analyzing image..."):
                results = model.predict(img, conf=conf_threshold)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Processed Image", use_container_width=True)
                st.success("Detection Complete!")

    # --- VIDEO PROCESSING ---
    elif file_extension in ['mp4', 'mov', 'avi']:
        st.video(uploaded_file)
        
        if st.button("Start AI Tracking"):
            # Save uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            st.info("🔄 AI is processing the video. This may take a moment depending on length...")
            
            # Run Tracking
            # results[0].save_dir will show where YOLO saves the output
            results = model.track(source=tfile.name, conf=conf_threshold, persist=True, save=True)
            
            st.success("✅ AI Processing Finished!")
            st.write("For cloud deployment, the processed video is stored in the session directory. Download options are best handled in local environments.")
