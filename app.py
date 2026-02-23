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
@st.cache_resource
def load_model():
    return YOLO('yolov8m.pt') 

model = load_model()

# --- 3. Sidebar Configuration ---
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

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

    # --- VIDEO PROCESSING (FIXED) ---
    elif file_extension in ['mp4', 'mov', 'avi']:
        st.video(uploaded_file)
        
        if st.button("Start AI Tracking"):
            # Create a temporary file with a proper suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tfile:
                tfile.write(uploaded_file.read())
                temp_path = tfile.name # Save the path

            st.info("🔄 AI is processing the video. This may take a moment...")
            
            try:
                # Run Tracking using the saved path
                # We add 'persist=True' to keep tracking IDs
                results = model.track(source=temp_path, conf=conf_threshold, persist=True, save=True)
                
                st.success("✅ AI Processing Finished!")
                st.write("Video processing is complete. Check the 'runs' directory in your GitHub files for the output.")
                
            except Exception as e:
                st.error(f"Error during tracking: {e}")
            finally:
                # Clean up the temporary file after processing
                if os.path.exists(temp_path):
                    os.remove(temp_path)
