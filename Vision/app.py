import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.main import DiagnosticEngine

# Set up page config
st.set_page_config(
    page_title="SIH Edge AI Monitor",
    page_icon="🌱",
    layout="wide"
)

# Initialize the Diagnostic Engine once using caching
@st.cache_resource
def load_engine():
    return DiagnosticEngine()

engine = load_engine()

st.title("🌱 SIH Edge AI: Crop Health Monitor")
st.markdown("Take a photo of a crop leaf to analyze Disease, Pest, and Nutrient deficiencies.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Camera Input")
    # st.camera_input is mobile-friendly, uses the device browser's camera API, 
    # and natively allows Android users to switch to the back camera!
    camera_image = st.camera_input("Take a picture of the crop")

with col2:
    st.subheader("Farmer Health Card")
    
    if camera_image is not None:
        # Convert the uploaded Web browser image buffer to an OpenCV frame
        image = Image.open(camera_image)
        frame = np.array(image)
        
        # Convert RGB (Pillow/Web format) to BGR (OpenCV format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Analyzing frame through AI Gatekeeper..."):
            health_card = engine.analyze_frame(frame_bgr)
            
        status = health_card.get("overall_status", "UNKNOWN")
        
        # Display Results
        if status == "REJECTED":
            st.warning(f"Status: {status} - {health_card.get('error', '')} (Time: {health_card.get('inference_time_seconds')}s)")
        elif status == "CRITICAL":
            st.error(f"Status: {status} (Time: {health_card.get('inference_time_seconds')}s)")
        else:
            st.success(f"Status: {status} (Time: {health_card.get('inference_time_seconds')}s)")
            
        # Show recommended actions clearly
        if health_card.get("recommended_actions"):
            st.warning(" | ".join(health_card["recommended_actions"]))
            
        # Show raw JSON output payload
        st.json(health_card)
    else:
        st.info("Waiting for camera input...")
