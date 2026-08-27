import streamlit as st
import cv2
import numpy as np
import time

from src.main import DiagnosticEngine

# Set up page config
st.set_page_config(
    page_title="SIH Edge AI Monitor",
    page_icon="🌱",
    layout="wide"
)

# Initialize the Diagnostic Engine once using caching so it doesn't reload weights every frame
@st.cache_resource
def load_engine():
    return DiagnosticEngine()

engine = load_engine()

st.title("🌱 SIH Edge AI: Crop Health Monitor")
st.markdown("Real-time on-device inference for Disease, Pest, and Nutrient analysis.")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Live Camera Feed")
    run_stream = st.checkbox('Start Live Stream')
    FRAME_WINDOW = st.image([])

with col2:
    st.subheader("Farmer Health Card")
    health_card_placeholder = st.empty()

camera = None

if run_stream:
    # 0 is usually the default built-in webcam. 
    # For Raspberry Pi, it might be 0, or you might need a specific CSI camera pipeline.
    camera = cv2.VideoCapture(0)
    
    # Optional: Lower resolution to speed up inference on weak edge devices
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_counter = 0

while run_stream:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to capture stream from camera. Make sure the webcam is connected and accessible.")
        break
        
    # Convert BGR to RGB for Streamlit rendering
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Render the frame to the UI immediately
    FRAME_WINDOW.image(frame_rgb)
    
    # To prevent UI lag on edge devices, you might only run the heavy AI models every N frames.
    # For this prototype, we'll run it every 5 frames.
    if frame_counter % 5 == 0:
        # Run Edge AI Pipeline
        health_card = engine.analyze_frame(frame)
        
        # Display the health card JSON
        with health_card_placeholder.container():
            status = health_card.get("overall_status", "UNKNOWN")
            
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
            
    frame_counter += 1
    
    # Small sleep to yield execution and prevent freezing the browser
    time.sleep(0.01)

else:
    if camera is not None:
        camera.release()
    st.info('Check "Start Live Stream" to open the camera.')
