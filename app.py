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
st.markdown("Take a photo or upload an image from your gallery to analyze Disease, Pest, and Nutrient deficiencies.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Image Input")
    
    # Allow farmers to either use the camera or upload from their gallery
    input_method = st.radio("Select Input Method:", ["Camera", "Gallery Upload"])
    
    image_data = None
    if input_method == "Camera":
        image_data = st.camera_input("Take a picture of the crop")
    else:
        image_data = st.file_uploader("Upload a crop image from your gallery", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("Farmer Health Card")
    
    if image_data is not None:
        # Convert the uploaded Web browser image buffer to an OpenCV frame
        image = Image.open(image_data)
        frame = np.array(image)
        
        # Convert RGB (Pillow/Web format) to BGR (OpenCV format)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            # Handle PNGs with alpha channel
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
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
            
        # Display each detection separately to match the new Fine-Grained schema
        if "detections" in health_card and health_card["detections"]:
            st.markdown("### 🔬 Detailed Taxonomic Analysis")
            for det in health_card["detections"]:
                diag = det["diagnosis"]
                plan = det["treatment_plan"]
                
                with st.expander(f"{det['detection_type'].capitalize()}: {diag['common_name']}"):
                    st.write(f"**Scientific Name:** *{diag['scientific_name']}*")
                    st.write(f"**Confidence:** {diag['confidence_score']}")
                    if diag.get("bounding_box_coordinates"):
                        st.write(f"**Location:** {diag['bounding_box_coordinates']}")
                        
                    st.markdown("#### Treatment Plan")
                    st.write(f"**Urgency:** {plan['urgency_level']}")
                    st.write(f"**Organic Control:** {plan['organic_control']}")
                    st.write(f"**Chemical Control:** {plan['chemical_control']}")
        
        # Show raw JSON output payload
        st.markdown("### Raw Output Payload")
        st.json(health_card)
    else:
        st.info("Waiting for image input...")
