import streamlit as st
from PIL import Image
from typing import Any
from src.engine.decision_engine import DecisionEngine

# Page Configuration
st.set_page_config(
    page_title="Crop Diagnostics Engine",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHE THE AI ENGINE ---
@st.cache_resource
def load_engine() -> Any:
    try:
        return DecisionEngine()
    except Exception as e:
        st.error(f"Engine Initialization Failed: {e}")
        return None

def main():
    # Sidebar Setup
    with st.sidebar:
        st.title("⚙️ System Status")
        engine = load_engine()
        
        if engine:
            st.success("✅ Decision Engine: ONLINE")
            st.success("✅ MobileNetV4 (Disease): ONLINE")
            st.warning("⏳ RT-DETR (Pests): Awaiting Weights") 
        else:
            st.error("❌ Engine Offline")
            st.caption("Waiting for Kaggle weights to be placed in `src/weights/`")

        st.markdown("---")
        st.markdown("**Pipeline Components:**")
        st.markdown("- Visual Disease Inference\n- Pest Localization\n- 1-10 Health Scoring\n- RAG Treatment Advisory")

    # Main Dashboard UI
    st.title("🌿 Crop Health Diagnostics Dashboard")
    st.markdown("Upload a leaf scan from the field to generate a localized diagnosis and treatment strategy.")

    # Two-column layout for a clean, responsive UI
    col1, col2 = st.columns([1, 1.2], gap="large")

    # Initialize defaults to prevent "possibly unbound" Pylance errors
    analyze_button = False
    image = None

    with col1:
        st.subheader("1. Field Scan Input")
        uploaded_file = st.file_uploader("Upload leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            # Fixed the deprecated Streamlit parameter here
            st.image(image, caption="Live Camera Feed / Upload", use_container_width=True)
            
            # Action Button
            analyze_button = st.button("Run Diagnostic Pipeline", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Actionable Intelligence")
        
        if uploaded_file is not None and analyze_button and image is not None:
            if engine is None:
                st.error("Cannot run diagnostics. The backend engine is currently missing its model weights.")
            else:
                with st.spinner("Processing image through Vision Models and RAG..."):
                    # Process image
                    results = engine.process_image(image)
                    
                    # Top-Level Metrics
                    metric_col1, metric_col2 = st.columns(2)
                    # Forcing string conversions to satisfy Pylance
                    metric_col1.metric(label="Crop Health Index", value=f"{results.get('score', 'N/A')} / 10")
                    metric_col2.metric(label="Detected Issue", value=str(results.get('disease', 'Unknown')))
                    
                    st.divider()
                    
                    # Detailed RAG Output
                    st.markdown("### 📋 Treatment & Advisory Plan")
                    advice = str(results.get('advice', 'No RAG advice generated.'))
                    st.info(advice)
                    
                    # Expandable raw data for debugging
                    with st.expander("View Raw JSON Output"):
                        st.json(results)
        else:
            st.info("Awaiting image upload to generate report.")

if __name__ == "__main__":
    main()