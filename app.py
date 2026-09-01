# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportPossiblyUnboundVariable=false
import os
import gc
import streamlit as st
import torch
from PIL import Image

# --- 1. Page Configuration & Custom Styling ---
st.set_page_config(
    page_title="Pestopia | Edge Diagnostics",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .treatment-box {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #c5e1a5;
        color: #1b5e20;
        font-size: 1rem;
        line-height: 1.6;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px;
        color: #856404;
        font-size: 0.9rem;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# --- 2. LAZY LOADING ENGINE ---
@st.cache_resource(show_spinner=False)
def get_engine():
    from src.engine.decision_engine import DecisionEngine
    return DecisionEngine()


# --- 3. CLEAN & INSTANT SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-sun.png", width=80)
    st.title("Pestopia")
    st.divider()

    st.markdown("**System Status**")
    
    disease_path = os.path.join("src", "weights", "best_disease_model.pth")
    pest_path = os.path.join("src", "weights", "best_pest_model.pth")
    
    if os.path.exists(disease_path):
        st.caption("🟢 Disease AI: Ready")
    else:
        st.caption("🔴 Disease AI: Offline (Check Weights)")
        
    if os.path.exists(pest_path):
        st.caption("🟢 Pest AI: Ready")
    else:
        st.caption("🔴 Pest AI: Offline (Check Weights)")
        
    st.divider()
    st.caption("Edge Diagnostic Pipeline")


# --- 4. MAIN UI & INPUT HANDLING ---
st.markdown('<div class="main-header">🌱 Pestopia Diagnostics</div>', unsafe_allow_html=True)
st.write("") 

input_mode = st.radio("Select Input Method:", ["Upload Image", "Use Camera"], horizontal=True)

if input_mode == "Upload Image":
    image_file = st.file_uploader(
        "Upload a leaf or insect photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="image_uploader"
    )
else:
    image_file = st.camera_input("Take a picture", key="camera_input")


# --- 5. INSTANT CACHE PURGE ON REMOVAL ---
def clear_active_cache():
    """Immediately purges previous diagnostic data and frees system RAM/VRAM."""
    keys_to_clear = ['last_result', 'last_image_name', 'health_score']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
            
    # Force Python and PyTorch memory garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if image_file is None:
    # Trigger instant cleanup when no image is loaded
    clear_active_cache()
    st.info("👆 Please provide an image to begin the diagnosis.")

else:
    # Check if a new image was uploaded to clear previous stale states
    current_file_id = getattr(image_file, 'name', 'camera_frame')
    if st.session_state.get('last_image_name') != current_file_id:
        clear_active_cache()
        st.session_state['last_image_name'] = current_file_id

    try:
        input_image = Image.open(image_file).convert("RGB")
    except Exception as e:
        st.error(f"Invalid image format: {e}")
        st.stop()

    st.image(input_image, use_container_width=True, caption="Analyzed Sample")

    # Run Inference
    with st.spinner("Analyzing image features..."):
        engine = get_engine() 
        result = engine.process_image(input_image)
        st.session_state['last_result'] = result

    # Extract Results
    full_detection = result.get('disease', 'Healthy Crop')
    health_score = result.get('score', 10)
    treatment_advice = result.get('advice', 'No immediate chemical intervention required. Continue regular crop monitoring.')

    disease_part = "No disease detected."
    pest_part = "No pests detected."

    if "Healthy" not in full_detection:
        parts = full_detection.split(" & ")
        for p in parts:
            if "Disease" in p: 
                disease_part = p
            if "Pest" in p: 
                pest_part = p
    else:
        disease_part = "Healthy Crop"
        pest_part = "No pests detected."

    # --- 6. SEPARATED TABS ---
    tab1, tab2, tab3 = st.tabs(["🦠 Disease Detection", "🐛 Pest Detection", "💊 Treatment Plan"])

    with tab1:
        st.markdown("### Crop Pathology Analysis")
        if "Disease Class" in disease_part:
            st.warning(f"**Identified:** {disease_part}")
            st.progress(health_score / 10.0)
            st.caption(f"Overall Plant Health Score: {health_score}/10")
        elif "Healthy" in disease_part:
            st.success(f"**Identified:** {disease_part}")
            st.progress(1.0)
            st.caption("Overall Plant Health Score: 10/10")
        else:
            st.info(disease_part)

    with tab2:
        st.markdown("### Entomological Analysis")
        if "Pest Class" in pest_part:
            st.error(f"**Identified:** {pest_part}")
            st.caption("Note: High-resolution texture analysis utilized for pest classification.")
        else:
            st.success(pest_part)

    with tab3:
        st.markdown("### Recommended Action")
        st.markdown(f'<div class="treatment-box">{treatment_advice}</div>', unsafe_allow_html=True)
        
        st.markdown(
            '<div class="disclaimer-box"><strong>⚠️ Disclaimer:</strong> AI models can misidentify visual symptoms. '
            'If the condition is severe, or if the recommended treatment does not match your field observations, '
            'please do not apply chemicals blindly.</div>',
            unsafe_allow_html=True
        )
        
        st.divider()
        st.markdown("#### 🧑‍🌾 Unsure? Consult Human Experts")
        st.write("Redirect to official agricultural portals for verified, localized assistance:")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.link_button("🌐 Locate your nearest KVK", "https://kvk.icar.gov.in/", use_container_width=True)
        with col_btn2:
            st.link_button("📞 Kisan Call Center (1551)", "https://mkisan.gov.in/", use_container_width=True)