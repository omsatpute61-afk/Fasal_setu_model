# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportPossiblyUnboundVariable=false
import os
import gc
import streamlit as st
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
    .escalation-box {
        background-color: #ffebe6;
        border-left: 5px solid #ff3d00;
        padding: 20px;
        color: #b32a00;
        font-size: 1.05rem;
        border-radius: 4px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
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
    
    disease_path = os.path.join("src", "weights", "disease_model.onnx")
    pest_path = os.path.join("src", "weights", "pest_model.onnx")
    
    if os.path.exists(disease_path):
        st.caption("✅ Disease AI: Ready")
    else:
        st.caption("❌ Disease AI: Offline (Check Weights)")
        
    if os.path.exists(pest_path):
        st.caption("✅ Pest AI: Ready")
    else:
        st.caption("❌ Pest AI: Offline (Check Weights)")
        
    st.divider()
    st.caption("Pure ONNX Edge Diagnostic Pipeline")


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
            
    # Force Python memory garbage collection
    gc.collect()


if image_file is None:
    # Trigger instant cleanup when no image is loaded
    clear_active_cache()
    st.info("⬆️ Please provide an image to begin the diagnosis.")

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
    with st.spinner("Executing ONNX Inference..."):
        engine = get_engine() 
        result = engine.process_image(input_image)
        st.session_state['last_result'] = result

    # Extract Results
    disease_text = str(result.get('disease_text', 'Healthy Crop'))
    pest_text = str(result.get('pest_text', 'No pests detected.'))
    health_score = int(result.get('score', 10))
    disease_escalate = bool(result.get('disease_escalate', False))
    pest_escalate = bool(result.get('pest_escalate', False))

    # --- 6. SEPARATED TABS ---
    tab1, tab2 = st.tabs(["🦠 Disease Detection", "🐛 Pest Detection"])

    escalation_html = (
        '<div class="escalation-box">'
        '<strong>⚠️ ALERT: Low Confidence Diagnosis / Possible Out-of-Distribution Sample</strong><br><br>'
        'The AI engine could not classify this sample with sufficient statistical confidence. '
        'To prevent incorrect agricultural interventions, the system has automatically triggered an escalation. '
        'Please consult a human expert or official agricultural authority before taking action.'
        '</div>'
    )

    with tab1:
        st.markdown("### Crop Pathology Analysis")
        if "Healthy" in disease_text:
            st.success(f"**{disease_text}**")
        elif "Uncertain" in disease_text or disease_escalate:
            st.warning(f"**{disease_text}**")
        else:
            st.error(f"**{disease_text}**")
        
        st.progress(health_score / 10.0)
        st.caption(f"Overall Plant Health Score: {health_score}/10")
        
        if disease_escalate:
            st.markdown(escalation_html, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Entomological Analysis")
        if "No pests" in pest_text:
            st.success(f"**{pest_text}**")
        elif "Uncertain" in pest_text or "Possible" in pest_text or pest_escalate:
            st.warning(f"**{pest_text}**")
        else:
            st.error(f"**{pest_text}**")
        st.caption("Note: High-resolution texture analysis utilized for pest classification.")
        
        if pest_escalate:
            st.markdown(escalation_html, unsafe_allow_html=True)

    # --- 7. CONSULT HUMAN EXPERTS (Always visible at bottom) ---
    st.divider()
    st.markdown("#### 👨‍🌾 Consult Human Experts")
    st.write("Redirect to official agricultural portals for verified, localized assistance:")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.link_button("📍 Locate your nearest KVK", "https://kvk.icar.gov.in/", use_container_width=True)
    with col_btn2:
        st.link_button("📞 Kisan Call Center (1551)", "https://mkisan.gov.in/", use_container_width=True)