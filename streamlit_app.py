import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd
import numpy as np

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="NanbaProject - AI Plant Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. STYLE & CSS (Green Theme) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #2e7d32; /* Dark Green */
        font-family: 'Arial', sans-serif;
    }
    
    /* Metrics Box */
    .metric-card {
        background-color: #f1f8e9;
        border: 1px solid #c5e1a5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1b5e20;
    }
    .metric-label {
        font-size: 16px;
        color: #555;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #e8f5e9;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. NAVIGATION (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.title("NanbaProject")
    st.markdown("---")
    
    # Menu Options
    page = st.radio(
        "Navigation", 
        ["🏠 Home (Overview)", "📖 Methodology", "📊 Performance", "🚀 Live Simulation"],
        index=0
    )
    
    st.markdown("---")
    st.info("Developed by: **Muthusamy A**\nFinal Year Project")

# --- 4. LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)

model = load_model()

# ==========================================
# PAGE 1: HOME (OVERVIEW)
# ==========================================
if page == "🏠 Home (Overview)":
    st.title("AI-Powered Plant Doctor 🌿")
    st.markdown("""
    ### புரட்சிகரமான விவசாய தொழில்நுட்பம்
    **Nanba Project** என்பது **YOLOv8 (You Only Look Once)** தொழில்நுட்பத்தைப் பயன்படுத்தி, 
    பயிர் நோய்களை நொடிப்பொழுதில் கண்டறியும் ஒரு நவீன செயற்கை நுண்ணறிவு (AI) தளமாகும்.
    """)
    
    st.write("---")
    
    # Metrics Rows (Like your screenshot)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">99.5%</div>
            <div class="metric-label">Model Accuracy (mAP)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">54,300+</div>
            <div class="metric-label">Dataset Size (Images)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">< 20ms</div>
            <div class="metric-label">Inference Time (Speed)</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    st.image("https://ultralytics.com/static/blog/yolov8-state-of-the-art-performance.jpg", caption="YOLOv8 Architecture Overview", use_column_width=True)

# ==========================================
# PAGE 2: METHODOLOGY
# ==========================================
elif page == "📖 Methodology":
    st.title("🔬 Research Methodology")
    st.write("எங்கள் ப்ராஜெக்ட் செயல்படும் விதம்:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("1. Data Collection")
        st.write("50,000+ படங்கள் PlantVillage தரவுத்தொகுப்பில் இருந்து சேகரிக்கப்பட்டன.")
    with col2:
        st.warning("2. Preprocessing")
        st.write("படங்கள் 640x640 அளவுக்கு மாற்றப்பட்டு, Noise நீக்கப்பட்டது.")
    with col3:
        st.success("3. YOLOv8 Training")
        st.write("Google Colab T4 GPU-வில் 50 Epochs வரை பயிற்சி அளிக்கப்பட்டது.")
    with col4:
        st.error("4. Deployment")
        st.write("Streamlit Cloud மூலம் விவசாயிகளின் பயன்பாட்டிற்கு கொண்டுவரப்பட்டது.")

# ==========================================
# PAGE 3: PERFORMANCE
# ==========================================
elif page == "📊 Performance":
    st.title("📈 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Comparison")
        # Creating a fake dataframe for the chart based on your screenshot
        data = pd.DataFrame({
            'Model': ['Nanba (YOLOv8)', 'Custom CNN', 'VGG16'],
            'Accuracy': [99.5, 92.1, 96.8]
        })
        st.bar_chart(data.set_index('Model'))
        st.caption("YOLOv8 மற்ற மாடல்களை விட அதிக துல்லியம் (99.5%) தருகிறது.")

    with col2:
        st.subheader("Inference Speed (Time taken)")
        speed_data = pd.DataFrame({
            'Model': ['Nanba (YOLOv8)', 'Custom CNN', 'VGG16'],
            'Time (ms)': [15, 340, 800]
        })
        st.line_chart(speed_data.set_index('Model'))
        st.caption("YOLOv8 மிக மிக வேகமாக (15ms) செயல்படுகிறது.")

# ==========================================
# PAGE 4: LIVE SIMULATION (THE SCANNER)
# ==========================================
elif page == "🚀 Live Simulation":
    st.title("🌿 Live Disease Detection")
    st.write("இலையின் படத்தை கீழே பதிவேற்றம் செய்து பரிசோதிக்கவும்.")
    
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
            
        with col2:
            if st.button("🔍 Analyze Now", type="primary"):
                if model:
                    with st.spinner("AI is Scanning..."):
                        results = model(image, conf=0.4)
                        
                        if len(results[0].boxes) == 0:
                            st.warning("⚠️ No Disease Detected / Unknown Leaf")
                        else:
                            st.success("✅ Detection Successful!")
                            res_plotted = results[0].plot()
                            st.image(res_plotted, use_column_width=True, caption="YOLOv8 Prediction")
                            
                            # Show Details
                            for box in results[0].boxes:
                                name = model.names[int(box.cls[0])]
                                conf = float(box.conf[0]) * 100
                                st.write(f"🩺 **Diagnosis:** {name}")
                                st.write(f"🎯 **Confidence:** {conf:.2f}%")
