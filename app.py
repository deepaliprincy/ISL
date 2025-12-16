import streamlit as st
import numpy as np
import string
from PIL import Image
import random

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="ISL Recognition System",
    page_icon="🤟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ================= CUSTOM CSS =================
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; }
    .stApp { background: transparent; }
    .creator-badge { background: rgba(255, 255, 255, 0.95); padding: 1.5rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.18); }
    .creator-name { font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
    .creator-subtitle { color: #666; font-size: 1rem; margin-top: 0.5rem; }
    .title-container { background: white; padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15); }
    .main-title { font-size: 2.5rem; font-weight: 800; color: #2d3748; margin: 0; line-height: 1.2; }
    .subtitle { color: #718096; font-size: 1.1rem; margin-top: 0.5rem; }
    .upload-section { background: white; padding: 2rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15); }
    .result-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 20px; text-align: center; margin-top: 2rem; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2); }
    .predicted-sign { font-size: 4rem; font-weight: 900; margin: 1rem 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2); }
    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.75rem 2rem; border-radius: 10px; font-weight: 600; font-size: 1rem; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
    </style>
""", unsafe_allow_html=True)

# =============joel==== UI COMPONENTS =================

# Creator Badge
st.markdown("""
    <div class="creator-badge">
        <p class="creator-name">Demo Created by Deepali Yadav</p>
        <p class="creator-subtitle">Indian Sign Language Recognition System</p>
    </div>
""", unsafe_allow_html=True)

# Title Section
st.markdown("""
    <div class="title-container">
        <h1 class="main-title">🤟 ISL Recognition System</h1>
        <p class="subtitle">Upload a hand sign image to get a random prediction</p>
    </div>
""", unsafe_allow_html=True)

# Upload Section
with st.container():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image showing a hand sign gesture"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Main Processing
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Random prediction
    alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)
    label = random.choice(alphabet)
    confidence = random.uniform(50, 100)  # random confidence for demo

    st.markdown(f"""
        <div class="result-box">
            <h2 style="margin: 0; font-size: 1.5rem;">✨ Recognition Result</h2>
            <div class="predicted-sign">{label}</div>
            <p>Confidence: {confidence:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)
