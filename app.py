

import streamlit as st
import numpy as np
import pandas as pd
import string
from PIL import Image
from tensorflow import keras

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage
from mediapipe import ImageFormat

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
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    .stApp {
        background: transparent;
    }
    .creator-badge {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .creator-name {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .creator-subtitle {
        color: #666;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .title-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2d3748;
        margin: 0;
        line-height: 1.2;
    }
    .subtitle {
        color: #718096;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    }
    .predicted-sign {
        font-size: 4rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    .confidence-text {
        font-size: 1.3rem;
        opacity: 0.95;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .error-card {
        background: rgba(254, 202, 202, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #f56565;
        color: #742a2a;
    }
    </style>
""", unsafe_allow_html=True)


# ================= LOAD MODEL =================
@st.cache_resource
def load_models():
    model = keras.models.load_model("model.h5")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1
    )

    landmarker = vision.HandLandmarker.create_from_options(options)
    return model, landmarker


model, landmarker = load_models()

alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)


# ================= LANDMARK PROCESSING =================
def calc_landmark_list(landmarks, image_width, image_height):
    points = []
    for landmark in landmarks:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        points.append([x, y])
    return points


def pre_process_landmark(landmark_list):
    import copy, itertools
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]

    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y

    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(map(abs, temp))
    return [n / max_value for n in temp]


# ================= UI COMPONENTS =================

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
        <p class="subtitle">Upload a hand sign image to recognize Indian Sign Language gestures</p>
    </div>
""", unsafe_allow_html=True)

# Upload Section
with st.container():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image showing a hand sign gesture"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# Info Section
if not uploaded_file:
    st.markdown("""
        <div class="info-card">
            <h3>📋 Instructions:</h3>
            <ul>
                <li>Upload a clear image of a hand sign gesture</li>
                <li>Ensure good lighting and a clean background</li>
                <li>The hand should be clearly visible in the frame</li>
                <li>System recognizes numbers (1-9) and letters (A-Z)</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ================= MAIN PROCESSING =================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape

    # Display uploaded image
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Processing indicator
    with st.spinner('🔍 Analyzing hand gesture...'):
        # Convert to MediaPipe Image
        mp_image = MPImage(
            image_format=ImageFormat.SRGB,
            data=image_np
        )

        # Detect hand landmarks
        mp_results = landmarker.detect(mp_image)

    if mp_results.hand_landmarks:
        for hand_landmarks in mp_results.hand_landmarks:
            landmark_list = calc_landmark_list(hand_landmarks, w, h)
            processed = pre_process_landmark(landmark_list)

            df = pd.DataFrame(processed).T
            preds = model.predict(df, verbose=0)

            label = alphabet[np.argmax(preds)]
            confidence = np.max(preds) * 100

            # Display results
            st.markdown(f"""
                <div class="result-box">
                    <h2 style="margin: 0; font-size: 1.5rem;">✨ Recognition Result</h2>
                    <div class="predicted-sign">{label}</div>
                    <p class="confidence-text">Confidence: {confidence:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.progress(confidence / 100)

    else:
        st.markdown("""
            <div class="error-card">
                <h3>❌ No Hand Detected</h3>
                <p>Please upload a clearer image with a visible hand gesture. Make sure:</p>
                <ul>
                    <li>The hand is clearly visible</li>
                    <li>There is adequate lighting</li>
                    <li>The background is not too cluttered</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: white; opacity: 0.8; padding: 1rem;">
        <p>Powered by TensorFlow & MediaPipe | Built with ❤️ by Deepali Yadav</p>
    </div>
""", unsafe_allow_html=True)