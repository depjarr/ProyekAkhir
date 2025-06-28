import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# --- Page config ---
st.set_page_config(
    page_title="Vehicle Detection with YOLOv8",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {background-color: #f8fafc;}
    .stButton>button {background-color: #2563eb; color: white;}
    .css-1cpxqw2 {background-color: #f1f5f9;}
    .credit {
        text-align: center;
        font-size: 16px;
        color: #888888;
        margin-top: 40px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Title & Description ---
st.title("🚗 Smart Vehicle Detection")
st.write(
    "Upload your vehicle photo, and this app will automatically detect vehicles using a custom-trained YOLOv8 model."
)

# --- Sidebar settings ---
st.sidebar.header("Detection Settings")
confidence = st.sidebar.slider("Detection Confidence", 0.25, 0.95, 0.5, 0.05)

# --- File uploader ---
uploaded_file = st.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png"])

# --- Load YOLOv8 model ---
@st.cache_resource
def load_model():
    model = YOLO("yolov8s.pt")  # Ensure yolov8s.pt is in the same directory
    return model

model = load_model()

# --- Prediction function ---
def predict(image, conf):
    results = model.predict(image, conf=conf)
    return results[0]

# --- Main logic ---
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting vehicles..."):
        results = predict(np.array(image), conf=confidence)
        result_img = results.plot()  # Visualize bounding boxes
        st.image(result_img, caption="Detection Result", use_container_width=True)

        # Detection details
        st.subheader("Detection Details")
        if len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                cls = model.model.names[int(box.cls)]
                conf_score = float(box.conf)
                st.markdown(
                    f"- **Object {i+1}:** `{cls}` with confidence **{conf_score:.2f}**"
                )
        else:
            st.info("No vehicles detected in this image.")

    st.success("Detection complete!")

st.markdown("---")
st.caption("This app uses a custom-trained YOLOv8 model. For further development or to add more detection classes, please contact the admin.")

# --- Credit line ---
st.markdown(
    '<div class="credit">Create by : <b>Devi Zahra Aulia</b></div>',
    unsafe_allow_html=True
)
