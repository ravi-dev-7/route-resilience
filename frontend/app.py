"""
app.py
Streamlit dashboard for Route Resilience.
Upload a satellite image -> view predicted road mask -> see graph stats
and resilience index from node ablation.
"""

import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(page_title="Route Resilience Dashboard", layout="wide")

API_URL = "http://127.0.0.1:8000"

st.title("🛰️ Route Resilience Dashboard")
st.caption("Occlusion-robust road extraction + topological stress-testing | ISRO Bharatiya Antariksh Hackathon 2026")

st.divider()

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    if st.button("Run Road Extraction", type="primary"):
        with st.spinner("Running segmentation model..."):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{API_URL}/api/predict", files=files, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    mask_bytes = base64.b64decode(result["mask_base64"])
                    mask_image = Image.open(io.BytesIO(mask_bytes))

                    with col2:
                        st.subheader("Predicted Road Mask")
                        st.image(mask_image, use_container_width=True)

                    st.success(f"Road pixels detected: {result['road_pixel_count']}")
                else:
                    st.error(f"API error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend API. Make sure FastAPI server is running on port 8000.")
else:
    st.info("Upload a satellite image to begin.")

st.divider()
st.caption("Phase 1-4 demo | Resilience Index simulation coming in next phase")