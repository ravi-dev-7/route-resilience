"""
app.py
Streamlit dashboard for Route Resilience - full pipeline:
Segmentation -> Graph Healing -> Resilience Index, all in one click.
"""

import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(page_title="Route Resilience", layout="wide")

API_URL = "http://127.0.0.1:8000"

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    div[data-testid="stImage"] img {
        max-height: 360px;
        object-fit: contain;
    }
    h4 {
        font-size: 0.95rem !important;
        margin-bottom: 0.2rem !important;
    }
    .metric-card {
        background-color: rgba(255,255,255,0.05);
        padding: 0.7rem 0.9rem;
        border-radius: 0.4rem;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-label {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.55);
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 600;
        line-height: 1.1;
    }
    .gatekeeper-row {
        font-size: 0.78rem;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

st.title("Route Resilience")
st.caption("Occlusion-robust road extraction + topological resilience | ISRO Bharatiya Antariksh Hackathon 2026")


def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


upload_col, button_col = st.columns([3, 1])

with upload_col:
    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with button_col:
    st.write("")
    run_button = st.button("▶ Run Full Analysis", type="primary", use_container_width=True, disabled=(uploaded_file is None))

col1, col2, col3 = st.columns([1, 1, 0.9])

mask_image = None
road_pixel_count = None
graph_result = None
sim_result = None

if uploaded_file is not None:
    with col1:
        st.markdown("#### Input Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    if run_button:
        file_bytes = uploaded_file.getvalue()
        file_payload = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}

        with st.spinner("Running segmentation..."):
            try:
                resp1 = requests.post(f"{API_URL}/api/predict", files=file_payload, timeout=60)
                if resp1.status_code == 200:
                    result = resp1.json()
                    mask_bytes = base64.b64decode(result["mask_base64"])
                    mask_image = Image.open(io.BytesIO(mask_bytes))
                    road_pixel_count = result["road_pixel_count"]
                else:
                    st.error(f"Segmentation error: {resp1.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend API. Make sure FastAPI server is running on port 8000.")

        if mask_image is not None:
            with st.spinner("Running graph healing..."):
                try:
                    resp2 = requests.post(f"{API_URL}/api/graph-analysis", files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)}, timeout=60)
                    if resp2.status_code == 200:
                        graph_result = resp2.json()
                    else:
                        st.error(f"Graph analysis error: {resp2.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Graph analysis connection failed.")

            with st.spinner("Computing resilience index..."):
                try:
                    resp3 = requests.post(f"{API_URL}/api/simulate-resilience", files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)}, timeout=60)
                    if resp3.status_code == 200:
                        sim_result = resp3.json()
                    else:
                        st.warning(f"Resilience simulation skipped: {resp3.json().get('message', resp3.text)}")
                except requests.exceptions.ConnectionError:
                    st.error("Resilience simulation connection failed.")

    with col2:
        st.markdown("#### Predicted Road Mask")
        if mask_image is not None:
            st.image(mask_image, use_container_width=True)
        else:
            st.info("Click 'Run Full Analysis' to see the predicted mask.")

    with col3:
        st.markdown("#### Analysis")
        if road_pixel_count is not None:
            metric_card("Road Pixels Detected", f"{road_pixel_count:,}")

        if graph_result is not None:
            metric_card("Graph Nodes / Edges", f"{graph_result['total_nodes']} / {graph_result['total_edges']}")
            metric_card("Bridges Healed (MST)", f"{graph_result['bridges_added']}")

        if sim_result is not None:
            metric_card("Resilience Index", f"{sim_result['overall_resilience_index']}")

            st.markdown("**Top Gatekeeper Nodes**")
            for node_data in sim_result["gatekeeper_nodes"][:5]:
                st.markdown(f"""
                <div class="gatekeeper-row">
                    Node {node_data['node']} — Centrality: {node_data['centrality_score']} |
                    R: {node_data['resilience_index']} |
                    Reachability loss: {node_data['reachability_loss_pct']}%
                </div>
                """, unsafe_allow_html=True)
        elif road_pixel_count is not None:
            st.caption("Resilience metrics unavailable for this image (graph too small or disconnected).")

        if road_pixel_count is None:
            st.caption("Run analysis to see metrics here.")

else:
    st.info("Upload a satellite image to begin.")