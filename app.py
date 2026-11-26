import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Herbarium Leaf Morphometry", layout="wide")

# --- CUSTOM BOTANICAL CSS STYLING ---
st.markdown("""
    <style>
    .stApp {
        background-color: #F4F9F4;
        color: #102A1E;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 60%, #40916C 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(27, 67, 50, 0.18);
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        margin: 0 0 0.4rem 0;
        font-size: 2.1rem;
        font-weight: 700;
    }
    .hero-banner p {
        color: #D8F3DC;
        margin: 0;
        font-size: 1.05rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #E6F2E6;
        border-right: 1px solid #B7E4C7;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E6F2E6;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #1B4332;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1B4332 !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #B7E4C7;
        border-left: 5px solid #2D6A4F;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .stButton > button, .stDownloadButton > button {
        background-color: #1B4332;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #2D6A4F;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER BANNER ---
st.markdown("""
    <div class="hero-banner">
        <h1>Herbarium Leaf Morphometry System</h1>
        <p>Automated Computer Vision Pipeline for Plant Leaf Trait Extraction, Calibration & Statistical Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "px_per_cm" not in st.session_state:
    st.session_state.px_per_cm = 100.0

# --- SIDEBAR: SETTINGS & MORPHOLOGY CONTROLS ---
st.sidebar.header("1. Scale Calibration")
px_per_cm = st.sidebar.number_input(
    "Current scale (pixels per 1 cm)",
    min_value=1.0,
    value=float(st.session_state.px_per_cm),
    step=1.0,
    help="Calibrate using the 'Interactive Ruler Calibration' tab or enter manually"
)
st.session_state.px_per_cm = px_per_cm

st.sidebar.header("2. Segmentation & Cleaning")
auto_thresh = st.sidebar.checkbox("Automatic threshold (Otsu's method)", value=True)
manual_thresh = st.sidebar.slider("Manual threshold", 0, 255, 180, disabled=auto_thresh)
petiole_kernel = st.sidebar.slider(
    "Petiole removal / Edge smoothing",
    min_value=0,
    max_value=25,
    value=0,
    step=2,
    help="Removes thin stems/petioles to measure only the leaf blade"
)
close_gaps = st.sidebar.slider(
    "Bridge mounting tape gaps",
    min_value=0,
    max_value=25,
    value=0,
    step=2,
    help="Connects leaf parts separated by herbarium paper strips"
)
min_area = st.sidebar.number_input("Ignore small noise (px2)", min_value=10, value=500, step=100)

# --- MAIN TABS ---
tab_batch, tab_single, tab_ruler = st.tabs([
    "Batch Processing & Histograms",
    "Single Leaf (ROI Crop)",
    "Interactive Ruler Calibration"
])

# --- CORE MORPHOMETRY FUNCTION ---
def analyze_leaf(img_np, scale_px, use_auto, m_thresh, min_a, open_k, close_k):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if use_auto:
        t_val, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        t_val = m_thresh
        _, mask = cv2.threshold(blurred, m_thresh, 255, cv2.THRESH_BINARY_INV)
        
    if close_k > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        
    if open_k > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_a]
    
    if not valid_contours:
        return None, mask, None, t_val
        
    leaf = max(valid_contours, key=cv2.contourArea)
    area_px = cv2.contourArea(leaf)
    perim_px = cv2.arcLength(leaf, True)
    
    hull = cv2.convexHull(leaf)
    hull_area = cv2.contourArea(hull)
    solidity = area_px / hull_area if hull_area > 0 else 0
    
    rect = cv2.minAreaRect(leaf)
    box = np.int32(cv2.boxPoints(rect))
    (cx, cy), (w_px, h_px), angle = rect
    
    area_cm2 = area_px / (scale_px ** 2)
    perim_cm = perim_px / scale_px
    length_cm = max(w_px, h_px) / scale_px
    width_cm = min(w_px, h_px) / scale_px
    aspect_ratio = length_cm / width_cm if width_cm > 0 else 0
    circularity = (4 * np.pi * area_px) / (perim_px ** 2) if perim_px > 0 else 0
    
    result_img = img_np.copy()
    cv2.drawContours(result_img, [leaf], -1, (27, 130, 75), 3)
    cv2.drawContours(result_img, [box], 0, (214, 69, 65), 2)
    
    mid1 = ((box[0] + box[1]) // 2)
    mid2 = ((box[2] + box[3]) // 2)
    mid3 = ((box[1] + box[2]) // 2)
    mid4 = ((box[3] + box[0]) // 2)
    cv2.line(result_img, tuple(mid1), tuple(mid2), (45, 106, 79), 2)
    cv2.line(result_img, tuple(mid3), tuple(mid4), (64, 145, 108), 2)
    
    metrics = {
        "Area (cm2)": round(area_cm2, 3),
        "Length (cm)": round(length_cm, 2),
        "Width (cm)": round(width_cm, 2),
        "Perimeter (cm)": round(perim_cm, 2),
        "Aspect Ratio (L/W)": round(aspect_ratio, 2),
        "Circularity (0-1)": round(circularity, 3),
        "Solidity (0-1)": round(solidity, 3)
    }
    return metrics, mask, result_img, t_val

# ==========================================
# TAB 1: BATCH PROCESSING & HISTOGRAMS
# ==========================================
with tab_batch:
    st.subheader("Batch Herbarium Analysis & Trait Distributions")
    uploaded_files = st.file_uploader(
        "Upload one or multiple herbarium images (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if uploaded_files:
        results_list = []
        preview_cols = st.columns(min(len(uploaded_files), 4))
        
        for idx, file in enumerate(uploaded_files):
            img = Image.open(file).convert("RGB")
            img_np = np.array(img)
            metrics, _, res_img, _ = analyze_leaf(
                img_np, px_per_cm, auto_thresh, manual_thresh, min_area, petiole_kernel, close_gaps
            )
            
            if metrics:
                row = {"File Name": file.name, **metrics}
                results_list.append(row)
                if idx < 4:
                    preview_cols[idx].image(res_img, caption=file.name, use_container_width=True)
            else:
                st.warning(f"Leaf not detected in: {file.name}")
                
        if results_list:
            df_batch = pd.DataFrame(results_list)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Specimens Analyzed", len(df_batch))
            m2.metric("Mean Area", f"{df_batch['Area (cm2)'].mean():.2f} cm2")
            m3.metric("Mean Length", f"{df_batch['Length (cm)'].mean():.2f} cm")
            m4.metric("Mean Width", f"{df_batch['Width (cm)'].mean():.2f} cm")
            
            st.subheader("Morphometric Measurements Table")
            st.dataframe(df_batch, use_container_width=True)
            
            st.subheader("Descriptive Statistics (Mean and SD)")
            stats_df = df_batch.drop(columns=["File Name"]).agg(["mean", "std", "min", "max"]).round(3)
            stats_df.index = ["Mean", "Std. Dev (SD)", "Min", "Max"]
            st.dataframe(stats_df, use_container_width=True)
            
            if len(df_batch) > 1:
                st.subheader("Morphological Trait Distributions (Histograms)")
                metric_to_plot = st.selectbox(
                    "Select trait to visualize:",
                    ["Area (cm2)", "Length (cm)", "Width (cm)", "Perimeter (cm)", "Aspect Ratio (L/W)", "Circularity (0-1)", "Solidity (0-1)"]
                )
                
                counts, bin_edges = np.histogram(df_batch[metric_to_plot], bins="auto")
                bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(counts))]
                hist_df = pd.DataFrame({"Range": bin_labels, "Sample Count": counts}).set_index("Range")
                
                st.bar_chart(hist_df, color="#2D6A4F")
            else:
                st.info("Upload 2 or more images to generate distribution histograms.")
                
            csv_batch = df_batch.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download Batch Results (CSV / Excel)",
                data=csv_batch,
                file_name="batch_leaf_metrics.csv",
                mime="text/csv"
            )

# ==========================================
# TAB 2: SINGLE LEAF WITH ROI CROP
# ==========================================
with tab_single:
    st.subheader("Crop Region of Interest (ROI) to Isolate Target Leaf")
    single_file = st.file_uploader("Upload herbarium sheet image", type=["jpg", "jpeg", "png"], key="roi_uploader")
    
    if single_file:
        img_roi = Image.open(single_file).convert("RGB")
        st.write("Drag the dark green box corners to isolate only the target leaf (excluding labels or scale squares):")
        cropped_img = st_cropper(img_roi, realtime_update=True, box_color="#1B4332", aspect_ratio=None, key="leaf_cropper")
        
        crop_np = np.array(cropped_img)
        metrics, mask, res_img, t_val = analyze_leaf(
            crop_np, px_per_cm, auto_thresh, manual_thresh, min_area, petiole_kernel, close_gaps
        )
        
        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.image(cropped_img, caption="1. Cropped ROI", use_container_width=True)
            c2.image(mask, caption="2. Binary Leaf Mask", use_container_width=True)
            c3.image(res_img, caption="3. Contour, Bounding Box & Axes", use_container_width=True)
            
            df_single = pd.DataFrame([{"File Name": single_file.name, **metrics}])
            st.dataframe(df_single, use_container_width=True)
            
            csv_single = df_single.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download Leaf Metrics (CSV)",
                data=csv_single,
                file_name=f"roi_{single_file.name.split('.')[0]}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No leaf detected in the selected area. Try adjusting threshold settings in the sidebar.")

# ==========================================
# TAB 3: INTERACTIVE RULER CALIBRATION
# ==========================================
with tab_ruler:
    st.subheader("Calibrate Scale Using a Ruler or Reference Marker")
    st.markdown("1. Upload a photo containing a ruler or scale bar.\n2. Align the green box along the known distance.\n3. Enter the real length in cm and click **Apply This Scale to All Measurements**.")
    
    ruler_file = st.file_uploader("Upload image with a ruler", type=["jpg", "jpeg", "png"], key="ruler_uploader")
    
    if ruler_file:
        img_ruler = Image.open(ruler_file).convert("RGB")
        rect_coords = st_cropper(
            img_ruler,
            realtime_update=True,
            box_color="#2D6A4F",
            aspect_ratio=None,
            return_type="box",
            key="ruler_cropper"
        )
        
        box_w = rect_coords["width"]
        box_h = rect_coords["height"]
        measured_px = max(box_w, box_h)
        
        col_r1, col_r2 = st.columns(2)
        real_cm = col_r1.number_input("Real length of selected segment (cm):", min_value=0.1, value=1.0, step=0.5)
        calculated_scale = measured_px / real_cm
        
        col_r2.metric("Calculated Scale", f"{calculated_scale:.2f} px / cm", help=f"Measured {measured_px} pixels across {real_cm} cm")
        
        if st.button("Apply This Scale to All Measurements"):
            st.session_state.px_per_cm = round(calculated_scale, 2)
            st.rerun()
