import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Leaf Morphometry Pro", layout="wide")
st.title("Automated Measurement of Plant Leaf Morphological Traits")

# Initialize session state for scale calibration
if "px_per_cm" not in st.session_state:
    st.session_state.px_per_cm = 100.0

# --- SIDEBAR: SETTINGS & CALIBRATION ---
st.sidebar.header("1. Scale Calibration")
px_per_cm = st.sidebar.number_input(
    "Current scale (pixels per 1 cm)",
    min_value=1.0,
    value=float(st.session_state.px_per_cm),
    step=1.0,
    help="Use the 'Interactive Ruler Calibration' tab or enter manually"
)
st.session_state.px_per_cm = px_per_cm

st.sidebar.header("2. Segmentation Parameters")
auto_thresh = st.sidebar.checkbox("Automatic threshold (Otsu's method)", value=True)
manual_thresh = st.sidebar.slider("Manual threshold", 0, 255, 180, disabled=auto_thresh)
min_area = st.sidebar.number_input("Ignore small noise (px2)", min_value=10, value=500, step=100)

# --- MAIN TABS ---
tab_batch, tab_single, tab_ruler = st.tabs([
    "Batch Processing & Histograms",
    "Single Leaf (ROI Crop)",
    "Interactive Ruler Calibration"
])

# Helper function to analyze a single image array
def analyze_leaf(img_np, scale_px, use_auto, m_thresh, min_a):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if use_auto:
        t_val, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        t_val = m_thresh
        _, mask = cv2.threshold(blurred, m_thresh, 255, cv2.THRESH_BINARY_INV)
        
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_a]
    
    if not valid_contours:
        return None, mask, None, t_val
        
    leaf = max(valid_contours, key=cv2.contourArea)
    area_px = cv2.contourArea(leaf)
    perim_px = cv2.arcLength(leaf, True)
    
    rect = cv2.minAreaRect(leaf)
    box = np.int32(cv2.boxPoints(rect))
    w_px, h_px = rect[1]
    
    area_cm2 = area_px / (scale_px ** 2)
    perim_cm = perim_px / scale_px
    length_cm = max(w_px, h_px) / scale_px
    width_cm = min(w_px, h_px) / scale_px
    aspect_ratio = length_cm / width_cm if width_cm > 0 else 0
    circularity = (4 * np.pi * area_px) / (perim_px ** 2) if perim_px > 0 else 0
    
    result_img = img_np.copy()
    cv2.drawContours(result_img, [leaf], -1, (0, 220, 0), 3)
    cv2.drawContours(result_img, [box], 0, (255, 50, 50), 2)
    
    metrics = {
        "Area (cm2)": round(area_cm2, 3),
        "Length (cm)": round(length_cm, 2),
        "Width (cm)": round(width_cm, 2),
        "Perimeter (cm)": round(perim_cm, 2),
        "Aspect Ratio (L/W)": round(aspect_ratio, 2),
        "Circularity (0-1)": round(circularity, 3)
    }
    return metrics, mask, result_img, t_val

# ==========================================
# TAB 1: BATCH PROCESSING & HISTOGRAMS
# ==========================================
with tab_batch:
    st.subheader("Upload Multiple Herbarium Images")
    uploaded_files = st.file_uploader(
        "Select one or multiple leaf images (JPG, PNG)",
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
            metrics, _, res_img, _ = analyze_leaf(img_np, px_per_cm, auto_thresh, manual_thresh, min_area)
            
            if metrics:
                row = {"File Name": file.name, **metrics}
                results_list.append(row)
                if idx < 4:
                    preview_cols[idx].image(res_img, caption=file.name, use_container_width=True)
            else:
                st.warning(f"Leaf not detected in: {file.name}")
                
        if results_list:
            df_batch = pd.DataFrame(results_list)
            st.subheader("Summary Table")
            st.dataframe(df_batch, use_container_width=True)
            
            # Descriptive statistics for thesis
            st.subheader("Descriptive Statistics (Mean and SD)")
            stats_df = df_batch.drop(columns=["File Name"]).agg(["mean", "std", "min", "max"]).round(3)
            stats_df.index = ["Mean", "Std. Dev (SD)", "Min", "Max"]
            st.dataframe(stats_df, use_container_width=True)
            
            # Histograms / Distributions
            if len(df_batch) > 1:
                st.subheader("Morphological Trait Distributions (Histograms)")
                metric_to_plot = st.selectbox(
                    "Select trait to visualize:",
                    ["Area (cm2)", "Length (cm)", "Width (cm)", "Perimeter (cm)", "Aspect Ratio (L/W)", "Circularity (0-1)"]
                )
                
                counts, bin_edges = np.histogram(df_batch[metric_to_plot], bins="auto")
                bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(counts))]
                hist_df = pd.DataFrame({"Range": bin_labels, "Sample Count": counts}).set_index("Range")
                
                st.bar_chart(hist_df)
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
    st.subheader("Crop Region of Interest (ROI) to Exclude Labels & Rulers")
    single_file = st.file_uploader("Upload herbarium sheet image", type=["jpg", "jpeg", "png"], key="roi_uploader")
    
    if single_file:
        img_roi = Image.open(single_file).convert("RGB")
        st.write("Drag the red box corners to isolate only the target leaf:")
        cropped_img = st_cropper(img_roi, realtime_update=True, box_color="#FF0000", aspect_ratio=None, key="leaf_cropper")
        
        crop_np = np.array(cropped_img)
        metrics, mask, res_img, t_val = analyze_leaf(crop_np, px_per_cm, auto_thresh, manual_thresh, min_area)
        
        if auto_thresh:
            st.caption(f"Detected Otsu threshold: {int(t_val)}")
            
        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.image(cropped_img, caption="1. Cropped ROI", use_container_width=True)
            c2.image(mask, caption="2. Binary Mask", use_container_width=True)
            c3.image(res_img, caption="3. Contour & Bounding Box", use_container_width=True)
            
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
    st.subheader("Calibrate Pixels-per-Centimeter Using a Ruler in the Image")
    st.markdown("1. Upload a photo containing a ruler or scale bar.\n2. Stretch the blue box along a known distance on the ruler (e.g., from **0 cm to 5 cm**).\n3. Enter the real length in cm and click **Apply This Scale to All Measurements**.")
    
    ruler_file = st.file_uploader("Upload image with a ruler", type=["jpg", "jpeg", "png"], key="ruler_uploader")
    
    if ruler_file:
        img_ruler = Image.open(ruler_file).convert("RGB")
        rect_coords = st_cropper(
            img_ruler,
            realtime_update=True,
            box_color="#0066FF",
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
