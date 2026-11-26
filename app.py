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
min_area = st.sidebar.number_input("Ignore small noise (px²)", min_value=10, value=500, step=100)

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
        "Area (cm²)": round(area_cm2, 3),
        "Length (cm)": round(length_cm, 2),
        "Width (cm)": round(width_cm, 2),
        "Perimeter (cm)": round(perim_cm, 2),
        "Aspect Ratio (L/W
