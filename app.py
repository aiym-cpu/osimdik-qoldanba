import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Leaf Morphometry", layout="wide")
st.title("Automated Measurement of Plant Leaf Morphological Traits")

# Sidebar settings
st.sidebar.header("1. Scale Calibration")
px_per_cm = st.sidebar.number_input(
    "Pixels per 1 cm", 
    min_value=1.0, 
    value=100.0, 
    step=1.0,
    help="Measure the length of a 1 cm reference scale (ruler) in pixels"
)

st.sidebar.header("2. Segmentation Parameters")
auto_thresh = st.sidebar.checkbox("Automatic threshold (Otsu's method)", value=True)
manual_thresh = st.sidebar.slider("Manual threshold (if auto fails)", 0, 255, 180, disabled=auto_thresh)
min_area = st.sidebar.number_input("Ignore small noise (px²)", min_value=10, value=500, step=100)

uploaded_file = st.file_uploader("Upload herbarium specimen image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    
    # Preprocessing: grayscale + Gaussian blur to reduce dust and paper texture
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Segmentation (separating dark leaf from light herbarium sheet)
    if auto_thresh:
        thresh_val, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        st.sidebar.info(f"Auto-threshold value: {int(thresh_val)}")
    else:
        _, mask = cv2.threshold(blurred, manual_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Contour detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if valid_contours:
        # Select the largest contour (leaf blade)
        leaf = max(valid_contours, key=cv2.contourArea)
        
        # Basic measurements in pixels
        area_px = cv2.contourArea(leaf)
        perim_px = cv2.arcLength(leaf, True)
        
        # Rotated bounding box (accurately measures length and width at any angle)
        rect = cv2.minAreaRect(leaf)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        w_px, h_px = rect[1]
        
        # Convert to centimeters
        area_cm2 = area_px / (px_per_cm ** 2)
        perim_cm = perim_px / px_per_cm
        length_cm = max(w_px, h_px) / px_per_cm
        width_cm = min(w_px, h_px) / px_per_cm
        
        # Morphological indices
        aspect_ratio = length_cm / width_cm if width_cm > 0 else 0
        circularity = (4 * np.pi * area_px) / (perim_px ** 2) if perim_px > 0 else 0
        
        # Visualization
        result_img = img_np.copy()
        cv2.drawContours(result_img, [leaf], -1, (0, 220, 0), 3)      # Green leaf contour
        cv2.drawContours(result_img, [box], 0, (255, 50, 50), 2)      # Red bounding box
        
        col1, col2, col3 = st.columns(3)
        col1.image(image, caption="1. Original Image", use_container_width=True)
        col2.image(mask, caption="2. Binary Mask (Background/Leaf)", use_container_width=True)
        col3.image(result_img, caption="3. Contour & Bounding Box", use_container_width=True)
        
        st.subheader("Morphometric Analysis Results")
        df = pd.DataFrame([{
            "File Name": uploaded_file.name,
            "Area (cm²)": round(area_cm2, 3),
            "Length (cm)": round(length_cm, 2),
            "Width (cm)": round(width_cm, 2),
            "Perimeter (cm)": round(perim_cm, 2),
            "Aspect Ratio (L/W)": round(aspect_ratio, 2),
            "Circularity (0–1)": round(circularity, 3)
        }])
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Download Results (CSV / Excel)",
            data=csv,
            file_name=f"metrics_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No leaf detected. Try disabling auto-threshold and adjusting the slider manually.")
