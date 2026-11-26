import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Гербарий морфометриясы", layout="wide")

# --- АКАДЕМИЯЛЫҚ ДИЗАЙН: TIMES NEW ROMAN, АҚ ЖӘНЕ ҚАРА ЖАСЫЛ ---
st.markdown("""
    <style>
    /* Барлық элементтерге Times New Roman шрифтін орнату */
    html, body, [class*="css"], .stApp, h1, h2, h3, h4, p, div, span, label, input, button, table {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* Негізгі таза ақ фон */
    .stApp {
        background-color: #FFFFFF;
        color: #111111;
    }
    
    /* Жоғарғы баннер — таза қара жасыл және ақ мәтін */
    .hero-banner {
        background-color: #1B4332;
        padding: 1.8rem 2.2rem;
        border-radius: 10px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        border: 1px solid #1B4332;
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        margin: 0 0 0.4rem 0;
        font-size: 2.0rem;
        font-weight: 700;
    }
    .hero-banner p {
        color: #FFFFFF;
        opacity: 0.92;
        margin: 0;
        font-size: 1.1rem;
    }
    
    /* Бүйірлік панель — ақ фон, жіңішке жиек */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Қойындылар (Tabs) — ақ және қара жасыл */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
        border-bottom: 2px solid #1B4332;
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #1B4332;
        border-radius: 6px;
        color: #1B4332;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1B4332 !important;
        color: #FFFFFF !important;
    }
    
    /* Көрсеткіштер карточкасы */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D9D9D9;
        border-left: 5px solid #1B4332;
        padding: 12px 16px;
        border-radius: 6px;
    }
    
    /* Батырмалар */
    .stButton > button, .stDownloadButton > button {
        background-color: #1B4332;
        color: #FFFFFF;
        border-radius: 6px;
        border: 1px solid #1B4332;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        font-size: 1rem;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #122E22;
        color: #FFFFFF;
        border-color: #122E22;
    }
    </style>
""", unsafe_allow_html=True)

# --- ЖОҒАРҒЫ БАННЕР ---
st.markdown("""
    <div class="hero-banner">
        <h1>Өсімдік жапырақтарының морфологиялық белгілерін автоматтандырылған өлшеу</h1>
        <p>Гербарий үлгілерінің бейнелері бойынша морфометриялық талдау, калибрлеу және статистикалық өңдеу жүйесі</p>
    </div>
""", unsafe_allow_html=True)

# Масштаб күйін сақтау
if "px_per_cm" not in st.session_state:
    st.session_state.px_per_cm = 100.0

# --- БҮЙІРЛІК ПАНЕЛЬ: БАПТАУЛАР ---
st.sidebar.header("1. Масштабты калибрлеу")
px_per_cm = st.sidebar.number_input(
    "Ағымдағы масштаб (1 см-дегі пиксель саны)",
    min_value=1.0,
    value=float(st.session_state.px_per_cm),
    step=1.0,
    help="«Интерактивті сызғыш» қойындысы арқылы өлшеңіз немесе қолмен енгізіңіз"
)
st.session_state.px_per_cm = px_per_cm

st.sidebar.header("2. Сегментация және тазарту")
auto_thresh = st.sidebar.checkbox("Автоматты табалдырық (Оцу әдісі)", value=True)
manual_thresh = st.sidebar.slider("Қолмен табалдырық (авто-режим өшірулі болса)", 0, 255, 180, disabled=auto_thresh)
petiole_kernel = st.sidebar.slider(
    "Сағақты алып тастау / жиекті тегістеу",
    min_value=0,
    max_value=25,
    value=0,
    step=2,
    help="Тек жапырақ тақтасын өлшеу үшін жіңішке сағақты жояды"
)
close_gaps = st.sidebar.slider(
    "Жапсырма жолақтардың үзілісін біріктіру",
    min_value=0,
    max_value=25,
    value=0,
    step=2,
    help="Гербарий қағазындағы бекіткіш жолақтар бөлген жапырақ бөліктерін қосады"
)
min_area = st.sidebar.number_input("Ұсақ шуды елемеу (пикс²)", min_value=10, value=500, step=100)

# --- НЕГІЗГІ ҚОЙЫНДЫЛАР ---
tab_batch, tab_single, tab_ruler = st.tabs([
    "Топтық өңдеу және гистограммалар",
    "Жеке жапырақ (Аймақты қию / ROI)",
    "Интерактивті сызғышпен калибрлеу"
])

# --- НЕГІЗГІ МОРФОМЕТРИЯ ФУНКЦИЯСЫ ---
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
    cv2.drawContours(result_img, [leaf], -1, (27, 67, 50), 3)       # Қара жасыл контур
    cv2.drawContours(result_img, [box], 0, (180, 35, 35), 2)       # Жиектеме рамкасы
    
    mid1 = ((box[0] + box[1]) // 2)
    mid2 = ((box[2] + box[3]) // 2)
    mid3 = ((box[1] + box[2]) // 2)
    mid4 = ((box[3] + box[0]) // 2)
    cv2.line(result_img, tuple(mid1), tuple(mid2), (27, 67, 50), 2)
    cv2.line(result_img, tuple(mid3), tuple(mid4), (27, 67, 50), 2)
    
    metrics = {
        "Ауданы (см²)": round(area_cm2, 3),
        "Ұзындығы (см)": round(length_cm, 2),
        "Ені (см)": round(width_cm, 2),
        "Периметрі (см)": round(perim_cm, 2),
        "Созылыңқылық (Ұ/Е)": round(aspect_ratio, 2),
        "Дөңгелектік (0-1)": round(circularity, 3),
        "Тұтастық / Solidity (0-1)": round(solidity, 3)
    }
    return metrics, mask, result_img, t_val

# ==========================================
# 1-ҚОЙЫНДЫ: ТОПТЫҚ ӨҢДЕУ ЖӘНЕ ГИСТОГРАММАЛАР
# ==========================================
with tab_batch:
    st.subheader("Гербарий үлгілерін топтық талдау және үлестірім гистограммалары")
    uploaded_files = st.file_uploader(
        "Бір немесе бірнеше гербарий суретін жүктеңіз (JPG, PNG)",
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
                row = {"Файл атауы": file.name, **metrics}
                results_list.append(row)
                if idx < 4:
                    preview_cols[idx].image(res_img, caption=file.name, use_container_width=True)
            else:
                st.warning(f"Мына суреттен жапырақ табылмады: {file.name}")
                
        if results_list:
            df_batch = pd.DataFrame(results_list)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Талданған үлгілер", len(df_batch))
            m2.metric("Орташа ауданы", f"{df_batch['Ауданы (см²)'].mean():.2f} см²")
            m3.metric("Орташа ұзындығы", f"{df_batch['Ұзындығы (см)'].mean():.2f} см")
            m4.metric("Орташа ені", f"{df_batch['Ені (см)'].mean():.2f} см")
            
            st.subheader("Морфометриялық өлшемдер кестесі")
            st.dataframe(df_batch, use_container_width=True)
            
            st.subheader("Сипаттамалық статистика (Орташа мән және SD)")
            stats_df = df_batch.drop(columns=["Файл атауы"]).agg(["mean", "std", "min", "max"]).round(3)
            stats_df.index = ["Орташа мән (Mean)", "Стандартты ауытқу (SD)", "Минимум (Min)", "Максимум (Max)"]
            st.dataframe(stats_df, use_container_width=True)
            
            if len(df_batch) > 1:
                st.subheader("Морфологиялық белгілердің үлестірім гистограммасы")
                metric_to_plot = st.selectbox(
                    "График құру үшін белгіні таңдаңыз:",
                    [
                        "Ауданы (см²)", 
                        "Ұзындығы (см)", 
                        "Ені (см)", 
                        "Периметрі (см)", 
                        "Созылыңқылық (Ұ/Е)", 
                        "Дөңгелектік (0-1)", 
                        "Тұтастық / Solidity (0-1)"
                    ]
                )
                
                counts, bin_edges = np.histogram(df_batch[metric_to_plot], bins="auto")
                bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(counts))]
                hist_df = pd.DataFrame({"Аралық": bin_labels, "Үлгілер саны": counts}).set_index("Аралық")
                
                st.bar_chart(hist_df, color="#1B4332")
            else:
                st.info("Гистограмма құру үшін кемінде 2 сурет жүктеңіз.")
                
            csv_batch = df_batch.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Топтық нәтижелерді жүктеу (CSV / Excel)",
                data=csv_batch,
                file_name="herbarium_batch_metrics.csv",
                mime="text/csv"
            )

# ==========================================
# 2-ҚОЙЫНДЫ: ЖЕКЕ ЖАПЫРАҚ (ROI ҚИЮ)
# ==========================================
with tab_single:
    st.subheader("Затбелгі мен сызғышты алып тастау үшін қажетті аймақты (ROI) қию")
    single_file = st.file_uploader("Гербарий парағының суретін жүктеңіз", type=["jpg", "jpeg", "png"], key="roi_uploader")
    
    if single_file:
        img_roi = Image.open(single_file).convert("RGB")
        st.write("Тек өлшенетін жапырақты бөліп алу үшін қара жасыл жақтаудың бұрыштарын жылжытыңыз:")
        cropped_img = st_cropper(img_roi, realtime_update=True, box_color="#1B4332", aspect_ratio=None, key="leaf_cropper")
        
        crop_np = np.array(cropped_img)
        metrics, mask, res_img, t_val = analyze_leaf(
            crop_np, px_per_cm, auto_thresh, manual_thresh, min_area, petiole_kernel, close_gaps
        )
        
        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.image(cropped_img, caption="1. Қиып алынған аймақ (ROI)", use_container_width=True)
            c2.image(mask, caption="2. Бинарлы маска (Фон / Жапырақ)", use_container_width=True)
            c3.image(res_img, caption="3. Контур, жиектеме және осьтер", use_container_width=True)
            
            df_single = pd.DataFrame([{"Файл атауы": single_file.name, **metrics}])
            st.dataframe(df_single, use_container_width=True)
            
            csv_single = df_single.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Өлшем нәтижелерін жүктеу (CSV)",
                data=csv_single,
                file_name=f"roi_{single_file.name.split('.')[0]}.csv",
                mime="text/csv"
            )
        else:
            st.warning("Таңдалған аймақтан жапырақ табылмады. Бүйірлік панельдегі табалдырық мәнін өзгертіп көріңіз.")

# ==========================================
# 3-ҚОЙЫНДЫ: ИНТЕРАКТИВТІ СЫЗҒЫШПЕН КАЛИБРЛЕУ
# ==========================================
with tab_ruler:
    st.subheader("Суреттегі сызғыш бойынша 1 см-дегі пиксель санын анықтау")
    st.markdown("1. Кадрда сызғышы бар суретті жүктеңіз.\n2. Жақтауды сызғыштағы белгілі бір кесінді бойымен созыңыз (мысалы, **0 см-ден 5 см-ге дейін**).\n3. Нақты ұзындықты см-мен енгізіп, **Осы масштабты барлық өлшемдерге қолдану** батырмасын басыңыз.")
    
    ruler_file = st.file_uploader("Сызғышы бар суретті жүктеңіз", type=["jpg", "jpeg", "png"], key="ruler_uploader")
    
    if ruler_file:
        img_ruler = Image.open(ruler_file).convert("RGB")
        rect_coords = st_cropper(
            img_ruler,
            realtime_update=True,
            box_color="#1B4332",
            aspect_ratio=None,
            return_type="box",
            key="ruler_cropper"
        )
        
        box_w = rect_coords["width"]
        box_h = rect_coords["height"]
        measured_px = max(box_w, box_h)
        
        col_r1, col_r2 = st.columns(2)
        real_cm = col_r1.number_input("Таңдалған кесіндінің нақты ұзындығы (см):", min_value=0.1, value=1.0, step=0.5)
        calculated_scale = measured_px / real_cm
        
        col_r2.metric("Есептелген масштаб", f"{calculated_scale:.2f} пикс / см", help=f"{real_cm} см аралықта {measured_px} пиксель өлшенді")
        
        if st.button("Осы масштабты барлық өлшемдерге қолдану"):
            st.session_state.px_per_cm = round(calculated_scale, 2)
            st.rerun()
