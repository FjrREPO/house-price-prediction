import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="House Price Prediction Models",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
     .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
     }
     .sub-header {
        color: #424242;
        font-size: 1.1rem;
        margin-bottom: 2rem;
     }
     .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E88E5;
        border-left: 4px solid #1E88E5;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
     }
     .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
     }
     .expander-header {
        font-weight: 600;
        color: #424242;
     }
     .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
     }
     .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-weight: 600;
     }
     .file-stats {
        margin-bottom: 10px;
        padding: 5px 0;
     }
     .plotly-graph {
        border-radius: 10px;
        background-color: #f8f9fa;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
     }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">🏠 House Price Prediction Models</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Analisis dan visualisasi model prediksi harga rumah dengan perbandingan kinerja berbagai model seperti Random Forest, Fuzzy Tsukamoto, dan model Hybrid.</div>',
    unsafe_allow_html=True,
)
last_updated = datetime.now().strftime("%d %B %Y, %H:%M:%S")
st.markdown(f"*Terakhir diperbarui: {last_updated}*")

with st.sidebar:
    st.markdown("## 🔍 Menu")
    st.markdown("### 📊 Filter")
    show_rows = st.slider("Baris yang ditampilkan", 5, 50, 10)

    st.markdown("### 🧭 Navigasi")
    nav_options = [
        "Overview",
        "Model Performance",
        "Feature Importance",
        # Etc
    ]
    selected_section = st.radio("Pilih bagian:", nav_options)

    st.markdown("### ℹ️ Tentang")
    st.info(
        "Halaman ini menampilkan hasil evaluasi model prediksi harga rumah. "
        "Gunakan filter di atas untuk menyesuaikan tampilan Anda dan navigasi untuk menuju bagian tertentu."
    )

    if st.button("🔄 Segarkan Data"):
        st.cache_data.clear()
        st.success("Data berhasil diperbarui!")
