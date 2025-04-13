import os
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Prediction", page_icon="🏠", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_dir, "../model")
EVAL_DIR = os.path.join(MODEL_DIR, "evaluation")

OPTIMIZED_RF_PATH = os.path.join(MODEL_DIR, "optimized_rf_model.pkl")


@st.cache_resource
def load_models():
    models = {}

    try:
        with open(OPTIMIZED_RF_PATH, "rb") as f:
            models["rf"] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading RF model: {e}")
        models["rf"] = None

    return models


def predict_price_rf(rf_model, bedroom_val, bathroom_val, lt_val, lb_val):
    input_data = np.array([[bedroom_val, bathroom_val, lt_val, lb_val]])
    return rf_model.predict(input_data)[0]


models = load_models()

st.title("🏠 Prediksi Harga Rumah di Kota Yogyakarta")
st.markdown(
    """
    Halo, selamat datang di aplikasi prediksi harga rumah. Aplikasi ini menggunakan model:
    
    **Model Random Forest**: Model machine learning berbasis ensemble learning yang dioptimasi dengan algoritma genetik
    
    Masukkan data yang diperlukan untuk memprediksi harga rumah berdasarkan fitur-fitur yang ada.
    """
)

st.subheader("Masukkan Data Rumah")

col1, col2 = st.columns(2)

with col1:
    bedroom = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=2)
    bathroom = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=10, value=1)

with col2:
    lt = st.number_input("Luas Tanah (m²)", min_value=50, max_value=1000, value=120)
    lb = st.number_input("Luas Bangunan (m²)", min_value=30, max_value=500, value=80)

if st.button("Prediksi Harga"):
    success = False

    results_container = st.container()

    with results_container:
        if models["rf"] is not None:
            try:
                rf_prediction = predict_price_rf(
                    models["rf"], bedroom, bathroom, lt, lb
                )
                st.success(f"**Prediksi Harga Rumah**: Rp {rf_prediction:,.0f}")
                success = True
            except Exception as e:
                st.error(f"Error melakukan prediksi: {e}")
        else:
            st.error("Model Random Forest tidak tersedia")

    if not success:
        st.error(
            "Prediksi tidak berhasil. Periksa apakah model tersedia dan input valid."
        )

with st.expander("Informasi Tambahan"):
    st.markdown(
        """
    ### Tentang Model Prediksi
    
    **Model Random Forest** menggunakan ensemble learning yang telah dioptimasi dengan algoritma genetika untuk menemukan parameter terbaik.
    
    Parameter model:
    - n_estimators: {n_est}
    - max_depth: {max_d}
    - min_samples_split: {min_split}
    - min_samples_leaf: {min_leaf}
    
    ### Rentang Input yang Direkomendasikan
    
    - **Kamar Tidur**: 1-5
    - **Kamar Mandi**: 1-4
    - **Luas Tanah**: 70-500 m²
    - **Luas Bangunan**: 40-300 m²
    
    *Prediksi mungkin kurang akurat untuk nilai di luar rentang tersebut.*
    """.format(
            n_est=models["rf"].n_estimators if models["rf"] else "N/A",
            max_d=models["rf"].max_depth if models["rf"] else "N/A",
            min_split=models["rf"].min_samples_split if models["rf"] else "N/A",
            min_leaf=models["rf"].min_samples_leaf if models["rf"] else "N/A",
        )
    )

st.sidebar.title("Tentang Aplikasi")
st.sidebar.info(
    """
### Sistem Prediksi Harga Rumah
Aplikasi ini menggunakan model Random Forest yang telah dioptimasi menggunakan algoritma genetika.

Dataset berisi informasi rumah di Yogyakarta yang telah diproses dan dibersihkan.
"""
)

st.sidebar.success(
    """
### Metrik Model
- MAPE Random Forest: ~9-10% 

*Nilai aktual mungkin berbeda tergantung pada dataset yang digunakan.*
"""
)
