import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_dir, "model")
EVAL_DIR = os.path.join(MODEL_DIR, "evaluation")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
PREDICTION_INFO_PATH = os.path.join(EVAL_DIR, "prediction_info.json")
FEATURE_IMPORTANCE_PATH = os.path.join(EVAL_DIR, "feature_importance.csv")
BEST_PARAMS_PATH = os.path.join(EVAL_DIR, "best_params.json")


@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_info():
    with open(PREDICTION_INFO_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_data
def load_best_params():
    with open(BEST_PARAMS_PATH, "r") as f:
        return json.load(f)


model = load_model()
feature_info = load_feature_info()
feature_names = feature_info["features"]


st.title("🏠 Prediksi Harga Rumah")
st.markdown(
    """
Halo, selamat datang di aplikasi prediksi harga rumah. Masukkan data yang diperlukan untuk memprediksi harga rumah berdasarkan fitur-fitur yang ada.
"""
)


st.subheader("Masukkan Nilai Fitur")

input_data = {}
city_columns = [f for f in feature_names if f.startswith("city_")]

for feature in feature_names:
    if feature.startswith("city_"):
        continue
    input_data[feature] = st.number_input(f"{feature}", value=0, step=1)

selected_city = st.selectbox(
    "Kota", options=[col.replace("city_", "") for col in city_columns]
)
input_data["city"] = selected_city

if st.button("Prediksi"):
    df = pd.DataFrame(columns=feature_names)
    df.loc[0] = 0

    for feature, value in input_data.items():
        if feature in feature_names:
            df.loc[0, feature] = value
        elif feature == "city":
            city_col = "city_" + value
            if city_col in feature_names:
                df.loc[0, city_col] = 1

    prediction = model.predict(df)[0]
    st.success(f"Perkiraan Harga Rumah: Rp {prediction:,.0f}")
