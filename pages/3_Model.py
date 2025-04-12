import streamlit as st
import pandas as pd
import joblib
import json
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_dir, "../model")
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


st.sidebar.title("Eksplorasi Model")
section = st.sidebar.radio(
    "Pilih Menu", ["Info Model", "Pentingnya Fitur", "Visualisasi Model"]
)


model = load_model()
feature_info = load_feature_info()
feature_names = feature_info["features"]
feature_importance_df = load_feature_importance()
best_params = load_best_params()


if section == "Info Model":
    st.title("🏠 Info Model Harga Rumah")
    st.subheader("Path Model")
    st.code(MODEL_PATH)

    st.subheader("Parameter Terbaik")
    st.json(best_params)

    st.subheader("Fitur yang Digunakan")
    st.write(feature_names)


elif section == "Pentingnya Fitur":
    st.title("📊  Fitur Penting")
    st.dataframe(feature_importance_df)

    st.subheader("15 Fitur dengan Importance Tertinggi")
    st.image(os.path.join(EVAL_DIR, "feature_importance.png"))


elif section == "Visualisasi Model":
    st.title("📈 Visualisasi Model")

    st.subheader("Aktual vs Prediksi")
    st.image(os.path.join(EVAL_DIR, "actual_vs_predicted.png"))

    st.subheader("Plot Residual")
    st.image(os.path.join(EVAL_DIR, "residuals.png"))

    st.subheader("Distribusi Residual")
    st.image(os.path.join(EVAL_DIR, "residual_distribution.png"))