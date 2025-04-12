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


st.sidebar.title("Model Explorer")
section = st.sidebar.radio(
    "Go to", ["Model Info", "Feature Importance", "Model Plots", "Make Prediction"]
)


model = load_model()
feature_info = load_feature_info()
feature_names = feature_info["features"]
feature_importance_df = load_feature_importance()
best_params = load_best_params()


if section == "Model Info":
    st.title("🏠 House Price Model Info")
    st.subheader("Model Path")
    st.code(MODEL_PATH)

    st.subheader("Best Parameters")
    st.json(best_params)

    st.subheader("Features Used")
    st.write(feature_names)


elif section == "Feature Importance":
    st.title("📊 Feature Importance")
    st.dataframe(feature_importance_df)

    st.subheader("Top 15 Feature Importances")
    st.image(os.path.join(EVAL_DIR, "feature_importance.png"))


elif section == "Model Plots":
    st.title("📈 Model Visualizations")

    st.subheader("Actual vs Predicted")
    st.image(os.path.join(EVAL_DIR, "actual_vs_predicted.png"))

    st.subheader("Residuals Plot")
    st.image(os.path.join(EVAL_DIR, "residuals.png"))

    st.subheader("Residual Distribution")
    st.image(os.path.join(EVAL_DIR, "residual_distribution.png"))


elif section == "Make Prediction":
    st.title("🔍 Predict House Price")
    st.subheader("Input Feature Values")

    input_data = {}
    city_columns = [f for f in feature_names if f.startswith("city_")]

    for feature in feature_names:
        if feature.startswith("city_"):
            continue
        input_data[feature] = st.number_input(f"{feature}", value=0, step=1)

    selected_city = st.selectbox(
        "City", options=[col.replace("city_", "") for col in city_columns]
    )
    input_data["city"] = selected_city

    if st.button("Predict"):
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
        st.success(f"Estimated House Price: Rp {prediction:,.0f}")
