import os
import pickle
import json
import sys
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train.rf_model import predict_price_rf

st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠", layout="wide")


st.markdown(
    """
<style>
    .prediction-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-value {
        font-size: 36px;
        font-weight: bold;
        color: #0068c9;
        text-align: center;
    }
    .prediction-label {
        font-size: 16px;
        color: #6c757d;
        text-align: center;
    }
    .feature-importance {
        margin-top: 10px;
        padding: 10px;
        background-color: #f1f1f1;
        border-radius: 5px;
    }
    .confidence-indicator {
        margin-top: 15px;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .high-confidence {
        background-color: #d4edda;
        color: #155724;
    }
    .medium-confidence {
        background-color: #fff3cd;
        color: #856404;
    }
    .low-confidence {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Prediksi Harga Rumah")
st.markdown(
    """
Gunakan formulir di bawah ini untuk memprediksi harga rumah berdasarkan karakteristiknya.
Model ini telah dilatih menggunakan Random Forest yang telah dioptimalkan.
"""
)

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_dir, "../model")
EVAL_DIR = os.path.join(MODEL_DIR, "evaluation")

OPTIMIZED_RF_PATH = os.path.join(MODEL_DIR, "optimized_rf_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "optimized_rf_model_metadata.json")


@st.cache_data
def load_metadata():
    try:
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.error(f"Error loading model metadata: {e}")
        return None


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


metadata = load_metadata()
models = load_models()

feature_list = [
    "price",
    "bedroom",
    "bathroom",
    "LT",
    "LB",
    "carport",
    "kecamatan_encoded",
    "price_per_m2_land",
    "price_per_m2_building",
    "building_efficiency",
    "listing_age_days",
    "fuzzy_area_quality",
    "fuzzy_quality_score",
]


prediction_features = [f for f in feature_list if f != "price"]

feature_importance = metadata.get("feature_importance", {}) if metadata else {}


def format_price(price):
    """Format price with Rp symbol and thousand separators"""
    return f"Rp {price:,.0f}"


kecamatan_list = [
    "Bantul",
    "Caturtunggal",
    "Cebongan",
    "Danurejan",
    "Demangan",
    "Gedong Tengen",
    "Gondokusuman",
    "Gondomanan",
    "Jetis",
    "Kaliurang",
    "Kotagede",
    "Kraton",
    "Kulonprogo",
    "Maguwoharjo",
    "Mantrijeron",
    "Mergangsan",
    "Minomartani",
    "Ngampilan",
    "Nologaten",
    "Pakualaman",
    "Pogung",
    "Purwomartani",
    "Seturan",
    "Sidoarum",
    "Sleman",
    "Tegalrejo",
    "Umbulharjo",
    "Wirobrajan",
]


def load_kecamatan_encoding():
    encoding_path = os.path.join(MODEL_DIR, "../reports/kecamatan_encoding.json")
    if os.path.exists(encoding_path):
        try:
            with open(encoding_path, "r") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load kecamatan encoding: {e}")
    return {k: i for i, k in enumerate(kecamatan_list)}


def calculate_fuzzy_area_quality(lt, lb):
    try:

        lt_fuzzy = ctrl.Antecedent(np.linspace(0, 2000, 100), "lt")
        lb_fuzzy = ctrl.Antecedent(np.linspace(0, 1000, 100), "lb")
        area_quality = ctrl.Consequent(np.linspace(0, 100, 100), "area_quality")

        lt_fuzzy["very_small"] = fuzz.trapmf(lt_fuzzy.universe, [0, 0, 60, 110])
        lt_fuzzy["small"] = fuzz.trimf(lt_fuzzy.universe, [80, 150, 250])
        lt_fuzzy["medium"] = fuzz.trimf(lt_fuzzy.universe, [200, 350, 600])
        lt_fuzzy["large"] = fuzz.trimf(lt_fuzzy.universe, [450, 750, 1200])
        lt_fuzzy["very_large"] = fuzz.trapmf(lt_fuzzy.universe, [900, 1500, 2000, 2000])

        lb_fuzzy["very_small"] = fuzz.trapmf(lb_fuzzy.universe, [0, 0, 40, 80])
        lb_fuzzy["small"] = fuzz.trimf(lb_fuzzy.universe, [60, 100, 170])
        lb_fuzzy["medium"] = fuzz.trimf(lb_fuzzy.universe, [140, 220, 380])
        lb_fuzzy["large"] = fuzz.trimf(lb_fuzzy.universe, [320, 480, 650])
        lb_fuzzy["very_large"] = fuzz.trapmf(lb_fuzzy.universe, [550, 750, 1000, 1000])

        area_quality["poor"] = fuzz.trapmf(area_quality.universe, [0, 0, 20, 40])
        area_quality["standard"] = fuzz.trimf(area_quality.universe, [30, 50, 70])
        area_quality["excellent"] = fuzz.trapmf(
            area_quality.universe, [60, 80, 100, 100]
        )

        area_rules = [
            ctrl.Rule(
                lt_fuzzy["very_small"] & lb_fuzzy["very_small"], area_quality["poor"]
            ),
            ctrl.Rule(lt_fuzzy["very_small"] & lb_fuzzy["small"], area_quality["poor"]),
            ctrl.Rule(lt_fuzzy["small"] & lb_fuzzy["very_small"], area_quality["poor"]),
            ctrl.Rule(lt_fuzzy["small"] & lb_fuzzy["small"], area_quality["standard"]),
            ctrl.Rule(lt_fuzzy["small"] & lb_fuzzy["medium"], area_quality["standard"]),
            ctrl.Rule(lt_fuzzy["medium"] & lb_fuzzy["small"], area_quality["standard"]),
            ctrl.Rule(
                lt_fuzzy["medium"] & lb_fuzzy["medium"], area_quality["standard"]
            ),
            ctrl.Rule(
                lt_fuzzy["medium"] & lb_fuzzy["large"], area_quality["excellent"]
            ),
            ctrl.Rule(
                lt_fuzzy["large"] & lb_fuzzy["medium"], area_quality["excellent"]
            ),
            ctrl.Rule(lt_fuzzy["large"] & lb_fuzzy["large"], area_quality["excellent"]),
            ctrl.Rule(
                lt_fuzzy["very_large"] & lb_fuzzy["large"], area_quality["excellent"]
            ),
            ctrl.Rule(
                lt_fuzzy["very_large"] & lb_fuzzy["very_large"],
                area_quality["excellent"],
            ),
        ]

        area_ctrl = ctrl.ControlSystem(area_rules)
        area_system = ctrl.ControlSystemSimulation(area_ctrl)

        area_system.input["lt"] = min(lt, 2000)
        area_system.input["lb"] = min(lb, 1000)

        area_system.compute()

        return area_system.output["area_quality"] / 100.0
    except Exception as e:
        st.warning(f"Could not calculate fuzzy area quality: {e}")
        return 0.5


def calculate_fuzzy_room_quality(bedroom, bathroom):
    try:

        bedroom_fuzzy = ctrl.Antecedent(np.arange(0, 12, 0.5), "bedroom")
        bathroom_fuzzy = ctrl.Antecedent(np.arange(0, 12, 0.5), "bathroom")
        room_quality = ctrl.Consequent(np.linspace(0, 100, 100), "room_quality")

        bedroom_fuzzy["few"] = fuzz.trapmf(bedroom_fuzzy.universe, [0, 0, 1, 2.5])
        bedroom_fuzzy["standard"] = fuzz.trimf(bedroom_fuzzy.universe, [1.5, 3, 5])
        bedroom_fuzzy["many"] = fuzz.trapmf(bedroom_fuzzy.universe, [4, 6, 11, 11])

        bathroom_fuzzy["few"] = fuzz.trapmf(bathroom_fuzzy.universe, [0, 0, 1, 2])
        bathroom_fuzzy["standard"] = fuzz.trimf(bathroom_fuzzy.universe, [1, 2.5, 4])
        bathroom_fuzzy["many"] = fuzz.trapmf(bathroom_fuzzy.universe, [3, 5, 11, 11])

        room_quality["poor"] = fuzz.trapmf(room_quality.universe, [0, 0, 20, 40])
        room_quality["standard"] = fuzz.trimf(room_quality.universe, [30, 50, 70])
        room_quality["excellent"] = fuzz.trapmf(
            room_quality.universe, [60, 80, 100, 100]
        )

        room_rules = [
            ctrl.Rule(
                bedroom_fuzzy["few"] & bathroom_fuzzy["few"], room_quality["poor"]
            ),
            ctrl.Rule(
                bedroom_fuzzy["few"] & bathroom_fuzzy["standard"],
                room_quality["standard"],
            ),
            ctrl.Rule(
                bedroom_fuzzy["standard"] & bathroom_fuzzy["few"],
                room_quality["standard"],
            ),
            ctrl.Rule(
                bedroom_fuzzy["standard"] & bathroom_fuzzy["standard"],
                room_quality["standard"],
            ),
            ctrl.Rule(
                bedroom_fuzzy["standard"] & bathroom_fuzzy["many"],
                room_quality["excellent"],
            ),
            ctrl.Rule(
                bedroom_fuzzy["many"] & bathroom_fuzzy["standard"],
                room_quality["excellent"],
            ),
            ctrl.Rule(
                bedroom_fuzzy["many"] & bathroom_fuzzy["many"],
                room_quality["excellent"],
            ),
        ]

        room_ctrl = ctrl.ControlSystem(room_rules)
        room_system = ctrl.ControlSystemSimulation(room_ctrl)

        room_system.input["bedroom"] = min(bedroom, 11)
        room_system.input["bathroom"] = min(bathroom, 11)

        room_system.compute()

        return room_system.output["room_quality"] / 100.0
    except Exception as e:
        st.warning(f"Could not calculate fuzzy room quality: {e}")
        return 0.5


def calculate_fuzzy_quality_score(bedroom, bathroom, lt, lb, carport):
    try:

        room_quality = calculate_fuzzy_room_quality(bedroom, bathroom)
        area_quality = calculate_fuzzy_area_quality(lt, lb)

        carport_bonus = min(carport * 0.15, 0.3)
        bathroom_bonus = min(bathroom * 0.08, 0.25)
        quality_score = (
            (room_quality * 1.2 + area_quality) / 2 + carport_bonus + bathroom_bonus
        )

        quality_score = min(quality_score, 1.0)

        return quality_score
    except Exception as e:
        st.warning(f"Could not calculate fuzzy quality score: {e}")
        return 0.5


def calculate_confidence(input_values, metadata):
    confidence_score = 1.0
    feature_ranges = metadata.get("feature_ranges", {})

    for feature, value in input_values.items():
        if feature in feature_ranges:
            min_val = feature_ranges[feature]["min"]
            max_val = feature_ranges[feature]["max"]

            range_size = max_val - min_val
            if range_size == 0:
                continue

            center = (min_val + max_val) / 2
            distance_from_center = abs(value - center) / (range_size / 2)

            if distance_from_center > 1.5:
                confidence_score *= 0.6
            elif distance_from_center > 1.0:
                confidence_score *= 0.75
            elif distance_from_center > 0.7:
                confidence_score *= 0.9

    return confidence_score


st.header("Input Data Rumah")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        bedroom = st.number_input(
            "Jumlah Kamar Tidur", min_value=1, max_value=10, value=3
        )
        bathroom = st.number_input(
            "Jumlah Kamar Mandi", min_value=1, max_value=10, value=2
        )
        lt = st.number_input("Luas Tanah (m²)", min_value=20, max_value=2000, value=150)
        lb = st.number_input(
            "Luas Bangunan (m²)", min_value=20, max_value=1000, value=100
        )

    with col2:
        carport = st.number_input("Jumlah Carport", min_value=0, max_value=5, value=1)
        kecamatan = st.selectbox("Kecamatan", options=sorted(kecamatan_list))
        listing_age = st.slider(
            "Usia Listing (hari)", min_value=0, max_value=365, value=30
        )

    submit_button = st.form_submit_button("Prediksi Harga")


if submit_button or "last_prediction" in st.session_state:

    if submit_button:

        building_efficiency = lb / lt if lt > 0 else 0.5

        fuzzy_area_quality = calculate_fuzzy_area_quality(lt, lb)
        fuzzy_quality_score = calculate_fuzzy_quality_score(
            bedroom, bathroom, lt, lb, carport
        )

        base_price_estimate = (
            1500000 * lt + 5000000 * bedroom + 100000000 * bathroom + 75000000 * carport
        )

        bathroom_factor = 10000000 * bathroom
        carport_factor = 6000000 * carport

        price_per_m2_land = (
            (base_price_estimate + bathroom_factor) / lt if lt > 0 else 0
        )
        price_per_m2_building = (
            (base_price_estimate + carport_factor) / lb if lb > 0 else 0
        )

        kecamatan_encoding = load_kecamatan_encoding()
        kecamatan_encoded = kecamatan_encoding.get(kecamatan, 0)

        input_data = {
            "bedroom": bedroom,
            "bathroom": bathroom,
            "LT": lt,
            "LB": lb,
            "carport": carport,
            "kecamatan_encoded": kecamatan_encoded,
            "building_efficiency": building_efficiency,
            "listing_age_days": listing_age,
            "fuzzy_area_quality": fuzzy_area_quality,
            "fuzzy_quality_score": fuzzy_quality_score,
            "price_per_m2_land": price_per_m2_land,
            "price_per_m2_building": price_per_m2_building,
            "price": base_price_estimate,
        }

        st.session_state.last_prediction = input_data
        st.session_state.last_input = {
            "bedroom": bedroom,
            "bathroom": bathroom,
            "LT": lt,
            "LB": lb,
            "carport": carport,
            "kecamatan": kecamatan,
            "listing_age": listing_age,
        }
    else:

        input_data = st.session_state.last_prediction

    predicted_price = 0
    confidence = 0

    if models["rf"] is not None:
        try:

            predict_features = {
                "bedroom": [input_data["bedroom"]],
                "bathroom": [input_data["bathroom"]],
                "LT": [input_data["LT"]],
                "LB": [input_data["LB"]],
                "carport": [input_data["carport"]],
                "kecamatan_encoded": [input_data["kecamatan_encoded"]],
                "building_efficiency": [input_data["building_efficiency"]],
                "listing_age_days": [input_data["listing_age_days"]],
                "price_per_m2_land": [input_data["price_per_m2_land"]],
                "price_per_m2_building": [input_data["price_per_m2_building"]],
                "bathroom_impact": [input_data["bathroom"] * 100],
                "carport_impact": [input_data["carport"] * 100],
                "fuzzy_area_quality": [input_data["fuzzy_area_quality"]],
                "fuzzy_quality_score": [input_data["fuzzy_quality_score"]],
                "price": [input_data["price"]],
            }

            predict_df = pd.DataFrame(predict_features)

            if hasattr(models["rf"], "feature_names_in_"):
                required_columns = models["rf"].feature_names_in_

                for col in required_columns:
                    if col not in predict_df.columns:
                        predict_df[col] = 0

                predict_df = predict_df[required_columns]

            predicted_price = models["rf"].predict(predict_df)[0]

            st.session_state.predicted_price = predicted_price

            st.expander("Debug Info - Feature Values").write(
                {
                    "bedroom": input_data["bedroom"],
                    "bathroom": input_data["bathroom"],
                    "LT": input_data["LT"],
                    "LB": input_data["LB"],
                    "carport": input_data["carport"],
                    "kecamatan_encoded": input_data["kecamatan_encoded"],
                    "building_efficiency": input_data["building_efficiency"],
                    "listing_age_days": input_data["listing_age_days"],
                    "price_per_m2_land": input_data["price_per_m2_land"],
                    "price_per_m2_building": input_data["price_per_m2_building"],
                    "fuzzy_area_quality": input_data["fuzzy_area_quality"],
                    "fuzzy_quality_score": input_data["fuzzy_quality_score"],
                    "base_price_estimate": input_data["price"],
                }
            )

            confidence = calculate_confidence(input_data, metadata) if metadata else 0.7
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.exception(e)
            predicted_price = 5000000000

    st.header("Hasil Prediksi")

    st.markdown(
        f"""
    <div class="prediction-container">
        <p class="prediction-label">Prediksi Harga Rumah</p>
        <p class="prediction-value">{format_price(predicted_price)}</p>
        <div class="prediction-label">Harga per m² Tanah: {format_price(predicted_price / input_data['LT'])}</div>
        <div class="prediction-label">Harga per m² Bangunan: {format_price(predicted_price / input_data['LB'])}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if confidence > 0.85:
        confidence_class = "high-confidence"
        confidence_text = "Tinggi"
    elif confidence > 0.7:
        confidence_class = "medium-confidence"
        confidence_text = "Sedang"
    else:
        confidence_class = "low-confidence"
        confidence_text = "Rendah"

    st.markdown(
        f"""
    <div class="confidence-indicator {confidence_class}">
        <p><strong>Tingkat Confidence: {confidence_text}</strong> ({confidence:.1%})</p>
        <p>{'Prediksi ini memiliki tingkat confidence tinggi berdasarkan data training.' if confidence > 0.85 else 
            'Prediksi ini memiliki beberapa ketidakpastian, nilai mungkin berbeda dari kenyataan.' if confidence > 0.7 else
            'Prediksi ini memiliki ketidakpastian tinggi, nilai mungkin berbeda secara signifikan dari kenyataan.'}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if (
        metadata
        and "price_ranges" in metadata
        and metadata["price_ranges"].get(str(kecamatan_encoded))
    ):
        st.subheader("Perbandingan dengan Properti Serupa")

        area_range = metadata["price_ranges"][str(kecamatan_encoded)]

        avg_price = area_range.get("avg", predicted_price)
        min_price = area_range.get("min", predicted_price * 0.7)
        max_price = area_range.get("max", predicted_price * 1.3)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_price,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Prediksi Relatif terhadap Properti di {kecamatan}"},
                gauge={
                    "axis": {"range": [min_price, max_price]},
                    "bar": {"color": "royalblue"},
                    "steps": [
                        {"range": [min_price, avg_price], "color": "lightblue"},
                        {"range": [avg_price, max_price], "color": "lightgray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": avg_price,
                    },
                },
            )
        )
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.write(
            f"Harga rata-rata untuk properti di {kecamatan}: {format_price(avg_price)}"
        )
        st.write(
            f"Rentang harga: {format_price(min_price)} - {format_price(max_price)}"
        )

st.markdown(
    """
---
**Catatan:** Prediksi ini menggunakan model Random Forest yang dilatih dengan data rumah di Yogyakarta.
Hasil prediksi adalah perkiraan dan harga sebenarnya dapat bervariasi berdasarkan kondisi pasar dan faktor-faktor lain.
"""
)
