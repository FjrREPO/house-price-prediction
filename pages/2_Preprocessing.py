import os
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error

from utils.helper import preprocess_data, remove_outliers

st.set_page_config(
    page_title="Preprocessing Data",
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
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        border-left: 4px solid #FF4B4B;
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
</style>
""",
    unsafe_allow_html=True,
)

# Load and preprocess data
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "../dataset/houses.csv")
df = pd.read_csv(dataset_path)

df_preprocessed = preprocess_data(df)

# Remove outliers
columns_to_check = ["price", "bedroom", "bathroom", "LT", "LB"]
df_cleaned = remove_outliers(df_preprocessed, columns_to_check)


# Create fuzzy logic model based on cleaned data
def create_fuzzy_model(df):
    # Calculate quartiles for each feature
    bedroom_quartiles = np.percentile(df["bedroom"], [25, 50, 75])
    bathroom_quartiles = np.percentile(df["bathroom"], [25, 50, 75])
    LT_quartiles = np.percentile(df["LT"], [25, 50, 75])
    LB_quartiles = np.percentile(df["LB"], [25, 50, 75])
    price_quartiles = np.percentile(df["price"], [25, 50, 75])

    # Create antecedents and consequent
    bathroom = ctrl.Antecedent(np.arange(0, max(df["bathroom"]) + 1, 1), "bathroom")
    bedroom = ctrl.Antecedent(np.arange(0, max(df["bedroom"]) + 1, 1), "bedroom")
    LT = ctrl.Antecedent(np.arange(0, max(df["LT"]) + 1, 10), "LT")
    LB = ctrl.Antecedent(np.arange(0, max(df["LB"]) + 1, 10), "LB")

    price = ctrl.Consequent(
        np.arange(
            min(df["price"]),
            max(df["price"]) + 1,
            100_000,
        ),
        "price",
    )

    # Membership functions for bathroom
    bathroom["poor"] = fuzz.trapmf(
        bathroom.universe,
        [0, bathroom_quartiles[0], bathroom_quartiles[0], bathroom_quartiles[1]],
    )
    bathroom["average"] = fuzz.trapmf(
        bathroom.universe,
        [
            bathroom_quartiles[0],
            bathroom_quartiles[1],
            bathroom_quartiles[1],
            bathroom_quartiles[2],
        ],
    )
    bathroom["good"] = fuzz.trapmf(
        bathroom.universe,
        [
            bathroom_quartiles[1],
            bathroom_quartiles[2],
            bathroom_quartiles[2],
            df["bathroom"].max(),
        ],
    )

    # Membership functions for bedroom
    bedroom["poor"] = fuzz.trapmf(
        bedroom.universe,
        [0, bedroom_quartiles[0], bedroom_quartiles[0], bedroom_quartiles[1]],
    )
    bedroom["average"] = fuzz.trapmf(
        bedroom.universe,
        [
            bedroom_quartiles[0],
            bedroom_quartiles[1],
            bedroom_quartiles[1],
            bedroom_quartiles[2],
        ],
    )
    bedroom["good"] = fuzz.trapmf(
        bedroom.universe,
        [
            bedroom_quartiles[1],
            bedroom_quartiles[2],
            bedroom_quartiles[2],
            df["bedroom"].max(),
        ],
    )

    # Membership functions for LT (Land Area)
    LT["poor"] = fuzz.trapmf(
        LT.universe, [0, LT_quartiles[0], LT_quartiles[0], LT_quartiles[1]]
    )
    LT["average"] = fuzz.trapmf(
        LT.universe,
        [LT_quartiles[0], LT_quartiles[1], LT_quartiles[1], LT_quartiles[2]],
    )
    LT["good"] = fuzz.trapmf(
        LT.universe,
        [
            LT_quartiles[1],
            LT_quartiles[2],
            LT_quartiles[2],
            df["LT"].max(),
        ],
    )

    # Membership functions for LB (Building Area)
    LB["poor"] = fuzz.trapmf(
        LB.universe, [0, LB_quartiles[0], LB_quartiles[0], LB_quartiles[1]]
    )
    LB["average"] = fuzz.trapmf(
        LB.universe,
        [LB_quartiles[0], LB_quartiles[1], LB_quartiles[1], LB_quartiles[2]],
    )
    LB["good"] = fuzz.trapmf(
        LB.universe,
        [
            LB_quartiles[1],
            LB_quartiles[2],
            LB_quartiles[2],
            df["LB"].max(),
        ],
    )

    # Calculate means and spreads for price membership functions
    price_mean_poor = price_quartiles[0]
    price_mean_average = price_quartiles[1]
    price_mean_good = price_quartiles[2]

    spread_poor = (price_quartiles[1] - price_quartiles[0]) / 2
    spread_average = (price_quartiles[2] - price_quartiles[1]) / 2
    spread_good = (df["price"].max() - price_quartiles[2]) / 2

    # Membership functions for price
    price["poor"] = fuzz.gaussmf(price.universe, price_mean_poor, spread_poor)
    price["average"] = fuzz.gaussmf(price.universe, price_mean_average, spread_average)
    price["good"] = fuzz.gaussmf(price.universe, price_mean_good, spread_good)

    # Define fuzzy rules
    rule1 = ctrl.Rule(
        bedroom["good"] & bathroom["good"] & LT["good"] & LB["good"], price["good"]
    )
    rule2 = ctrl.Rule(
        bedroom["average"] & bathroom["average"] & LT["average"] & LB["average"],
        price["average"],
    )
    rule3 = ctrl.Rule(
        bedroom["poor"] & bathroom["poor"] & LT["poor"] & LB["poor"], price["poor"]
    )

    rule4 = ctrl.Rule(
        bedroom["good"] & bathroom["average"] & LT["good"] & LB["good"], price["good"]
    )
    rule5 = ctrl.Rule(
        bedroom["average"] & bathroom["good"] & LT["average"] & LB["good"],
        price["average"],
    )
    rule6 = ctrl.Rule(
        bedroom["poor"] & bathroom["good"] & LT["good"] & LB["average"],
        price["average"],
    )
    rule7 = ctrl.Rule(
        bedroom["average"] & bathroom["poor"] & LT["average"] & LB["average"],
        price["poor"],
    )
    rule8 = ctrl.Rule(
        bedroom["good"] & bathroom["good"] & LT["poor"] & LB["average"],
        price["average"],
    )

    # Create control system
    pricing_ctrl = ctrl.ControlSystem(
        [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8]
    )

    pricing_sim = ctrl.ControlSystemSimulation(pricing_ctrl)

    return pricing_sim, {
        "bathroom_quartiles": bathroom_quartiles,
        "bedroom_quartiles": bedroom_quartiles,
        "LT_quartiles": LT_quartiles,
        "LB_quartiles": LB_quartiles,
        "price_quartiles": price_quartiles,
    }


def predict_price_fuzzy(sim, bedroom_val, bathroom_val, LT_val, LB_val):
    try:
        sim.input["bedroom"] = bedroom_val
        sim.input["bathroom"] = bathroom_val
        sim.input["LT"] = LT_val
        sim.input["LB"] = LB_val

        sim.compute()

        return sim.output["price"] if "price" in sim.output else None
    except Exception as e:
        st.error(f"Error predicting price: {str(e)}")
        return None


# Main app
st.markdown(
    '<div class="main-header">🏠 Preprocessing Data</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Analyze and predict property prices using fuzzy logic.</div>',
    unsafe_allow_html=True,
)

last_updated = datetime.now().strftime("%d %B %Y, %H:%M:%S")
st.markdown(f"*Last updated: {last_updated}*")

with st.sidebar:
    st.markdown("## 🔍 Menu")
    st.markdown("### 📊 Filter")
    show_rows = st.slider("Rows to display", 5, 50, 10)
    st.markdown("### 🧭 Navigation")
    nav_options = ["Overview", "Data Analysis", "Price Prediction"]
    selected_section = st.radio("Select section:", nav_options)

    if "kabupaten_kota" in df_cleaned.columns:
        st.markdown("### 🌍 Region")
        available_regions = ["All"] + list(df_cleaned["kabupaten_kota"].unique())
        selected_region = st.selectbox("Select region:", available_regions)

# Filter by region if selected
if "kabupaten_kota" in df_cleaned.columns and selected_region != "All":
    filtered_df = df_cleaned[df_cleaned["kabupaten_kota"] == selected_region]
else:
    filtered_df = df_cleaned

# Create fuzzy model
fuzzy_model, quartiles = create_fuzzy_model(filtered_df)

if selected_section == "Overview":
    st.markdown(
        '<div class="section-header">📊 Data Summary</div>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Properties", len(filtered_df))
    with col2:
        st.metric("Average Price", f"{filtered_df['price'].mean():,.2f}")

    st.write(filtered_df.head(show_rows))

    st.markdown(
        '<div class="section-header">📈 Data Distribution</div>', unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(
        ["Price Distribution", "Property Features", "Correlation"]
    )

    with tab1:
        fig = px.histogram(filtered_df, x="price", nbins=30, title="Price Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                filtered_df,
                y=["bedroom", "bathroom"],
                title="Bedroom and Bathroom Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                filtered_df, y=["LT", "LB"], title="Land and Building Area Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        corr = filtered_df[["price", "bedroom", "bathroom", "LT", "LB"]].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Between Features")
        st.plotly_chart(fig, use_container_width=True)

elif selected_section == "Data Analysis":
    st.markdown(
        '<div class="section-header">🧹 Cleaned Data Statistics</div>',
        unsafe_allow_html=True,
    )
    st.write(filtered_df.describe())

    st.markdown(
        '<div class="section-header">📊 Quartile Analysis</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Quartiles")
        st.write(f"Lower (25%): {quartiles['price_quartiles'][0]:,.2f}")
        st.write(f"Median (50%): {quartiles['price_quartiles'][1]:,.2f}")
        st.write(f"Upper (75%): {quartiles['price_quartiles'][2]:,.2f}")

    with col2:
        st.subheader("Property Features Quartiles")
        st.write(f"Bedroom: {quartiles['bedroom_quartiles']}")
        st.write(f"Bathroom: {quartiles['bathroom_quartiles']}")
        st.write(f"Land Area (LT): {quartiles['LT_quartiles']}")
        st.write(f"Building Area (LB): {quartiles['LB_quartiles']}")

    st.markdown(
        '<div class="section-header">🔍 Feature Analysis</div>',
        unsafe_allow_html=True,
    )

    feature = st.selectbox(
        "Select feature to analyze:", ["bedroom", "bathroom", "LT", "LB"]
    )

    fig = px.scatter(
        filtered_df,
        x=feature,
        y="price",
        title=f"Relationship between {feature} and Price",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_section == "Price Prediction":
    st.markdown(
        '<div class="section-header">⚙️ Fuzzy Logic Price Prediction</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        bedroom_val = st.slider(
            "Number of Bedrooms",
            int(filtered_df["bedroom"].min()),
            int(filtered_df["bedroom"].max()),
            int(filtered_df["bedroom"].median()),
        )

        bathroom_val = st.slider(
            "Number of Bathrooms",
            int(filtered_df["bathroom"].min()),
            int(filtered_df["bathroom"].max()),
            int(filtered_df["bathroom"].median()),
        )

    with col2:
        lt_val = st.slider(
            "Land Area (m²)",
            int(filtered_df["LT"].min()),
            int(filtered_df["LT"].max()),
            int(filtered_df["LT"].median()),
        )

        lb_val = st.slider(
            "Building Area (m²)",
            int(filtered_df["LB"].min()),
            int(filtered_df["LB"].max()),
            int(filtered_df["LB"].median()),
        )

    if st.button("Predict Price"):
        predicted_price = predict_price_fuzzy(
            fuzzy_model, bedroom_val, bathroom_val, lt_val, lb_val
        )

        if predicted_price is not None:
            st.success(f"Predicted Price: Rp {predicted_price:,.2f}")

            # Find similar properties
            st.markdown(
                '<div class="section-header">👨‍👩‍👧‍👦 Similar Properties</div>',
                unsafe_allow_html=True,
            )

            # Find properties with similar features
            bedroom_range = (bedroom_val - 1, bedroom_val + 1)
            bathroom_range = (bathroom_val - 1, bathroom_val + 1)
            lt_range = (lt_val * 0.8, lt_val * 1.2)
            lb_range = (lb_val * 0.8, lb_val * 1.2)

            similar_properties = filtered_df[
                (filtered_df["bedroom"].between(*bedroom_range))
                & (filtered_df["bathroom"].between(*bathroom_range))
                & (filtered_df["LT"].between(*lt_range))
                & (filtered_df["LB"].between(*lb_range))
            ]

            if len(similar_properties) > 0:
                st.write(f"Found {len(similar_properties)} similar properties:")
                st.write(
                    similar_properties[
                        ["price", "bedroom", "bathroom", "LT", "LB"]
                    ].head(5)
                )

                mean_price = similar_properties["price"].mean()
                st.write(f"Average price of similar properties: {mean_price:,.2f}")
                st.write(
                    f"Difference from prediction: {abs(predicted_price - mean_price):,.2f} ({abs(predicted_price - mean_price)/mean_price*100:.2f}%)"
                )
            else:
                st.write("No similar properties found.")

    st.markdown(
        '<div class="section-header">ℹ️ Model Performance</div>',
        unsafe_allow_html=True,
    )

    # Calculate MAPE by comparing prediction with actual values
    predictions = []
    sample_size = min(100, len(filtered_df))
    sample_df = filtered_df.sample(sample_size)

    for _, row in sample_df.iterrows():
        prediction = predict_price_fuzzy(
            fuzzy_model, row["bedroom"], row["bathroom"], row["LT"], row["LB"]
        )
        if prediction is not None:
            predictions.append(prediction)

    if len(predictions) > 0:
        mape = mean_absolute_percentage_error(
            sample_df["price"].iloc[: len(predictions)], predictions
        )
        st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape * 100:.2f}%")

        threshold_mape = 0.15
        accurate_predictions = sum(
            1
            for i, pred in enumerate(predictions)
            if abs(pred - sample_df["price"].iloc[i]) / sample_df["price"].iloc[i]
            <= threshold_mape
        )

        st.metric(
            "Predictions within 15% error",
            f"{accurate_predictions} out of {len(predictions)} ({accurate_predictions/len(predictions)*100:.2f}%)",
        )
