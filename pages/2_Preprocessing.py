import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Set page configuration
st.set_page_config(page_title="Housing Price Analysis", page_icon="🏠", layout="wide")

# Add title and description
st.title("🏠 Housing Price Analysis Dashboard")
st.markdown(
    """
This dashboard visualizes housing data across different regions with 
price predictions using a fuzzy logic model. Explore the data through 
various charts and statistics.
"""
)


# Load and process data
@st.cache_data
def load_data():
    try:
        data_path = os.path.join("dataset", "houses-cleaned.csv")
        df = pd.read_csv(data_path)

        # Clean and transform the data
        # Process price column (convert "Rp X,XX Miliar/Juta" to numeric)
        def convert_to_numeric(value: str) -> float:
            try:
                value_numeric = re.sub(r"Rp\s?", "", value)

                if "Miliar" in value_numeric:
                    value_numeric = (
                        float(re.sub(r"\s?Miliar", "", value_numeric).replace(",", ".")) * 1e9
                    )
                elif "Juta" in value_numeric:
                    value_numeric = (
                        float(re.sub(r"\s?Juta", "", value_numeric).replace(",", ".")) * 1e6
                    )
                else:
                    return None
                return value_numeric
            except ValueError:
                return None

        df["price_numeric"] = df["price"].apply(convert_to_numeric)

        # Process LT and LB columns (extract numeric values)
        def extract_area(area_str):
            if isinstance(area_str, str):
                # Extract numeric value from format ": XXX m²"
                match = re.search(r":\s*(\d+)\s*m²", area_str)
                if match:
                    return float(match.group(1))
            return np.nan

        df["LT_numeric"] = df["LT"].apply(extract_area)
        df["LB_numeric"] = df["LB"].apply(extract_area)

        # Extract location (kabupaten/kota)
        def extract_location(loc_str):
            if isinstance(loc_str, str):
                parts = loc_str.split(",")
                if len(parts) > 1:
                    return parts[1].strip()
                return parts[0].strip()
            return "Unknown"

        df["kabupaten_kota"] = df["location"].apply(extract_location)

        return df
    except FileNotFoundError:
        st.error("Data file not found. Please check the path to houses-cleaned.csv")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Debug information
        if "price" in locals():
            st.write(f"Problem values in 'price'")
        return pd.DataFrame()


df = load_data()

if not df.empty:
    # Display raw data issues if any
    if df["price_numeric"].isna().any():
        st.warning(
            f"Found {df['price_numeric'].isna().sum()} rows with invalid price data"
        )

        # Show sample of problematic rows
        problem_rows = df[df["price_numeric"].isna()][["title", "price"]].head(5)
        if not problem_rows.empty:
            st.write("Sample of problematic price data:")
            st.write(problem_rows)

    # Additional data cleaning
    df = df.dropna(subset=["price_numeric", "LT_numeric", "LB_numeric"])

    # Sidebar filters
    st.sidebar.header("Filters")

    # Region filter
    regions = ["All Regions"] + sorted(df["kabupaten_kota"].unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region", regions)

    # Price range filter
    min_price = int(df["price_numeric"].min())
    max_price = int(df["price_numeric"].max())
    price_range = st.sidebar.slider(
        "Price Range (in millions)",
        min_value=min_price // 1_000_000,
        max_value=max_price // 1_000_000,
        value=(min_price // 1_000_000, max_price // 1_000_000),
    )

    # Bedroom filter
    bedrooms = sorted(df["bedroom"].unique().tolist())
    selected_bedrooms = st.sidebar.multiselect(
        "Number of Bedrooms", options=bedrooms, default=bedrooms
    )

    # Filter the dataframe based on selections
    filtered_df = df.copy()

    if selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df["kabupaten_kota"] == selected_region]

    filtered_df = filtered_df[
        (filtered_df["price_numeric"] >= price_range[0] * 1_000_000)
        & (filtered_df["price_numeric"] <= price_range[1] * 1_000_000)
    ]

    if selected_bedrooms:
        filtered_df = filtered_df[filtered_df["bedroom"].isin(selected_bedrooms)]

    # Main layout with tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Price Analysis", "Property Features"])

    with tab1:
        st.header("Data Overview")

        # Display summary statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Summary Statistics")
            # Create a display version of the dataframe with properly formatted columns
            display_stats = (
                filtered_df[
                    [
                        "price_numeric",
                        "bedroom",
                        "bathroom",
                        "carport",
                        "LT_numeric",
                        "LB_numeric",
                    ]
                ]
                .describe()
                .round(2)
            )
            display_stats.columns = [
                "Price (Rp)",
                "Bedrooms",
                "Bathrooms",
                "Carport",
                "Land Area (m²)",
                "Building Area (m²)",
            ]
            st.dataframe(display_stats, use_container_width=True)

        with col2:
            st.subheader("Region Distribution")
            if selected_region == "All Regions":
                region_counts = df["kabupaten_kota"].value_counts().reset_index()
                region_counts.columns = ["Region", "Count"]
                fig = px.pie(region_counts, values="Count", names="Region", hole=0.4)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Displaying data for {selected_region}")
                st.metric("Number of Properties", filtered_df.shape[0])
                avg_price = filtered_df["price_numeric"].mean()
                if avg_price >= 1_000_000_000:
                    st.metric(
                        "Average Price", f"Rp {avg_price/1_000_000_000:.2f} Miliar"
                    )
                else:
                    st.metric("Average Price", f"Rp {avg_price/1_000_000:.2f} Juta")

        # Data table
        st.subheader("Data Sample")
        display_cols = [
            "title",
            "price",
            "bedroom",
            "bathroom",
            "carport",
            "LT",
            "LB",
            "location",
        ]
        st.dataframe(filtered_df[display_cols].head(10), use_container_width=True)

    with tab2:
        st.header("Price Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Price Distribution")
            # Determine the unit based on price range
            use_billions = filtered_df["price_numeric"].mean() > 1_000_000_000

            if use_billions:
                # Convert price to billions for better visualization
                filtered_df["price_display"] = (
                    filtered_df["price_numeric"] / 1_000_000_000
                )
                price_label = "Price (Miliar Rp)"
            else:
                # Convert price to millions for better visualization
                filtered_df["price_display"] = filtered_df["price_numeric"] / 1_000_000
                price_label = "Price (Juta Rp)"

            fig = px.histogram(
                filtered_df,
                x="price_display",
                nbins=30,
                title="Price Distribution Histogram",
                labels={"price_display": price_label},
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Price by Region")
            if selected_region == "All Regions":
                # Determine the unit based on price range
                use_billions = df["price_numeric"].mean() > 1_000_000_000

                if use_billions:
                    # Convert price to billions for better visualization
                    df["price_display"] = df["price_numeric"] / 1_000_000_000
                    price_label = "Price (Miliar Rp)"
                else:
                    # Convert price to millions for better visualization
                    df["price_display"] = df["price_numeric"] / 1_000_000
                    price_label = "Price (Juta Rp)"

                fig = px.box(
                    df,
                    x="kabupaten_kota",
                    y="price_display",
                    title="Price Distribution by Region",
                    labels={"kabupaten_kota": "Region", "price_display": price_label},
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Price statistics for {selected_region}")
                stats = filtered_df["price_numeric"].describe()

                # Format price display based on magnitude
                def format_price(price_value):
                    if price_value >= 1_000_000_000:
                        return f"Rp {price_value/1_000_000_000:.2f} Miliar"
                    else:
                        return f"Rp {price_value/1_000_000:.2f} Juta"

                min_price = stats["min"]
                max_price = stats["max"]
                median_price = stats["50%"]

                st.metric("Minimum Price", format_price(min_price))
                st.metric("Maximum Price", format_price(max_price))
                st.metric("Median Price", format_price(median_price))

        # Price correlation
        st.subheader("Price Correlation with Property Features")
        corr_cols = [
            "price_numeric",
            "bedroom",
            "bathroom",
            "carport",
            "LT_numeric",
            "LB_numeric",
        ]
        corr_labels = [
            "Price",
            "Bedrooms",
            "Bathrooms",
            "Carport",
            "Land Area",
            "Building Area",
        ]

        # Calculate correlation matrix
        corr_matrix = filtered_df[corr_cols].corr()
        corr_matrix.columns = corr_labels
        corr_matrix.index = corr_labels

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax
        )
        st.pyplot(fig)

    with tab3:
        st.header("Property Feature Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bedroom vs. Price")
            # Determine price display unit
            use_billions = filtered_df["price_numeric"].mean() > 1_000_000_000

            if use_billions:
                y_axis = "price_display"
                y_title = "Price (Miliar Rp)"
            else:
                y_axis = "price_display"
                y_title = "Price (Juta Rp)"

            fig = px.scatter(
                filtered_df,
                x="bedroom",
                y=y_axis,
                size="LB_numeric",
                color="kabupaten_kota" if selected_region == "All Regions" else None,
                hover_data=["bathroom", "LT_numeric", "title"],
                title="Price vs. Number of Bedrooms",
                labels={
                    "bedroom": "Bedrooms",
                    y_axis: y_title,
                    "LB_numeric": "Building Area (m²)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Building Area vs. Land Area")
            # Determine the unit based on price range
            use_billions = filtered_df["price_numeric"].mean() > 1_000_000_000

            if use_billions:
                color_axis = "price_display"
                color_title = "Price (Miliar Rp)"
            else:
                color_axis = "price_display"
                color_title = "Price (Juta Rp)"

            fig = px.scatter(
                filtered_df,
                x="LT_numeric",
                y="LB_numeric",
                size="price_numeric",
                color=color_axis,
                hover_data=["bedroom", "bathroom", "kabupaten_kota", "title"],
                title="Building Area vs. Land Area",
                labels={
                    "LT_numeric": "Land Area (m²)",
                    "LB_numeric": "Building Area (m²)",
                    color_axis: color_title,
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature distribution
        st.subheader("Feature Distributions")

        feature_map = {
            "bedroom": {"col": "bedroom", "title": "Bedrooms"},
            "bathroom": {"col": "bathroom", "title": "Bathrooms"},
            "carport": {"col": "carport", "title": "Carport"},
            "land_area": {"col": "LT_numeric", "title": "Land Area (m²)"},
            "building_area": {"col": "LB_numeric", "title": "Building Area (m²)"},
        }

        selected_feature_key = st.selectbox(
            "Select Feature to Analyze",
            options=list(feature_map.keys()),
            format_func=lambda x: feature_map[x]["title"],
        )

        selected_feature = feature_map[selected_feature_key]["col"]
        feature_title = feature_map[selected_feature_key]["title"]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                filtered_df,
                x=selected_feature,
                title=f"{feature_title} Distribution",
                nbins=20,
                labels={selected_feature: feature_title},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if selected_region == "All Regions":
                fig = px.box(
                    df,
                    x="kabupaten_kota",
                    y=selected_feature,
                    title=f"{feature_title} by Region",
                    labels={
                        "kabupaten_kota": "Region",
                        selected_feature: feature_title,
                    },
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"{feature_title} statistics for {selected_region}")
                stats = filtered_df[selected_feature].describe()
                cols = st.columns(4)
                cols[0].metric("Min", f"{stats['min']:.1f}")
                cols[1].metric("Max", f"{stats['max']:.1f}")
                cols[2].metric("Median", f"{stats['50%']:.1f}")
                cols[3].metric("Mean", f"{stats['mean']:.1f}")

        # Property type analysis (if badges contain this info)
        if "badges" in df.columns:
            st.subheader("Property Types & Features")

            # Extract property types from badges
            def extract_property_types(badges_str):
                if isinstance(badges_str, str):
                    return badges_str.split(", ")
                return []

            # Create a new column with list of property types
            filtered_df["property_features"] = filtered_df["badges"].apply(
                extract_property_types
            )

            # Count occurrences of each property type
            property_types = {}
            for features_list in filtered_df["property_features"]:
                for feature in features_list:
                    if feature in property_types:
                        property_types[feature] += 1
                    else:
                        property_types[feature] = 1

            # Create dataframe for visualization
            property_df = pd.DataFrame(
                {
                    "Feature": list(property_types.keys()),
                    "Count": list(property_types.values()),
                }
            ).sort_values(by="Count", ascending=False)

            fig = px.bar(
                property_df,
                x="Feature",
                y="Count",
                title="Popular Property Features",
                labels={"Feature": "Property Feature", "Count": "Number of Properties"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.error(
        "No data available. Please check if the dataset has been properly cleaned and saved."
    )

# Add footer
st.markdown("---")
st.markdown("Housing Price Analysis Dashboard | Data from houses-cleaned.csv")
