import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re


st.set_page_config(page_title="Preprocessing", page_icon="🏠", layout="wide")


st.title("🏠 Preprocessing Data")
st.markdown(
    """
Bagian ini memvisualisasikan data perumahan di berbagai wilayah dengan 
prediksi harga menggunakan model logika fuzzy. Jelajahi data melalui 
berbagai grafik dan statistik.
"""
)


@st.cache_data
def load_data():
    try:
        data_path = os.path.join("dataset", "houses-cleaned.csv")
        df = pd.read_csv(data_path)

        def convert_to_numeric(value: str) -> float:
            try:
                value_numeric = re.sub(r"Rp\s?", "", value)

                if "Miliar" in value_numeric:
                    value_numeric = (
                        float(re.sub(r"\s?Miliar", "", value_numeric).replace(",", "."))
                        * 1e9
                    )
                elif "Juta" in value_numeric:
                    value_numeric = (
                        float(re.sub(r"\s?Juta", "", value_numeric).replace(",", "."))
                        * 1e6
                    )
                else:
                    return None
                return value_numeric
            except ValueError:
                return None

        df["price_numeric"] = df["price"].apply(convert_to_numeric)

        def extract_area(area_str):
            if isinstance(area_str, str):

                match = re.search(r":\s*(\d+)\s*m²", area_str)
                if match:
                    return float(match.group(1))
            return np.nan

        df["LT_numeric"] = df["LT"].apply(extract_area)
        df["LB_numeric"] = df["LB"].apply(extract_area)

        def extract_location(loc_str):
            if isinstance(loc_str, str):
                parts = loc_str.split(",")
                if len(parts) > 1:
                    return parts[1].strip()
                return parts[0].strip()
            return "Tidak Diketahui"

        df["kabupaten_kota"] = df["location"].apply(extract_location)

        return df
    except FileNotFoundError:
        st.error(
            "File data tidak ditemukan. Silakan periksa jalur ke houses-cleaned.csv"
        )
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Kesalahan memuat data: {e}")

        if "price" in locals():
            st.write(f"Nilai bermasalah di kolom 'price'")
        return pd.DataFrame()


df = load_data()

if not df.empty:

    if df["price_numeric"].isna().any():
        st.warning(
            f"Ditemukan {df['price_numeric'].isna().sum()} baris dengan data harga tidak valid"
        )

        problem_rows = df[df["price_numeric"].isna()][["title", "price"]].head(5)
        if not problem_rows.empty:
            st.write("Contoh data harga bermasalah:")
            st.write(problem_rows)

    df = df.dropna(subset=["price_numeric", "LT_numeric", "LB_numeric"])

    st.sidebar.header("Filter")

    regions = ["Semua Wilayah"] + sorted(df["kabupaten_kota"].unique().tolist())
    selected_region = st.sidebar.selectbox("Pilih Wilayah", regions)

    min_price = int(df["price_numeric"].min())
    max_price = int(df["price_numeric"].max())
    price_range = st.sidebar.slider(
        "Rentang Harga (dalam juta)",
        min_value=min_price // 1_000_000,
        max_value=max_price // 1_000_000,
        value=(min_price // 1_000_000, max_price // 1_000_000),
    )

    bedrooms = sorted(df["bedroom"].unique().tolist())
    selected_bedrooms = st.sidebar.multiselect(
        "Jumlah Kamar Tidur", options=bedrooms, default=bedrooms
    )

    filtered_df = df.copy()

    if selected_region != "Semua Wilayah":
        filtered_df = filtered_df[filtered_df["kabupaten_kota"] == selected_region]

    filtered_df = filtered_df[
        (filtered_df["price_numeric"] >= price_range[0] * 1_000_000)
        & (filtered_df["price_numeric"] <= price_range[1] * 1_000_000)
    ]

    if selected_bedrooms:
        filtered_df = filtered_df[filtered_df["bedroom"].isin(selected_bedrooms)]

    tab1, tab2, tab3 = st.tabs(["Ikhtisar", "Analisis Harga", "Fitur Properti"])

    with tab1:
        st.header("Ikhtisar Data")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Statistik Ringkasan")

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
                "Harga (Rp)",
                "Kamar Tidur",
                "Kamar Mandi",
                "Carport",
                "Luas Tanah (m²)",
                "Luas Bangunan (m²)",
            ]
            st.dataframe(display_stats, use_container_width=True)

        with col2:
            st.subheader("Distribusi Wilayah")
            if selected_region == "Semua Wilayah":
                region_counts = df["kabupaten_kota"].value_counts().reset_index()
                region_counts.columns = ["Wilayah", "Jumlah"]
                fig = px.pie(region_counts, values="Jumlah", names="Wilayah", hole=0.4)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Menampilkan data untuk {selected_region}")
                st.metric("Jumlah Properti", filtered_df.shape[0])
                avg_price = filtered_df["price_numeric"].mean()
                if avg_price >= 1_000_000_000:
                    st.metric(
                        "Harga Rata-rata", f"Rp {avg_price/1_000_000_000:.2f} Miliar"
                    )
                else:
                    st.metric("Harga Rata-rata", f"Rp {avg_price/1_000_000:.2f} Juta")

        st.subheader("Contoh Data")
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
        st.header("Analisis Harga")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribusi Harga")

            use_billions = filtered_df["price_numeric"].mean() > 1_000_000_000

            if use_billions:

                filtered_df["price_display"] = (
                    filtered_df["price_numeric"] / 1_000_000_000
                )
                price_label = "Harga (Miliar Rp)"
            else:

                filtered_df["price_display"] = filtered_df["price_numeric"] / 1_000_000
                price_label = "Harga (Juta Rp)"

            fig = px.histogram(
                filtered_df,
                x="price_display",
                nbins=30,
                title="Histogram Distribusi Harga",
                labels={"price_display": price_label},
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Harga berdasarkan Wilayah")
            if selected_region == "Semua Wilayah":

                use_billions = df["price_numeric"].mean() > 1_000_000_000

                if use_billions:

                    df["price_display"] = df["price_numeric"] / 1_000_000_000
                    price_label = "Harga (Miliar Rp)"
                else:

                    df["price_display"] = df["price_numeric"] / 1_000_000
                    price_label = "Harga (Juta Rp)"

                fig = px.box(
                    df,
                    x="kabupaten_kota",
                    y="price_display",
                    title="Distribusi Harga berdasarkan Wilayah",
                    labels={"kabupaten_kota": "Wilayah", "price_display": price_label},
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Statistik harga untuk {selected_region}")
                stats = filtered_df["price_numeric"].describe()

                def format_price(price_value):
                    if price_value >= 1_000_000_000:
                        return f"Rp {price_value/1_000_000_000:.2f} Miliar"
                    else:
                        return f"Rp {price_value/1_000_000:.2f} Juta"

                min_price = stats["min"]
                max_price = stats["max"]
                median_price = stats["50%"]

                st.metric("Harga Minimum", format_price(min_price))
                st.metric("Harga Maksimum", format_price(max_price))
                st.metric("Harga Median", format_price(median_price))

        st.subheader("Korelasi Harga dengan Fitur Properti")
        corr_cols = [
            "price_numeric",
            "bedroom",
            "bathroom",
            "carport",
            "LT_numeric",
            "LB_numeric",
        ]
        corr_labels = [
            "Harga",
            "Kamar Tidur",
            "Kamar Mandi",
            "Carport",
            "Luas Tanah",
            "Luas Bangunan",
        ]

        corr_matrix = filtered_df[corr_cols].corr()
        corr_matrix.columns = corr_labels
        corr_matrix.index = corr_labels

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax
        )
        st.pyplot(fig)

    with tab3:
        st.header("Analisis Fitur Properti")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Kamar Tidur vs. Harga")

            use_billions = filtered_df["price_numeric"].mean() > 1_000_000_000

            if use_billions:
                y_axis = "price_display"
                y_title = "Harga (Miliar Rp)"
            else:
                y_axis = "price_display"
                y_title = "Harga (Juta Rp)"

            fig = px.scatter(
                filtered_df,
                x="bedroom",
                y=y_axis,
                size="LB_numeric",
                color="kabupaten_kota" if selected_region == "Semua Wilayah" else None,
                hover_data=["bathroom", "LT_numeric", "title"],
                title="Harga vs. Jumlah Kamar Tidur",
                labels={
                    "bedroom": "Kamar Tidur",
                    y_axis: y_title,
                    "LB_numeric": "Luas Bangunan (m²)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Luas Bangunan vs. Luas Tanah")

            use_billions = filtered_df["price_numeric"].mean() > 1_000_000_000

            if use_billions:
                color_axis = "price_display"
                color_title = "Harga (Miliar Rp)"
            else:
                color_axis = "price_display"
                color_title = "Harga (Juta Rp)"

            fig = px.scatter(
                filtered_df,
                x="LT_numeric",
                y="LB_numeric",
                size="price_numeric",
                color=color_axis,
                hover_data=["bedroom", "bathroom", "kabupaten_kota", "title"],
                title="Luas Bangunan vs. Luas Tanah",
                labels={
                    "LT_numeric": "Luas Tanah (m²)",
                    "LB_numeric": "Luas Bangunan (m²)",
                    color_axis: color_title,
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribusi Fitur")

        feature_map = {
            "bedroom": {"col": "bedroom", "title": "Kamar Tidur"},
            "bathroom": {"col": "bathroom", "title": "Kamar Mandi"},
            "carport": {"col": "carport", "title": "Carport"},
            "land_area": {"col": "LT_numeric", "title": "Luas Tanah (m²)"},
            "building_area": {"col": "LB_numeric", "title": "Luas Bangunan (m²)"},
        }

        selected_feature_key = st.selectbox(
            "Pilih Fitur untuk Dianalisis",
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
                title=f"Distribusi {feature_title}",
                nbins=20,
                labels={selected_feature: feature_title},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if selected_region == "Semua Wilayah":
                fig = px.box(
                    df,
                    x="kabupaten_kota",
                    y=selected_feature,
                    title=f"{feature_title} berdasarkan Wilayah",
                    labels={
                        "kabupaten_kota": "Wilayah",
                        selected_feature: feature_title,
                    },
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Statistik {feature_title} untuk {selected_region}")
                stats = filtered_df[selected_feature].describe()
                cols = st.columns(4)
                cols[0].metric("Min", f"{stats['min']:.1f}")
                cols[1].metric("Max", f"{stats['max']:.1f}")
                cols[2].metric("Median", f"{stats['50%']:.1f}")
                cols[3].metric("Rata-rata", f"{stats['mean']:.1f}")

        if "badges" in df.columns:
            st.subheader("Tipe & Fitur Properti")

            def extract_property_types(badges_str):
                if isinstance(badges_str, str):
                    return badges_str.split(", ")
                return []

            filtered_df["property_features"] = filtered_df["badges"].apply(
                extract_property_types
            )

            property_types = {}
            for features_list in filtered_df["property_features"]:
                for feature in features_list:
                    if feature in property_types:
                        property_types[feature] += 1
                    else:
                        property_types[feature] = 1

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
                title="Fitur Properti Populer",
                labels={"Feature": "Fitur Properti", "Count": "Jumlah Properti"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.error(
        "Tidak ada data tersedia. Mohon periksa apakah dataset telah dibersihkan dan disimpan dengan benar."
    )


st.markdown("---")
st.markdown("Preprocessing section")
