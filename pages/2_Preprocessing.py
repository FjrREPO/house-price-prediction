import os
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Preprocessing Dataset",
    page_icon="🔄",
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
</style>
""",
    unsafe_allow_html=True,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "../preprocessing/output")

st.markdown(
    '<div class="main-header">🔄 Preprocessing Dataset</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Pantau dan analisis hasil pemrosesan data properti termasuk dataset yang diterima, ditolak, dan diproses.</div>',
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
        "Accepted Data",
        "Rejected Data",
        "Processed Data",
        "Cleaned Data",
    ]
    selected_section = st.radio("Pilih bagian:", nav_options)

    st.markdown("### ℹ️ Tentang")
    st.info(
        "Preprocessing ini menampilkan hasil pemrosesan data properti. "
        "Gunakan filter di atas untuk menyesuaikan tampilan Anda dan navigasi untuk menuju bagian tertentu."
    )

    if st.button("🔄 Segarkan Data"):
        st.cache_data.clear()
        st.success("Data berhasil diperbarui!")


@st.cache_data(ttl=3600)
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        return None


def get_csv_files(folder_path):
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith(".csv")]


def get_data_metrics():
    categories = ["accepted", "rejected", "processed", "cleaned"]
    metrics = {}

    for category in categories:
        folder_path = os.path.join(output_dir, category)
        files = get_csv_files(folder_path)
        total_rows = 0

        for file in files:
            file_path = os.path.join(folder_path, file)
            df = load_csv(file_path)
            if df is not None:
                total_rows += df.shape[0]

        metrics[category] = {"file_count": len(files), "row_count": total_rows}

    return metrics


metrics = get_data_metrics()

if selected_section == "Overview" or selected_section == nav_options[0]:
    st.markdown(
        '<div class="section-header">📊 Ringkasan</div>', unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("File Diterima", metrics["accepted"]["file_count"])
        st.markdown(f"Jumlah Baris: {metrics['accepted']['row_count']:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("File Ditolak", metrics["rejected"]["file_count"])
        st.markdown(f"Jumlah Baris: {metrics['rejected']['row_count']:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("File Diproses", metrics["processed"]["file_count"])
        st.markdown(f"Jumlah Baris: {metrics['processed']['row_count']:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("File Bersih", metrics["cleaned"]["file_count"])
        st.markdown(f"Jumlah Baris: {metrics['cleaned']['row_count']:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-header">📈 Distribusi Data</div>', unsafe_allow_html=True
    )

    files_data = {
        "Kategori": list(metrics.keys()),
        "File": [metrics[cat]["file_count"] for cat in metrics],
    }
    files_df = pd.DataFrame(files_data)

    rows_data = {
        "Kategori": list(metrics.keys()),
        "Baris": [metrics[cat]["row_count"] for cat in metrics],
    }
    rows_df = pd.DataFrame(rows_data)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            files_df,
            x="Kategori",
            y="File",
            title="File Berdasarkan Kategori",
            color="Kategori",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig1.update_layout(xaxis_title="", yaxis_title="Jumlah File")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            rows_df,
            x="Kategori",
            y="Baris",
            title="Baris Berdasarkan Kategori",
            color="Kategori",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig2.update_layout(xaxis_title="", yaxis_title="Jumlah Baris")
        st.plotly_chart(fig2, use_container_width=True)


def display_csv_files(folder_path, label, icon, description, show_rows=10):
    st.markdown(
        f'<div class="section-header">{icon} Data {label}</div>', unsafe_allow_html=True
    )
    st.markdown(description)

    files = get_csv_files(folder_path)

    if not files:
        st.info(f"Tidak ada file data {label.lower()}.")
        return

    if len(files) > 1:
        tabs = st.tabs([f"{i+1}. {file}" for i, file in enumerate(files)])

        for i, (tab, file) in enumerate(zip(tabs, files)):
            with tab:
                file_path = os.path.join(folder_path, file)
                df = load_csv(file_path)

                if df is not None:
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        st.markdown('<div class="file-stats">', unsafe_allow_html=True)
                        st.markdown(f"**Baris:** {df.shape[0]:,}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="file-stats">', unsafe_allow_html=True)
                        st.markdown(f"**Kolom:** {df.shape[1]:,}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="file-stats">', unsafe_allow_html=True)
                        st.markdown(f"**File:** {file}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.dataframe(df.head(show_rows), use_container_width=True)

                    with st.expander("Informasi Kolom"):
                        col_info = pd.DataFrame(
                            {
                                "Kolom": df.columns,
                                "Tipe": df.dtypes.astype(str),
                                "Nilai Non-Null": df.count().values,
                                "Nilai Null": df.isnull().sum().values,
                                "Persen Null": (df.isnull().sum() / len(df) * 100)
                                .round(2)
                                .astype(str)
                                + "%",
                            }
                        )
                        st.dataframe(col_info, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            f"Tampilkan Data Lengkap #{i+1}", key=f"preview_{label}_{i}"
                        ):
                            st.dataframe(df, use_container_width=True)
                    with col2:
                        st.download_button(
                            label=f"Unduh Data #{i+1}",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name=file,
                            mime="text/csv",
                            key=f"download_{label}_{i}",
                        )
    else:
        for file in files:
            file_path = os.path.join(folder_path, file)
            df = load_csv(file_path)

            if df is not None:
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.markdown('<div class="file-stats">', unsafe_allow_html=True)
                    st.markdown(f"**Baris:** {df.shape[0]:,}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="file-stats">', unsafe_allow_html=True)
                    st.markdown(f"**Kolom:** {df.shape[1]:,}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="file-stats">', unsafe_allow_html=True)
                    st.markdown(f"**File:** {file}")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.dataframe(df.head(show_rows), use_container_width=True)

                with st.expander("Informasi Kolom"):
                    col_info = pd.DataFrame(
                        {
                            "Kolom": df.columns,
                            "Tipe": df.dtypes.astype(str),
                            "Nilai Non-Null": df.count().values,
                            "Nilai Null": df.isnull().sum().values,
                            "Persen Null": (df.isnull().sum() / len(df) * 100)
                            .round(2)
                            .astype(str)
                            + "%",
                        }
                    )
                    st.dataframe(col_info, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Tampilkan Data Lengkap", key=f"preview_{label}_0"):
                        st.dataframe(df, use_container_width=True)
                with col2:
                    st.download_button(
                        label="Unduh Data",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name=file,
                        mime="text/csv",
                    )


if selected_section == "Accepted Data" or selected_section == nav_options[1]:
    display_csv_files(
        os.path.join(output_dir, "accepted"),
        "Diterima",
        "✅",
        "Ini adalah file data properti yang diterima untuk pemrosesan lebih lanjut.",
        show_rows,
    )

if selected_section == "Rejected Data" or selected_section == nav_options[2]:
    display_csv_files(
        os.path.join(output_dir, "rejected"),
        "Ditolak",
        "❌",
        "Ini adalah file data properti yang ditolak dari pipeline pemrosesan.",
        show_rows,
    )

if selected_section == "Processed Data" or selected_section == nav_options[3]:
    display_csv_files(
        os.path.join(output_dir, "processed"),
        "Diproses",
        "⚙️",
        "Ini adalah file data properti yang telah diproses dengan transformasi awal.",
        show_rows,
    )

if selected_section == "Cleaned Data" or selected_section == nav_options[4]:
    display_csv_files(
        os.path.join(output_dir, "cleaned"),
        "Bersih",
        "🧹",
        "Ini adalah file data properti yang telah dibersihkan dan siap untuk analisis.",
        show_rows,
    )

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
    Preprocessing Dataset
    </div>
    """,
    unsafe_allow_html=True,
)
