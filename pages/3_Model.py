import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px

# Set konfigurasi halaman
st.set_page_config(
    page_title="Pelatihan Model Prediksi Harga", page_icon="🧠", layout="wide"
)

st.title("🧠 Pelatihan Model Prediksi Harga Perumahan")
st.markdown(
    """
    Halaman ini memungkinkan Anda untuk melatih model prediksi harga perumahan menggunakan 
    algoritma Random Forest dengan optimasi Algoritma Genetika (GA). Anda dapat menyesuaikan 
    parameter pelatihan dan melihat perbandingan performa.
    """
)


# Fungsi-fungsi model RF
def predict_price_rf(
    model, bedroom_val, bathroom_val, LT_val, LB_val, fallback_value=None
):
    input_data = pd.DataFrame(
        {
            "bedroom": [bedroom_val],
            "bathroom": [bathroom_val],
            "LT": [LT_val],
            "LB": [LB_val],
        }
    )
    try:
        prediction = model.predict(input_data)[0]
        return prediction
    except Exception as e:
        st.error(f"Kesalahan prediksi RF: {e}")
        return fallback_value if fallback_value is not None else 0


def train_base_rf_model(X, y):
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    base_model.fit(X, y)
    return base_model


def evaluate_rf_model(model, X, y):
    predictions = []
    for i in range(len(X)):
        prediction = predict_price_rf(
            model,
            X.iloc[i]["bedroom"],
            X.iloc[i]["bathroom"],
            X.iloc[i]["LT"],
            X.iloc[i]["LB"],
            fallback_value=y.iloc[i],
        )
        predictions.append(prediction)
    mape = mean_absolute_percentage_error(y, predictions)
    return mape, predictions


def train_optimized_rf_model(X, y, params):
    n_estimators = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    model.fit(X, y)
    return model


# Implementasi algoritma genetika
class GAOptimizer:
    def __init__(self):
        self.initial_population = {"Individu": [], "Parameter": []}
        self.generations_data = {
            "Generasi": [],
            "Best Fitness": [],
            "MAPE": [],
            "Mutation": [],
        }
        self.evaluation_data = {}
        self.best_individuals = {}

    def save_log(self, generation, individual_index, log_type, log_content):
        if log_type == "initial":
            self.initial_population.setdefault("Individu", []).append(
                f"Individu {individual_index + 1}"
            )
            self.initial_population.setdefault("Parameter", []).append(log_content)

        elif log_type == "fitness":
            gen_data = self.evaluation_data.setdefault(
                generation, {"Individu": [], "MAPE (%)": [], "Fitness": []}
            )
            gen_data["Individu"].append(f"Individu {individual_index + 1}")
            gen_data["MAPE (%)"].append(log_content.get("MAPE"))
            gen_data["Fitness"].append(log_content.get("Fitness"))

        elif log_type in ["crossover", "mutation"]:
            gen_data = self.generations_data.setdefault(generation, {})
            gen_data.setdefault(log_type.capitalize(), []).append(log_content)

        elif log_type == "elitism":
            self.best_individuals[generation] = log_content

        elif log_type == "final":
            self.generations_data.setdefault("Generasi", []).append(generation)
            self.generations_data.setdefault("Best Fitness", []).append(
                log_content.get("Best Fitness")
            )
            self.generations_data.setdefault("MAPE", []).append(log_content.get("MAPE"))

    def evaluate_rf_model(self, params, data, price_target):
        n_estimators, max_depth, min_samples_split, min_samples_leaf = params

        n_estimators = int(n_estimators)
        max_depth = int(max_depth) if max_depth > 0 else None
        min_samples_split = int(min_samples_split)
        min_samples_leaf = int(min_samples_leaf)

        model = train_optimized_rf_model(data, price_target, params)

        predictions = []
        for i in range(len(data)):
            prediction = predict_price_rf(
                model,
                data.iloc[i]["bedroom"],
                data.iloc[i]["bathroom"],
                data.iloc[i]["LT"],
                data.iloc[i]["LB"],
                fallback_value=price_target.iloc[i],
            )
            predictions.append(prediction)

        mape = mean_absolute_percentage_error(price_target, predictions)
        return mape

    def fitness_function(self, params, data, price_target):
        return -self.evaluate_rf_model(params, data, price_target)

    def create_initial_population(self, size, object_bounds):
        population = []
        for _ in range(size):
            individual = tuple(
                random.uniform(lower_bound, upper_bound)
                for lower_bound, upper_bound in object_bounds
            )
            population.append(individual)
        return population

    def selection(self, population, fitnesses, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(
                list(zip(population, fitnesses)), tournament_size
            )
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        alpha = random.random()
        child1 = tuple(
            alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)
        )
        child2 = tuple(
            alpha * p2 + (1 - alpha) * p1 for p1, p2 in zip(parent1, parent2)
        )
        return child1, child2

    def mutation(self, individual, mutation_rate, object_bounds):
        individual = list(individual)
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                lower_bound, upper_bound = object_bounds[i]
                mutation_amount = random.uniform(-0.1, 0.1) * (
                    upper_bound - lower_bound
                )
                individual[i] += mutation_amount
                individual[i] = max(min(individual[i], upper_bound), lower_bound)
        return tuple(individual)

    def genetic_algorithm(
        self,
        population_size,
        object_bounds,
        generations,
        mutation_rate,
        data,
        price_target,
        progress_bar=None,
        status_text=None,
    ):
        population = self.create_initial_population(population_size, object_bounds)

        for i, ind in enumerate(population):
            self.save_log(
                generation=0, individual_index=i, log_type="initial", log_content=ind
            )

        best_performers = []

        for generation in range(1, generations + 1):
            if status_text:
                status_text.text(f"Generasi {generation}/{generations}")
            if progress_bar:
                progress_bar.progress(generation / generations)

            fitnesses = [
                self.fitness_function(ind, data, price_target=price_target)
                for ind in population
            ]
            for i, (ind, fitness) in enumerate(zip(population, fitnesses)):
                log_content = {"MAPE": -fitness, "Fitness": fitness}
                self.save_log(generation, i, "fitness", log_content)

            best_index = fitnesses.index(max(fitnesses))
            best_individual = population[best_index]
            best_fitness = fitnesses[best_index]
            best_mape = -best_fitness
            best_performers.append((best_individual, best_fitness))

            self.generations_data["Generasi"].append(generation)
            self.generations_data["Best Fitness"].append(best_fitness)
            self.generations_data["MAPE"].append(best_mape)

            best_individual_data = {
                "Parameter": best_individual,
                "Fitness": best_fitness,
            }
            self.save_log(generation, best_index, "elitism", best_individual_data)

            selected_population = self.selection(population, fitnesses)

            next_population = []
            self.save_log(generation, -1, "crossover", "Fase Crossover")
            for i in range(0, len(selected_population) - 1, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                crossover_info = {
                    "Parent1": parent1,
                    "Parent2": parent2,
                    "Child1": child1,
                    "Child2": child2,
                }
                self.save_log(generation, i, "crossover", crossover_info)
                next_population.extend([child1, child2])

            if len(selected_population) % 2 != 0:
                next_population.append(selected_population[-1])
                self.save_log(
                    generation,
                    -1,
                    "crossover",
                    f"Individu ganjil terbawa ke generasi berikutnya: {selected_population[-1]}",
                )

            mutated_population = []
            self.save_log(generation, -1, "mutation", "Fase Mutasi")
            for i, ind in enumerate(next_population):
                mutated = self.mutation(ind, mutation_rate, object_bounds)
                mutation_info = {"Original": ind, "Mutated": mutated}
                self.save_log(generation, i, "mutation", mutation_info)
                mutated_population.append(mutated)

            mutated_population[0] = best_individual
            self.save_log(generation, 0, "elitism", best_individual)

            population = mutated_population

        final_best_individual = max(best_performers, key=lambda x: x[1])[0]
        final_best_fitness = max(best_performers, key=lambda x: x[1])[1]
        final_best_mape = -final_best_fitness
        final_results = {
            "Best Individual": final_best_individual,
            "Fitness": final_best_fitness,
            "MAPE": final_best_mape,
        }
        self.save_log(generations, -1, "final", final_results)

        return final_best_individual, best_performers

    def save_logs_to_json(self, file_path):
        data = {
            "initial_population": self.initial_population,
            "evaluation_data": self.evaluation_data,
            "generations_data": self.generations_data,
            "best_individuals": self.best_individuals,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


# Fungsi pemrosesan data
def preprocess_data(df):
    df_processed = df.copy()

    def convert_to_numeric(value):
        if pd.isna(value) or not isinstance(value, str):
            return np.nan

        try:
            value_numeric = re.sub(r"Rp\s?", "", value)

            if "Miliar" in value_numeric:
                value_numeric = (
                    float(re.sub(r"\s?Miliar", "", value_numeric).replace(",", "."))
                    * 1e9
                )
            elif "Juta" in value_numeric:
                value_numeric = (
                    float(re.sub(r"\s?Juta", "", value_numeric).replace(",", ".")) * 1e6
                )
            else:
                return np.nan
            return value_numeric
        except ValueError:
            return np.nan

    def extract_area(area_str):
        if pd.isna(area_str) or not isinstance(area_str, str):
            return np.nan

        match = re.search(r":\s*(\d+)\s*m²", area_str)
        if match:
            return float(match.group(1))
        return np.nan

    def extract_location(loc_str):
        if pd.isna(loc_str) or not isinstance(loc_str, str):
            return "Tidak Diketahui"

        parts = loc_str.split(",")
        if len(parts) > 1:
            return parts[1].strip()
        return parts[0].strip()

    # Konversi kolom
    df_processed["price"] = df_processed["price"].apply(convert_to_numeric)
    df_processed["LT"] = df_processed["LT"].apply(extract_area)
    df_processed["LB"] = df_processed["LB"].apply(extract_area)
    df_processed["kabupaten_kota"] = df_processed["location"].apply(extract_location)

    # Drop missing values
    df_processed = df_processed.dropna(
        subset=["price", "LT", "LB", "bedroom", "bathroom"]
    )

    # Konversi tipe data
    df_processed["bedroom"] = df_processed["bedroom"].astype(float)
    df_processed["bathroom"] = df_processed["bathroom"].astype(float)

    return df_processed


def remove_outliers(df, columns):
    df_clean = df.copy()

    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            ]

    return df_clean


# Fungsi memplot hasil
def plot_training_progress(ga_optimizer):
    if not ga_optimizer.generations_data["Generasi"]:
        st.warning("Belum ada data pelatihan")
        return

    # Plot MAPE per generasi
    fig = px.line(
        x=ga_optimizer.generations_data["Generasi"],
        y=ga_optimizer.generations_data["MAPE"],
        labels={"x": "Generasi", "y": "MAPE (%)"},
        title="Perkembangan MAPE per Generasi",
    )
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)


def plot_rf_comparison(base_mape, optimized_mape):
    models = ["RF Dasar", "RF Teroptimasi"]
    mapes = [base_mape, optimized_mape]

    fig = px.bar(
        x=models,
        y=[m * 100 for m in mapes],
        labels={"x": "Model", "y": "MAPE (%)"},
        title="Perbandingan Performa Model Random Forest",
    )

    # Tambahkan teks nilai di atas bar
    for i, mape in enumerate(mapes):
        fig.add_annotation(
            x=models[i],
            y=mape * 100,
            text=f"{mape*100:.2f}%",
            showarrow=False,
            yshift=10,
        )

    st.plotly_chart(fig, use_container_width=True)


def save_model(model, filename="optimized_rf_model.pkl"):
    # Simpan model ke file
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    st.success(f"Model berhasil disimpan ke {model_path}")
    return model_path


# Tambahkan fungsi untuk random
import random


# Halaman utama
@st.cache_data
def load_data():
    try:
        data_path = os.path.join("dataset", "houses-cleaned.csv")
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(
            "File data tidak ditemukan. Silakan periksa jalur ke houses-cleaned.csv"
        )
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Kesalahan memuat data: {e}")
        return pd.DataFrame()


# Sidebar untuk parameter pelatihan
st.sidebar.header("Parameter Pelatihan")

tab1, tab2 = st.tabs(["📊 Pelatihan Model", "📈 Hasil & Perbandingan"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data dan Parameter")

        df = load_data()
        if not df.empty:
            st.write(f"Jumlah total data: {len(df)}")

            # Tampilkan dataset yang dimuat
            st.write("Pratinjau Data:")
            st.dataframe(df.head(5), use_container_width=True)

            # Preprocessing data
            with st.spinner("Memproses data..."):
                df_preprocessed = preprocess_data(df)
                st.write(f"Data setelah preprocessing: {len(df_preprocessed)} baris")

            # Pilih region
            regions = ["Semua Wilayah"] + sorted(
                df_preprocessed["kabupaten_kota"].unique().tolist()
            )
            selected_region = st.selectbox("Pilih Wilayah untuk Pelatihan", regions)

            if selected_region != "Semua Wilayah":
                filtered_df = df_preprocessed[
                    df_preprocessed["kabupaten_kota"] == selected_region
                ]
            else:
                filtered_df = df_preprocessed

            st.write(f"Data untuk pelatihan: {len(filtered_df)} baris")

            # Opsi untuk menghapus outlier
            remove_outliers_option = st.checkbox("Hapus Outlier", value=True)

            if remove_outliers_option:
                columns_to_check = ["price", "bedroom", "bathroom", "LT", "LB"]
                filtered_df = remove_outliers(filtered_df, columns_to_check)
                st.write(f"Data setelah menghapus outlier: {len(filtered_df)} baris")

    with col2:
        st.subheader("Parameter Algoritma Genetika")

        population_size = st.slider("Ukuran Populasi", 10, 100, 30)
        generations = st.slider("Jumlah Generasi", 1, 20, 5)
        mutation_rate = st.slider("Tingkat Mutasi", 0.0, 1.0, 0.35)

        st.write("Batas Parameter Random Forest:")
        col1, col2 = st.columns(2)

        with col1:
            n_estimators_min = st.number_input("n_estimators (min)", 10, 100, 50)
            n_estimators_max = st.number_input(
                "n_estimators (max)", n_estimators_min, 500, 200
            )

            max_depth_min = st.number_input("max_depth (min)", 1, 10, 5)
            max_depth_max = st.number_input("max_depth (max)", max_depth_min, 50, 30)

        with col2:
            min_samples_split_min = st.number_input("min_samples_split (min)", 2, 5, 2)
            min_samples_split_max = st.number_input(
                "min_samples_split (max)", min_samples_split_min, 20, 10
            )

            min_samples_leaf_min = st.number_input("min_samples_leaf (min)", 1, 3, 1)
            min_samples_leaf_max = st.number_input(
                "min_samples_leaf (max)", min_samples_leaf_min, 10, 5
            )

    if not df.empty and len(filtered_df) > 0:
        # Siapkan data untuk pelatihan
        X = filtered_df[["bedroom", "bathroom", "LT", "LB"]]
        y = filtered_df["price"]

        # Tombol untuk memulai pelatihan
        if st.button("Mulai Pelatihan Model", type="primary"):
            # Melatih model RF dasar
            with st.spinner("Melatih model Random Forest dasar..."):
                base_rf_model = train_base_rf_model(X, y)
                base_mape, base_predictions = evaluate_rf_model(base_rf_model, X, y)
                st.success(
                    f"Model RF dasar selesai dilatih. MAPE: {base_mape*100:.2f}%"
                )

            # Melatih model RF yang dioptimasi dengan GA
            with st.spinner("Mengoptimasi model dengan Algoritma Genetika..."):
                object_bounds = [
                    (n_estimators_min, n_estimators_max),
                    (max_depth_min, max_depth_max),
                    (min_samples_split_min, min_samples_split_max),
                    (min_samples_leaf_min, min_samples_leaf_max),
                ]

                ga = GAOptimizer()

                # Tambahkan progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                best_solution, best_performers = ga.genetic_algorithm(
                    population_size,
                    object_bounds,
                    generations,
                    mutation_rate,
                    X,
                    price_target=y,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )

                # Tampilkan parameter terbaik
                n_estimators = int(best_solution[0])
                max_depth = int(best_solution[1])
                min_samples_split = int(best_solution[2])
                min_samples_leaf = int(best_solution[3])

                st.subheader("Parameter Terbaik:")
                param_cols = st.columns(4)
                param_cols[0].metric("n_estimators", n_estimators)
                param_cols[1].metric("max_depth", max_depth)
                param_cols[2].metric("min_samples_split", min_samples_split)
                param_cols[3].metric("min_samples_leaf", min_samples_leaf)

                # Latih model dengan parameter terbaik
                optimized_rf_model = train_optimized_rf_model(X, y, best_solution)
                optimized_rf_mape, optimized_predictions = evaluate_rf_model(
                    optimized_rf_model, X, y
                )

                improvement = ((base_mape - optimized_rf_mape) / base_mape) * 100

                st.success(
                    f"""
                Model RF teroptimasi selesai dilatih. 
                - MAPE: {optimized_rf_mape*100:.2f}%
                - Peningkatan: {improvement:.2f}%
                """
                )

                # Simpan model
                model_path = save_model(optimized_rf_model)

                # Simpan hasil optimasi
                ga.save_logs_to_json(os.path.join("model", "ga_optimization_logs.json"))

                # Simpan hasil ke session_state untuk tab perbandingan
                st.session_state["base_mape"] = base_mape
                st.session_state["optimized_mape"] = optimized_rf_mape
                st.session_state["ga_optimizer"] = ga
                st.session_state["best_solution"] = best_solution
                st.session_state["model_trained"] = True

                # Tampilkan grafik MAPE per generasi
                st.subheader("Perkembangan MAPE selama Pelatihan")
                plot_training_progress(ga)

                # Tampilkan grafik perbandingan
                st.subheader("Perbandingan Model")
                plot_rf_comparison(base_mape, optimized_rf_mape)
    else:
        st.warning(
            "Tidak ada data tersedia untuk pelatihan. Silakan periksa data dan filter."
        )

with tab2:
    st.subheader("Hasil dan Perbandingan Model")

    # Cek apakah model sudah dilatih
    if "model_trained" in st.session_state and st.session_state["model_trained"]:
        # Tampilkan parameter terbaik
        st.subheader("Parameter Terbaik Model Random Forest Teroptimasi")
        best_solution = st.session_state["best_solution"]
        n_estimators = int(best_solution[0])
        max_depth = int(best_solution[1])
        min_samples_split = int(best_solution[2])
        min_samples_leaf = int(best_solution[3])

        param_cols = st.columns(4)
        param_cols[0].metric("n_estimators", n_estimators)
        param_cols[1].metric("max_depth", max_depth)
        param_cols[2].metric("min_samples_split", min_samples_split)
        param_cols[3].metric("min_samples_leaf", min_samples_leaf)

        # Tampilkan perbandingan MAPE
        base_mape = st.session_state["base_mape"]
        optimized_mape = st.session_state["optimized_mape"]
        improvement = ((base_mape - optimized_mape) / base_mape) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("MAPE Model Dasar", f"{base_mape*100:.2f}%")
        col2.metric("MAPE Model Teroptimasi", f"{optimized_mape*100:.2f}%")
        col3.metric("Peningkatan", f"{improvement:.2f}%")

        # Tampilkan grafik perbandingan
        plot_rf_comparison(base_mape, optimized_mape)

        # Tampilkan grafik progres pelatihan
        if "ga_optimizer" in st.session_state:
            plot_training_progress(st.session_state["ga_optimizer"])

        # Tampilkan daftar file model yang tersedia
        st.subheader("File Model Tersedia")
        model_files = []
        model_dir = "model"
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

        if model_files:
            for model_file in model_files:
                st.write(f"📄 {model_file}")
        else:
            st.info("Tidak ada file model yang tersimpan.")
    else:
        st.info(
            "Belum ada model yang dilatih. Silakan ke tab 'Pelatihan Model' untuk melatih model."
        )

# Footer
st.markdown("---")
st.markdown(
    "Dashboard Pelatihan Model Prediksi Harga Perumahan | By: Data Science Team"
)
