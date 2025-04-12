import os
import re
import numpy as np
import pandas as pd
import skfuzzy as fuzz

from skfuzzy import control as ctrl
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error


current_dir = os.path.dirname(os.path.abspath(__file__))

output_dirs = [
    os.path.join(current_dir, "evaluation"),
    os.path.join(current_dir, "output"),
    os.path.join(current_dir, "output/accepted"),
    os.path.join(current_dir, "output/rejected"),
    os.path.join(current_dir, "output/processed"),
    os.path.join(current_dir, "output/cleaned"),
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)


df = pd.read_csv(os.path.join(current_dir, "../dataset/houses.csv"))


def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


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


def preprocess_updated(updated):
    today = datetime.today()

    if "minggu" in updated:
        weeks_ago = int(re.search(r"(\d+)", updated).group(1))
        return today - timedelta(weeks=weeks_ago)
    elif "bulan" in updated:
        months_ago = int(re.search(r"(\d+)", updated).group(1))
        return today - timedelta(days=months_ago * 30)
    elif "hari" in updated:
        days_ago = int(re.search(r"(\d+)", updated).group(1))
        return today - timedelta(days=days_ago)
    elif "jam" in updated:
        hours_ago = int(re.search(r"(\d+)", updated).group(1))
        return today - timedelta(hours=hours_ago)


def preprocess_area(area_str):
    if isinstance(area_str, str):
        match = re.search(r"(\d+(\.\d+)?)\s*m²", area_str)
        if match:
            return float(match.group(1))
    return None


def preprocess_location(location):
    if pd.isna(location):
        return pd.Series([None, None])
    split_location = location.split(",")
    kecamatan = split_location[0].strip() if len(split_location) > 0 else None
    kabupaten_kota = split_location[1].strip() if len(split_location) > 1 else None
    return pd.Series([kecamatan, kabupaten_kota])


def preprocess_data(df):
    df_processed = df.copy()

    df_processed["price"] = df_processed["price"].apply(convert_to_numeric)
    df_processed["LT"] = df_processed["LT"].apply(preprocess_area)
    df_processed["LB"] = df_processed["LB"].apply(preprocess_area)
    df_processed["updated"] = df_processed["updated"].apply(preprocess_updated)
    df_processed[["kecamatan", "kabupaten_kota"]] = df_processed["location"].apply(
        preprocess_location
    )

    df_processed["updated"] = pd.to_datetime(df_processed["updated"])

    df_processed["price"] = df_processed["price"].fillna(df_processed["price"].median())
    df_processed["LT"] = df_processed["LT"].fillna(df_processed["LT"].median())
    df_processed["LB"] = df_processed["LB"].fillna(df_processed["LB"].median())

    df_processed = df_processed.dropna(subset=["updated"])

    df_processed = df_processed.dropna(subset=["LT", "LB", "price"])

    df_processed = df_processed[
        ~df_processed["title"].str.contains("hotel|kost|Kos", case=False, na=False)
    ]

    df_processed["carport"] = df_processed["carport"].fillna(0)

    df_processed.dropna(subset=["bedroom", "bathroom"], inplace=True)
    df_processed.dropna(subset=["LT", "LB"], inplace=True)

    sixty_days_ago = datetime.today() - timedelta(days=60)
    df_processed = df_processed[df_processed["updated"] >= sixty_days_ago]

    df_processed = df_processed.drop_duplicates(subset="title", keep="first")

    return df_processed


df.head(10)

df_preprocessed = preprocess_data(df)
df_preprocessed.describe()
df_preprocessed.to_csv(
    os.path.join(current_dir, "output/processed/preprocessed.csv"), index=False
)

df_groupby = df_preprocessed.groupby("kabupaten_kota")

raw_df = pd.DataFrame()
total_count = 0
region_metrics = {}

for name, group in df_groupby:
    df_yogyakarta = group
    df_yogyakarta.head()
    df_yogyakarta.describe()

    columns_to_check = ["price", "bedroom", "bathroom", "LT", "LB"]
    df_yogyakarta_cleaned = remove_outliers(df_yogyakarta, columns_to_check)

    X = df_yogyakarta_cleaned[["bedroom", "bathroom", "LT", "LB"]]
    y = df_yogyakarta_cleaned["price"]

    bathroom = ctrl.Antecedent(
        np.arange(0, max(df_yogyakarta_cleaned["bathroom"]) + 1, 1), "bathroom"
    )
    bedroom = ctrl.Antecedent(
        np.arange(0, max(df_yogyakarta_cleaned["bedroom"]) + 1, 1), "bedroom"
    )
    LT = ctrl.Antecedent(np.arange(0, max(df_yogyakarta_cleaned["LT"]) + 1, 10), "LT")
    LB = ctrl.Antecedent(np.arange(0, max(df_yogyakarta_cleaned["LB"]) + 1, 10), "LB")

    price = ctrl.Consequent(
        np.arange(
            min(df_yogyakarta_cleaned["price"]),
            max(df_yogyakarta_cleaned["price"]) + 1,
            100_000,
        ),
        "price",
    )
    bedroom_quartiles = np.percentile(df_yogyakarta_cleaned["bedroom"], [25, 50, 75])
    bathroom_quartiles = np.percentile(df_yogyakarta_cleaned["bathroom"], [25, 50, 75])
    LT_quartiles = np.percentile(df_yogyakarta_cleaned["LT"], [25, 50, 75])
    LB_quartiles = np.percentile(df_yogyakarta_cleaned["LB"], [25, 50, 75])
    price_quartiles = np.percentile(df_yogyakarta_cleaned["price"], [25, 50, 75])

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
            df_yogyakarta_cleaned["bathroom"].max(),
        ],
    )

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
            df_yogyakarta_cleaned["bedroom"].max(),
        ],
    )

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
            df_yogyakarta_cleaned["LT"].max(),
        ],
    )

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
            df_yogyakarta_cleaned["LB"].max(),
        ],
    )

    price_mean_poor = price_quartiles[0]
    price_mean_average = price_quartiles[1]
    price_mean_good = price_quartiles[2]

    spread_poor = (price_quartiles[1] - price_quartiles[0]) / 2
    spread_average = (price_quartiles[2] - price_quartiles[1]) / 2
    spread_good = (df_yogyakarta_cleaned["price"].max() - price_quartiles[2]) / 2

    price["poor"] = fuzz.gaussmf(price.universe, price_mean_poor, spread_poor)
    price["average"] = fuzz.gaussmf(price.universe, price_mean_average, spread_average)
    price["good"] = fuzz.gaussmf(price.universe, price_mean_good, spread_good)

    rule1 = ctrl.Rule(
        bedroom["good"] & bathroom["good"] & LT["good"] & LB["good"],
        price["good"],
    )
    rule2 = ctrl.Rule(
        bedroom["average"] & bathroom["average"] & LT["average"] & LB["average"],
        price["average"],
    )
    rule3 = ctrl.Rule(
        bedroom["poor"] & bathroom["poor"] & LT["poor"] & LB["poor"],
        price["poor"],
    )
    rule4 = ctrl.Rule(
        bedroom["good"] & bathroom["average"] & LT["good"] & LB["good"],
        price["good"],
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

    pricing_ctrl = ctrl.ControlSystem(
        [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8]
    )

    pricing_sim = ctrl.ControlSystemSimulation(pricing_ctrl)

    def predict_price_fuzzy(bedroom_val, bathroom_val, LT_val, LB_val, fallback=None):
        pricing_sim.input["bedroom"] = bedroom_val
        pricing_sim.input["bathroom"] = bathroom_val
        pricing_sim.input["LT"] = LT_val
        pricing_sim.input["LB"] = LB_val

        pricing_sim.compute()

        return pricing_sim.output["price"] if "price" in pricing_sim.output else 0

    raw_data = []
    accepted_data = []
    rejected_data = []

    threshold_mape = 0.15

    predictions = []
    actual_prices = []
    prediction_errors = []

    for i in range(len(df_yogyakarta_cleaned)):
        prediction = predict_price_fuzzy(
            df_yogyakarta_cleaned.iloc[i]["bedroom"],
            df_yogyakarta_cleaned.iloc[i]["bathroom"],
            df_yogyakarta_cleaned.iloc[i]["LT"],
            df_yogyakarta_cleaned.iloc[i]["LB"],
            fallback=df_yogyakarta_cleaned.iloc[i]["price"],
        )

        expected = df_yogyakarta_cleaned.iloc[i]["price"]
        error = abs(prediction - expected) / expected

        prediction_errors.append(error)

        if error <= threshold_mape:
            raw_data.append(df_yogyakarta_cleaned.iloc[i])
            accepted_data.append(
                {
                    "index": i,
                    "predicted": prediction,
                    "actual": expected,
                    "error": error,
                    "bedroom": df_yogyakarta_cleaned.iloc[i]["bedroom"],
                    "bathroom": df_yogyakarta_cleaned.iloc[i]["bathroom"],
                    "LT": df_yogyakarta_cleaned.iloc[i]["LT"],
                    "LB": df_yogyakarta_cleaned.iloc[i]["LB"],
                }
            )
            print(f"Predicted: {prediction:.2f}, Expected: {expected:.2f}")
            total_count += 1
            print(total_count)
        else:
            rejected_data.append(
                {
                    "index": i,
                    "predicted": prediction,
                    "actual": expected,
                    "error": error,
                    "bedroom": df_yogyakarta_cleaned.iloc[i]["bedroom"],
                    "bathroom": df_yogyakarta_cleaned.iloc[i]["bathroom"],
                    "LT": df_yogyakarta_cleaned.iloc[i]["LT"],
                    "LB": df_yogyakarta_cleaned.iloc[i]["LB"],
                }
            )

        predictions.append(prediction)
        actual_prices.append(expected)

    raw_df = pd.concat([raw_df, pd.DataFrame(raw_data)], ignore_index=True)

    mape = mean_absolute_percentage_error(df_yogyakarta_cleaned["price"], predictions)

    region_metrics[name] = {
        "mape": mape,
        "mean_price": np.mean(actual_prices),
        "median_price": np.median(actual_prices),
        "price_quartiles": price_quartiles,
        "bedroom_quartiles": bedroom_quartiles,
        "bathroom_quartiles": bathroom_quartiles,
        "LT_quartiles": LT_quartiles,
        "LB_quartiles": LB_quartiles,
        "total_properties": len(df_yogyakarta_cleaned),
        "accepted_properties": len(accepted_data),
        "rejected_properties": len(rejected_data),
        "acceptance_rate": (
            len(accepted_data) / len(df_yogyakarta_cleaned)
            if len(df_yogyakarta_cleaned) > 0
            else 0
        ),
    }

    print("quartiels")
    print(price_quartiles)
    print(bedroom_quartiles)
    print(bathroom_quartiles)
    print(LT_quartiles)
    print(LB_quartiles)
    print(
        f"Mean Absolute Percentage Error (MAPE) of the fuzzy model: {mape * 100:.2f}%"
    )

    accepted_df = pd.DataFrame(accepted_data)
    rejected_df = pd.DataFrame(rejected_data) if rejected_data else pd.DataFrame()

    if not accepted_df.empty:
        region_name = name.replace(" ", "_").lower()
        accepted_df.to_csv(
            os.path.join(current_dir, f"output/accepted/accepted_{region_name}.csv"),
            index=False,
        )

    if not rejected_df.empty:
        region_name = name.replace(" ", "_").lower()
        rejected_df.to_csv(
            os.path.join(current_dir, f"output/rejected/rejected_{region_name}.csv"),
            index=False,
        )

raw_df.to_csv(os.path.join(current_dir, "output/processed/accepted_data.csv"), index=False)
print(raw_df.columns.tolist())  # untuk lihat semua kolom

region_summary_df = pd.DataFrame.from_dict(region_metrics, orient="index")
region_summary_df.reset_index(inplace=True)
region_summary_df.rename(columns={"index": "region"}, inplace=True)

region_summary_df.to_csv(
    os.path.join(current_dir, "evaluation/region_summary.csv"), index=False
)

raw_df["price_per_building_sqm"] = raw_df["price"] / raw_df["LB"]
raw_df["price_per_land_sqm"] = raw_df["price"] / raw_df["LT"]

price_insights = []
for region, metrics in region_metrics.items():
    region_df = raw_df[raw_df["kabupaten_kota"] == region]
    if not region_df.empty:
        price_insights.append(
            {
                "region": region,
                "avg_price": region_df["price"].mean(),
                "median_price": region_df["price"].median(),
                "avg_price_per_building_sqm": region_df[
                    "price_per_building_sqm"
                ].mean(),
                "avg_price_per_land_sqm": region_df["price_per_land_sqm"].mean(),
                "min_price": region_df["price"].min(),
                "max_price": region_df["price"].max(),
            }
        )

price_insights_df = pd.DataFrame(price_insights)
price_insights_df.to_csv(
    os.path.join(current_dir, "evaluation/price_insights.csv"), index=False
)

feature_importance = []
for region, metrics in region_metrics.items():
    region_df = raw_df[raw_df["kabupaten_kota"] == region]
    if not region_df.empty:
        corr_bedroom = region_df["bedroom"].corr(region_df["price"])
        corr_bathroom = region_df["bathroom"].corr(region_df["price"])
        corr_LT = region_df["LT"].corr(region_df["price"])
        corr_LB = region_df["LB"].corr(region_df["price"])

        feature_importance.append(
            {
                "region": region,
                "bedroom_importance": corr_bedroom,
                "bathroom_importance": corr_bathroom,
                "land_area_importance": corr_LT,
                "building_area_importance": corr_LB,
            }
        )

feature_importance_df = pd.DataFrame(feature_importance)
feature_importance_df.to_csv(
    os.path.join(current_dir, "evaluation/feature_importance.csv"), index=False
)


raw_df["updated"] = pd.to_datetime(raw_df["updated"])
raw_df["month"] = raw_df["updated"].dt.to_period("M")

market_trends = (
    raw_df.groupby(["month", "kabupaten_kota"])
    .agg(
        {
            "price": ["mean", "median", "count"],
            "price_per_building_sqm": "mean",
            "price_per_land_sqm": "mean",
        }
    )
    .reset_index()
)

market_trends.columns = [
    "month",
    "region",
    "avg_price",
    "median_price",
    "listing_count",
    "avg_price_per_building_sqm",
    "avg_price_per_land_sqm",
]
market_trends["month"] = market_trends["month"].astype(str)
market_trends.to_csv(
    os.path.join(current_dir, "evaluation/market_trends.csv"), index=False
)


summary_stats = {
    "total_properties_analyzed": len(df_preprocessed),
    "total_properties_accepted": total_count,
    "acceptance_rate": (
        total_count / len(df_preprocessed) if len(df_preprocessed) > 0 else 0
    ),
    "regions_analyzed": len(region_metrics),
    "avg_mape_across_regions": np.mean(
        [metrics["mape"] for metrics in region_metrics.values()]
    ),
    "most_expensive_region": (
        max(region_metrics.items(), key=lambda x: x[1]["mean_price"])[0]
        if region_metrics
        else None
    ),
    "least_expensive_region": (
        min(region_metrics.items(), key=lambda x: x[1]["mean_price"])[0]
        if region_metrics
        else None
    ),
    "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv(
    os.path.join(current_dir, "evaluation/analysis_summary.csv"), index=False
)


feature_columns = ["bedroom", "bathroom", "LT", "LB"]
for feature in feature_columns:

    q1 = raw_df[feature].quantile(0.25)
    q2 = raw_df[feature].quantile(0.5)
    q3 = raw_df[feature].quantile(0.75)

    bin_edges = np.percentile(df_yogyakarta_cleaned[feature], [0, 25, 50, 75, 100])
    bin_edges = np.unique(bin_edges)

    raw_df[f"{feature}_category"] = pd.cut(
        raw_df[feature], bins=bin_edges, duplicates="drop", include_lowest=True
    )


price_patterns = []
for region in raw_df["kabupaten_kota"].unique():
    region_df = raw_df[raw_df["kabupaten_kota"] == region]

    for feature in feature_columns:
        for category in ["very_low", "low", "medium", "high"]:
            filtered_df = region_df[region_df[f"{feature}_category"] == category]

            if not filtered_df.empty:
                price_patterns.append(
                    {
                        "region": region,
                        "feature": feature,
                        "category": category,
                        "avg_price": filtered_df["price"].mean(),
                        "median_price": filtered_df["price"].median(),
                        "count": len(filtered_df),
                    }
                )

price_patterns_df = pd.DataFrame(price_patterns)
price_patterns_df.to_csv(
    os.path.join(current_dir, "./evaluation/price_patterns.csv"), index=False
)

raw_df["price_clean"] = raw_df["price"]


def clean_area(area_str):
    if pd.isna(area_str):
        return np.nan
    number = re.findall(r"[\d,\.]+", area_str)
    if number:
        return float(number[0].replace(",", "."))
    return np.nan


raw_df["land_area"] = raw_df["LT"]
raw_df["building_area"] = raw_df["LB"]

raw_df["is_furnished"] = (
    raw_df["badges"].str.contains("Full Furnished", case=False, na=False).astype(int)
)
raw_df["near_shops"] = (
    raw_df["badges"]
    .str.contains("Dekat Pusat Perbelanjaan", case=False, na=False)
    .astype(int)
)
raw_df["near_school"] = (
    raw_df["badges"].str.contains("Dekat Sekolah", case=False, na=False).astype(int)
)
raw_df["near_healthcare"] = (
    raw_df["badges"]
    .str.contains("Dekat Fasilitas Kesehatan", case=False, na=False)
    .astype(int)
)
raw_df["ready_to_live"] = (
    raw_df["badges"].str.contains("Siap Huni", case=False, na=False).astype(int)
)
raw_df["is_premier"] = (
    raw_df["badges"].str.contains("Premier", case=False, na=False).astype(int)
)


def extract_city(location):
    if pd.isna(location):
        return np.nan
    parts = location.split(",")
    if parts:
        return parts[-1].strip()
    return np.nan


raw_df["city"] = raw_df["location"]
raw_df["has_pool"] = (
    raw_df["title"].str.contains("Kolam Renang", case=False, na=False).astype(int)
)

raw_df["listing_recency_hours"] = raw_df["updated"]


def extract_floors(title):
    if pd.isna(title):
        return 1

    title = title.lower()

    if "2 lantai" in title or "2 lt" in title:
        return 2
    elif "3 lantai" in title or "3 lt" in title:
        return 3
    elif "4 lantai" in title or "4 lt" in title:
        return 4
    return 1


raw_df["floors"] = raw_df["title"].apply(extract_floors)

final_columns = [
    "price_clean",
    "bedroom",
    "bathroom",
    "carport",
    "land_area",
    "building_area",
    "is_furnished",
    "near_shops",
    "near_school",
    "near_healthcare",
    "ready_to_live",
    "is_premier",
    "has_pool",
    "listing_recency_hours",
    "floors",
    "city",
]

raw_df_clean = raw_df[final_columns].copy()

numeric_columns = [
    "price_clean",
    "bedroom",
    "bathroom",
    "carport",
    "land_area",
    "building_area",
    "listing_recency_hours",
    "floors",
]
for col in numeric_columns:
    raw_df_clean[col] = (
        pd.to_numeric(raw_df_clean[col], errors="coerce").fillna(0).astype(int)
    )

binary_columns = [
    "is_furnished",
    "near_shops",
    "near_school",
    "near_healthcare",
    "ready_to_live",
    "is_premier",
    "has_pool",
]
for col in binary_columns:
    raw_df_clean[col] = raw_df_clean[col].fillna(0).astype(int)

raw_df_clean.dropna(subset=["price_clean"], inplace=True)

raw_df_clean["city"] = raw_df_clean["city"].astype(str)
raw_df_clean = pd.get_dummies(raw_df_clean, columns=["city"], drop_first=True)

print("Final cleaned dataset:")
print(raw_df_clean.head())

X = raw_df_clean.drop("price_clean", axis=1)
y = raw_df_clean["price_clean"]

print("\nFeatures (X):")
print(X.columns.tolist())
print("\nTarget (y): price_clean")
print("Number of samples:", len(raw_df_clean))

print("Missing values in original dataframe:")
print(raw_df.isnull().sum())

print("\nMissing values in features:")
print(X.isnull().sum())

print("\nPrice statistics (in IDR):")
print(y.describe())

output_file = os.path.join(current_dir, "output/cleaned/houses-cleaned.csv")
raw_df_clean.to_csv(output_file, index=False)

output_file_dataset = os.path.join(current_dir, "../dataset/houses-cleaned.csv")
raw_df_clean.to_csv(output_file_dataset, index=False)

print(f"Cleaned data saved to: {output_file} and {output_file_dataset}")

print("Analysis completed. Files saved:")
print("1. houses-cleaned.csv - Main cleaned data")
print("2. region_summary.csv - Summary metrics by region")
print("3. price_insights.csv - Price analysis by region")
print("4. feature_importance.csv - Feature correlation with price")
print("5. market_trends.csv - Price trends over time")
print("6. analysis_summary.csv - Overall analysis summary")
print("7. price_patterns.csv - Price patterns by feature categories")
print("8. accepted_*.csv - Accepted properties by region")
print("9. rejected_*.csv - Rejected properties by region")
