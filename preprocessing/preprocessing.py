import os
import sys
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import mean_absolute_percentage_error

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

from utils.helper import preprocess_data, remove_outliers

output_dirs = [
    os.path.join(current_dir, "../dataset"),
    os.path.join(current_dir, "../dataset/evaluation"),
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)


df = pd.read_csv(os.path.join(current_dir, "../dataset/houses.csv"))


df.head(10)

df_preprocessed = preprocess_data(df)
df_preprocessed.describe()

df_groupby = df_preprocessed.groupby("kabupaten_kota")

raw_df = pd.DataFrame()
total_count = 0
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

    threshold_mape = 0.15

    predictions = []
    for i in range(len(df_yogyakarta_cleaned)):
        prediction = predict_price_fuzzy(
            df_yogyakarta_cleaned.iloc[i]["bedroom"],
            df_yogyakarta_cleaned.iloc[i]["bathroom"],
            df_yogyakarta_cleaned.iloc[i]["LT"],
            df_yogyakarta_cleaned.iloc[i]["LB"],
            fallback=df_yogyakarta_cleaned.iloc[i]["price"],
        )

        expected = df_yogyakarta_cleaned.iloc[i]["price"]

        if abs(prediction - expected) / expected <= threshold_mape:

            raw_data.append(df.iloc[i])
            print(f"Predicted: {prediction:.2f}, Expected: {expected:.2f}")
            total_count += 1
            print(total_count)

        predictions.append(prediction)

    raw_df = pd.concat([raw_df, pd.DataFrame(raw_data)], ignore_index=True)

    mape = mean_absolute_percentage_error(df_yogyakarta_cleaned["price"], predictions)

    print("quartiels")
    print(price_quartiles)
    print(bedroom_quartiles)
    print(bathroom_quartiles)
    print(LT_quartiles)
    print(LB_quartiles)
    print(
        f"Mean Absolute Percentage Error (MAPE) of the fuzzy model: {mape * 100:.2f}%"
    )

# drop row with NaN values
raw_df = raw_df.dropna()

raw_df.to_csv(os.path.join(current_dir, "../dataset/houses-cleaned.csv"), index=False)
