import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns


current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "../dataset/houses-cleaned.csv"))


X = df.drop("price_clean", axis=1)
y = df["price_clean"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
}


results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape})

    print(f"✅ Model: {name}")
    print(f"MAE : {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"MAPE: {mape:.2f}%")
    print("-" * 30)


results_df = pd.DataFrame(results)


best_model_mae = results_df.loc[results_df["MAE"].idxmin()]["Model"]
best_model_rmse = results_df.loc[results_df["RMSE"].idxmin()]["Model"]
best_model_mape = results_df.loc[results_df["MAPE (%)"].idxmin()]["Model"]

print(f"Best model based on MAE: {best_model_mae}")
print(f"Best model based on RMSE: {best_model_rmse}")
print(f"Best model based on MAPE: {best_model_mape}")


best_models = [best_model_mae, best_model_rmse, best_model_mape]
overall_best = max(set(best_models), key=best_models.count)
print(f"\nOverall best model: {overall_best}")


plt.figure(figsize=(12, 5))
for metric in ["MAE", "RMSE", "MAPE (%)"]:
    plt.figure(figsize=(8, 4))
    sns.barplot(x="Model", y=metric, data=results_df, palette="viridis")
    plt.title(f"Model Comparison - {metric}")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()
