import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from colorama import Fore, init
from tabulate import tabulate

init(autoreset=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dirs = [os.path.join(current_dir, "../model/evaluation")]
for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)


class HousePriceModel:
    def __init__(self, data_path=None, model_path=None):
        self.model = None
        self.feature_names = None
        self.data_path = data_path
        self.model_path = model_path or os.path.join(current_dir, "../model/model.joblib")
        self.feature_importance = None
        self.metrics = {}
        self.feature_encoder = None

    def load_data(self):
        print(Fore.GREEN + "Loading data...\n" + "=" * 50)
        df = pd.read_csv(self.data_path)

        X = df.drop("price_clean", axis=1)
        y = df["price_clean"]
        self.feature_names = X.columns.tolist()

        with open(os.path.join(current_dir, "../model/evaluation/feature_names.json"), "w") as f:
            json.dump(
                {
                    "features": self.feature_names,
                    "categorical_features": [
                        col for col in self.feature_names if col.startswith("city_")
                    ],
                },
                f,
            )

        return X, y

    def train(self, X, y, test_size=0.2, random_state=42):
        print(Fore.CYAN + "\n" + "=" * 50)
        print(Fore.CYAN + "STEP 1: Initial Model Training and Evaluation")
        print("=" * 50)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model = RandomForestRegressor(random_state=random_state)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        self.metrics["initial"] = {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

        print(Fore.YELLOW + "Initial Random Forest Performance:")
        print(
            tabulate(
                [
                    ["MAE", f"Rp {mae:,.0f}"],
                    ["RMSE", f"Rp {rmse:,.0f}"],
                    ["MAPE", f"{mape:.2f}%"],
                    ["R²", f"{r2:.4f}"],
                ],
                headers=["Metric", "Value"],
                tablefmt="fancy_grid",
            )
        )

        return X_train, X_test, y_train, y_test

    def cross_validate(self, X, y, cv=5):
        print(Fore.CYAN + "\n" + "=" * 50)
        print(Fore.CYAN + "STEP 2: Cross-Validation")
        print("=" * 50)

        cv_scores = cross_val_score(
            self.model, X, y, cv=cv, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores)

        print(Fore.YELLOW + "Cross-Validation Results (RMSE):")
        print(
            tabulate(
                [
                    ["Fold {}".format(i + 1), f"Rp {score:,.0f}"]
                    for i, score in enumerate(cv_rmse)
                ],
                headers=["Fold", "RMSE"],
                tablefmt="fancy_grid",
            )
        )
        print(f"Mean RMSE: Rp {cv_rmse.mean():,.0f}")
        print(f"RMSE Std: Rp {cv_rmse.std():,.0f}")

        self.metrics["cross_validation"] = {
            "folds": cv_rmse.tolist(),
            "mean": cv_rmse.mean(),
            "std": cv_rmse.std(),
        }

    def analyze_feature_importance(self, X):
        print(Fore.CYAN + "\n" + "=" * 50)
        print(Fore.CYAN + "STEP 3: Feature Importance Analysis")
        print("=" * 50)

        self.feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": self.model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print(Fore.YELLOW + "Top 10 most important features:")
        print(self.feature_importance.head(10))

        self.feature_importance.to_csv(
            os.path.join(current_dir, "../model/evaluation/feature_importance.csv"), index=False
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=self.feature_importance.head(15))
        plt.title("Feature Importance - Top 15 Features")
        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, "../model/evaluation/feature_importance.png"))
        plt.close()

    def tune_hyperparameters(self, X_train, y_train, X_test, y_test):
        print(Fore.CYAN + "\n" + "=" * 50)
        print(Fore.CYAN + "STEP 4: Hyperparameter Tuning")
        print("=" * 50)

        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        print(Fore.YELLOW + "Starting grid search (this may take some time)...")

        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        print(Fore.GREEN + f"\nBest hyperparameters: {best_params}")

        with open(os.path.join(current_dir, "../model/evaluation/best_params.json"), "w") as f:
            json.dump(best_params, f)

        y_pred_tuned = self.model.predict(X_test)
        mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
        mse_tuned = mean_squared_error(y_test, y_pred_tuned)
        rmse_tuned = np.sqrt(mse_tuned)
        r2_tuned = r2_score(y_test, y_pred_tuned)
        mape_tuned = np.mean(np.abs((y_test - y_pred_tuned) / y_test)) * 100

        self.metrics["tuned"] = {
            "mae": mae_tuned,
            "rmse": rmse_tuned,
            "mape": mape_tuned,
            "r2": r2_tuned,
            "best_params": best_params,
        }

        return X_test, y_test, y_pred_tuned

    def evaluate_final_model(self, y_test, y_pred_tuned):
        print(Fore.CYAN + "\n" + "=" * 50)
        print(Fore.CYAN + "STEP 5: Final Model Evaluation")
        print("=" * 50)

        initial = self.metrics["initial"]
        tuned = self.metrics["tuned"]

        print(Fore.YELLOW + "Comparison of Initial vs. Tuned Model:")
        print(
            tabulate(
                [
                    [
                        "MAE",
                        f"Rp {initial['mae']:,.0f}",
                        f"Rp {tuned['mae']:,.0f}",
                        f"{(1 - tuned['mae']/initial['mae'])*100:.2f}%",
                    ],
                    [
                        "RMSE",
                        f"Rp {initial['rmse']:,.0f}",
                        f"Rp {tuned['rmse']:,.0f}",
                        f"{(1 - tuned['rmse']/initial['rmse'])*100:.2f}%",
                    ],
                    [
                        "MAPE",
                        f"{initial['mape']:.2f}%",
                        f"{tuned['mape']:.2f}%",
                        f"{(1 - tuned['mape']/initial['mape'])*100:.2f}%",
                    ],
                    [
                        "R²",
                        f"{initial['r2']:.4f}",
                        f"{tuned['r2']:.4f}",
                        f"{(tuned['r2']/initial['r2'] - 1)*100:.2f}%",
                    ],
                ],
                headers=["Metric", "Initial Model", "Tuned Model", "Improvement"],
                tablefmt="fancy_grid",
            )
        )

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_tuned, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs. Predicted Housing Prices")
        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, "../model/evaluation/actual_vs_predicted.png"))
        plt.close()

        residuals = y_test - y_pred_tuned
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred_tuned, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Price")
        plt.ylabel("Residuals")
        plt.title("Residual Analysis")
        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, "../model/evaluation/residuals.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Residual Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, "../model/evaluation/residual_distribution.png"))
        plt.close()

    def save_model(self):
        print(Fore.CYAN + "\n" + "=" * 50)
        print(Fore.CYAN + "STEP 6: Save the Final Model")
        print("=" * 50)

        joblib.dump(self.model, self.model_path)
        print(Fore.GREEN + f"Final model saved to: {self.model_path}")

        prediction_info = {
            "features": self.feature_names,
            "model_path": self.model_path,
        }

        with open(os.path.join(current_dir, "../model/evaluation/prediction_info.json"), "w") as f:
            json.dump(prediction_info, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(Fore.GREEN + f"Model loaded from: {self.model_path}")

            if os.path.exists(os.path.join(current_dir, "../model/evaluation/feature_names.json")):
                with open(
                    os.path.join(current_dir, "../model/evaluation/feature_names.json"), "r"
                ) as f:
                    feature_info = json.load(f)
                    self.feature_names = feature_info["features"]

            return True
        else:
            print(Fore.RED + f"No model found at: {self.model_path}")
            return False

    def predict(self, features_dict):

        if self.model is None:
            if not self.load_model():
                raise ValueError("No model available for prediction")

        if not self.feature_names:
            with open(os.path.join(current_dir, "../model/evaluation/feature_names.json"), "r") as f:
                feature_info = json.load(f)
                self.feature_names = feature_info["features"]

        input_data = pd.DataFrame(columns=self.feature_names)
        input_data.loc[0] = 0

        for feature, value in features_dict.items():
            if feature in self.feature_names:
                input_data.loc[0, feature] = value
            elif feature == "city" and "city_" + value in self.feature_names:
                city_col = "city_" + value
                input_data.loc[0, city_col] = 1

        prediction = self.model.predict(input_data)[0]

        return prediction

    def run_full_pipeline(self):
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = self.train(X, y)
        self.cross_validate(X, y)
        self.analyze_feature_importance(X)
        X_test, y_test, y_pred_tuned = self.tune_hyperparameters(
            X_train, y_train, X_test, y_test
        )
        self.evaluate_final_model(y_test, y_pred_tuned)
        self.save_model()


if __name__ == "__main__":
    data_path = os.path.join(current_dir, "../dataset/houses-cleaned.csv")
    model = HousePriceModel(data_path=data_path)
    model.run_full_pipeline()
