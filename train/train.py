import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from colorama import Fore, Style, init
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import datetime
import logging


init()


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from utils.preprocessing import (
    DataValidationError,
    FeatureEngineeringError,
    PreprocessingError,
    DataScalingError,
)
from rf_model import train_base_rf_model, evaluate_rf_model, train_optimized_rf_model
from ga_optimizer import GAOptimizer


output_dirs = [
    os.path.join(parent_dir, "model"),
    os.path.join(parent_dir, "model/evaluation"),
    os.path.join(parent_dir, "logs"),
    os.path.join(parent_dir, "reports"),
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, "logs/training.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("training")


def plot_rf_results(metrics_dict: Dict[str, Dict[str, float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Random Forest Model Performance Comparison", fontsize=16)

    models = list(metrics_dict.keys())
    mapes = [metrics_dict[model]["mape"] * 100 for model in models]
    axes[0, 0].bar(models, mapes, color=["#3498db", "#2ecc71"])
    axes[0, 0].set_ylabel("MAPE (%)")
    axes[0, 0].set_title("Mean Absolute Percentage Error (lower is better)")
    for i, v in enumerate(mapes):
        axes[0, 0].text(i, v + 0.5, f"{v:.2f}%", ha="center")

    r2_scores = [metrics_dict[model]["r2"] for model in models]
    axes[0, 1].bar(models, r2_scores, color=["#3498db", "#2ecc71"])
    axes[0, 1].set_ylabel("R²")
    axes[0, 1].set_title("R² Score (higher is better)")
    for i, v in enumerate(r2_scores):
        axes[0, 1].text(i, v + 0.01, f"{v:.4f}", ha="center")

    rmse_scores = [metrics_dict[model]["rmse"] / 1e6 for model in models]
    axes[1, 0].bar(models, rmse_scores, color=["#3498db", "#2ecc71"])
    axes[1, 0].set_ylabel("RMSE (Millions Rp)")
    axes[1, 0].set_title("Root Mean Squared Error (lower is better)")
    for i, v in enumerate(rmse_scores):
        axes[1, 0].text(i, v + 0.05, f"{v:.2f}M", ha="center")

    mae_scores = [metrics_dict[model]["mae"] / 1e6 for model in models]
    axes[1, 1].bar(models, mae_scores, color=["#3498db", "#2ecc71"])
    axes[1, 1].set_ylabel("MAE (Millions Rp)")
    axes[1, 1].set_title("Mean Absolute Error (lower is better)")
    for i, v in enumerate(mae_scores):
        axes[1, 1].text(i, v + 0.05, f"{v:.2f}M", ha="center")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    output_path = os.path.join(parent_dir, "model/evaluation/rf_model_comparison.png")
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved performance comparison chart to {output_path}")
    plt.close()


def plot_feature_importance(
    model: RandomForestRegressor,
    feature_names: List[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Generate and save feature importance plots.

    Args:
        model: Trained RandomForestRegressor model
        feature_names: List of feature names
        X_test: Test feature data
        y_test: Test target data

    Returns:
        None
    """

    fig, axes = plt.subplots(2, 1, figsize=(12, 16))

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sns.barplot(
        x=importances[indices][:10],
        y=[feature_names[i] for i in indices][:10],
        ax=axes[0],
    )
    axes[0].set_title("Feature Importance (MDI)")
    axes[0].set_xlabel("Mean Decrease in Impurity")

    try:
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42
        )
        perm_indices = np.argsort(perm_importance.importances_mean)[::-1]

        sns.barplot(
            x=perm_importance.importances_mean[perm_indices][:10],
            y=[feature_names[i] for i in perm_indices][:10],
            ax=axes[1],
        )
        axes[1].set_title("Feature Importance (Permutation)")
        axes[1].set_xlabel("Mean Decrease in Accuracy")
    except Exception as e:
        logger.warning(f"Could not calculate permutation importance: {str(e)}")
        axes[1].text(
            0.5,
            0.5,
            "Permutation importance calculation failed",
            ha="center",
            va="center",
            fontsize=12,
        )

    plt.tight_layout()

    output_path = os.path.join(parent_dir, "model/evaluation/feature_importance.png")
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved feature importance chart to {output_path}")
    plt.close()


def save_models(
    model: RandomForestRegressor,
    metrics: Dict[str, Any],
    feature_names: List[str],
    model_name: str = "optimized_rf_model",
    output_dir: str = "../model",
) -> None:
    model_path = os.path.join(current_dir, output_dir)
    os.makedirs(model_path, exist_ok=True)

    model_filename = f"{model_name}.pkl"
    with open(os.path.join(model_path, model_filename), "wb") as f:
        pickle.dump(model, f)

    params = {
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
        "max_features": model.max_features,
        "bootstrap": model.bootstrap,
        "random_state": model.random_state,
        "n_jobs": model.n_jobs,
    }

    feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))

    metadata = {
        "model_parameters": params,
        "features": feature_names,
        "feature_importance": feature_importance,
        "evaluation_metrics": metrics,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model.__class__.__name__,
    }

    with open(os.path.join(model_path, f"{model_name}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    with open(os.path.join(model_path, f"{model_name}_params.txt"), "w") as f:
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(
            f"Training Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write("\nHyperparameters:\n")
        for param, value in params.items():
            f.write(f"  {param}: {value}\n")

        f.write("\nFeatures Used:\n")
        for i, feature in enumerate(feature_names):
            importance = model.feature_importances_[i]
            f.write(f"  {feature}: {importance:.4f}\n")

        f.write("\nPerformance Metrics:\n")
        for metric, value in metrics.items():
            if metric == "mape":
                f.write(f"  MAPE: {value * 100:.2f}%\n")
            elif metric == "rmse" or metric == "mae":
                f.write(f"  {metric.upper()}: {value:,.2f}\n")
            else:
                f.write(f"  {metric}: {value}\n")

    logger.info(f"Saved model {model_name} to {model_path}")
    print(f"{Fore.GREEN}Model saved to {output_dir} as {model_name}{Style.RESET_ALL}")

    return os.path.join(model_path, model_filename)


def clean_numeric(val):
    try:
        if pd.isna(val):
            return None
        if not isinstance(val, str):
            return float(val)

        val = str(val)
        val = (
            val.replace("Rp", "")
            .replace("Miliar", "000000000")
            .replace("Juta", "000000")
        )
        val = val.replace("²", "").replace(" ", "")
        val = val.split("HEMAT")[0]
        val = val.split("M")[0]

        val = "".join(c for c in val if c.isdigit() or c == ".")
        if val:
            try:
                return float(val)
            except ValueError:
                return None
        return None
    except Exception as e:
        print(f"Error cleaning value '{val}': {str(e)}")
        return None


def main() -> None:
    logger.info("Starting House Price Prediction model training")
    print(
        f"{Fore.CYAN}Starting House Price Prediction model training...{Style.RESET_ALL}"
    )

    try:
        processed_data_dir = os.path.join(parent_dir, "processed_data")

        if os.path.exists(processed_data_dir):
            data_files = [
                f
                for f in os.listdir(processed_data_dir)
                if f.endswith(".pkl") and "preprocessed" in f
            ]

            if not data_files:
                raise FileNotFoundError("No preprocessed data files found")

            data_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(processed_data_dir, f)),
                reverse=True,
            )

            preprocessed_file = os.path.join(processed_data_dir, data_files[0])
            logger.info(
                f"Using most recent preprocessed data file: {preprocessed_file}"
            )
        else:
            raise FileNotFoundError("Processed data directory not found")

    except FileNotFoundError as e:
        logger.warning(f"Preprocessed data not found: {str(e)}")
        print(f"{Fore.YELLOW}Warning: {str(e)}")
        print(
            f"{Fore.YELLOW}Please make sure preprocessing has been run first.{Style.RESET_ALL}"
        )
        preprocessed_file = input(
            f"{Fore.CYAN}Enter the path to preprocessed data file (.pkl): {Style.RESET_ALL}"
        )

    try:
        print(f"{Fore.CYAN}Loading preprocessed data...{Style.RESET_ALL}")
        with open(preprocessed_file, "rb") as f:
            preprocessed_data = pickle.load(f)

        if isinstance(preprocessed_data, dict) and "data" in preprocessed_data:
            df = preprocessed_data["data"]
            feature_columns = preprocessed_data.get("feature_columns", [])
            preprocessing_info = preprocessed_data.get("preprocessing_info", {})
        else:

            df = preprocessed_data
            feature_columns = [col for col in df.columns if col != "price"]
            preprocessing_info = {}

        logger.info(
            f"Loaded preprocessed data with {len(df)} rows and {df.shape[1]} columns"
        )
        print(
            f"{Fore.GREEN}Successfully loaded data with {len(df):,} properties{Style.RESET_ALL}"
        )

        if len(preprocessing_info) > 0:
            print(f"{Fore.CYAN}Preprocessing summary:{Style.RESET_ALL}")
            for key, value in preprocessing_info.items():
                if isinstance(value, (int, float, str, bool)):
                    print(f"  - {key}: {value}")

    except Exception as e:
        logger.error(f"Error loading preprocessed data: {str(e)}")
        raise RuntimeError(f"Failed to load preprocessed data: {str(e)}")

    print(f"{Fore.CYAN}Preparing data for model training...{Style.RESET_ALL}")

    target_column = "price"

    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in data")
        raise ValueError(
            f"Target column '{target_column}' not found in preprocessed data"
        )

    if not feature_columns:
        feature_columns = [col for col in df.columns if col != target_column]

    X = df[feature_columns]
    y = df[target_column]

    logger.info(f"Using {len(feature_columns)} features for model training")
    print(
        f"{Fore.CYAN}Using {len(feature_columns)} features for model training{Style.RESET_ALL}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(
        f"{Fore.CYAN}Training set: {len(X_train):,} rows | Test set: {len(X_test):,} rows{Style.RESET_ALL}"
    )

    print(f"\n{Fore.CYAN}Training base Random Forest model...{Style.RESET_ALL}")
    base_rf_model, base_cv_mape = train_base_rf_model(X_train, y_train)

    base_test_mape, base_predictions, base_test_metrics = evaluate_rf_model(
        base_rf_model, X_test, y_test
    )

    print(f"\n{Fore.YELLOW}Base Random Forest Metrics:{Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}Cross-Validation MAPE: {Fore.WHITE}{base_cv_mape * 100:.2f}%{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}Test MAPE: {Fore.WHITE}{base_test_mape * 100:.2f}%{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}Test R²: {Fore.WHITE}{base_test_metrics['r2']:.4f}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}Test RMSE: {Fore.WHITE}{base_test_metrics['rmse']:.2f}{Style.RESET_ALL}"
    )

    print(
        f"\n{Fore.CYAN}Optimizing Random Forest with Genetic Algorithm...{Style.RESET_ALL}"
    )

    try:
        ga_optimizer = GAOptimizer()
        
        # Define the hyperparameter bounds for Random Forest
        # Format: [(min_value, max_value), ...] for each parameter
        param_bounds = [
            (50, 200),        # n_estimators
            (5, 30),          # max_depth
            (2, 20),          # min_samples_split
            (1, 10),          # min_samples_leaf
            (0.1, 1.0)        # max_features (proportion)
        ]
        
        # Run the genetic algorithm
        best_individual, best_performers = ga_optimizer.genetic_algorithm(
            population_size=20,
            object_bounds=param_bounds,
            generations=10,
            mutation_rate=0.2,
            data=X_train,
            price_target=y_train,
            cv_folds=5,
            early_stopping_generations=3
        )
        
        # Convert the best individual to hyperparameter dictionary
        best_params = {
            'n_estimators': int(round(best_individual[0])),
            'max_depth': int(round(best_individual[1])),
            'min_samples_split': int(round(best_individual[2])),
            'min_samples_leaf': int(round(best_individual[3])),
            'max_features': float(best_individual[4])
        }
    except Exception as e:
        logger.error(f"Error during GA optimization: {str(e)}")
        print(f"{Fore.YELLOW}Error during GA optimization: {str(e)}")
        print(f"{Fore.YELLOW}Using default parameters instead.{Style.RESET_ALL}")
        best_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt'
        }

    print(f"\n{Fore.CYAN}Best hyperparameters found:{Style.RESET_ALL}")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")

    print(f"\n{Fore.CYAN}Training optimized Random Forest model...{Style.RESET_ALL}")
    optimized_rf_model, opt_cv_mape = train_optimized_rf_model(
        X_train, y_train, best_params
    )

    opt_test_mape, opt_predictions, opt_test_metrics = evaluate_rf_model(
        optimized_rf_model, X_test, y_test
    )

    print(f"\n{Fore.YELLOW}Optimized Random Forest Metrics:{Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}Cross-Validation MAPE: {Fore.WHITE}{opt_cv_mape * 100:.2f}%{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}Test MAPE: {Fore.WHITE}{opt_test_mape * 100:.2f}%{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}Test R²: {Fore.WHITE}{opt_test_metrics['r2']:.4f}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}Test RMSE: {Fore.WHITE}{opt_test_metrics['rmse']:.2f}{Style.RESET_ALL}"
    )

    improvement_percentage = ((base_test_mape - opt_test_mape) / base_test_mape) * 100
    print(
        f"{Fore.YELLOW}Random Forest improvement on test data: {Fore.WHITE}{improvement_percentage:.2f}%{Style.RESET_ALL}"
    )

    r2_improvement = (
        (
            (opt_test_metrics["r2"] - base_test_metrics["r2"])
            / abs(base_test_metrics["r2"])
        )
        * 100
        if base_test_metrics["r2"] != 0
        else float("inf")
    )
    print(
        f"{Fore.YELLOW}R² improvement: {Fore.WHITE}{r2_improvement:.2f}%{Style.RESET_ALL}"
    )

    optimized_metrics = {
        "mape": opt_test_mape,
        "r2": opt_test_metrics["r2"],
        "rmse": opt_test_metrics["rmse"],
        "mae": opt_test_metrics["mae"],
    }
    save_models(optimized_rf_model, optimized_metrics, X_test.columns.tolist())

    plot_feature_importance(optimized_rf_model, X_test.columns.tolist(), X_test, y_test)

    metrics_dict = {
        "Base RF": {
            "mape": base_test_mape,
            "r2": base_test_metrics["r2"],
            "rmse": base_test_metrics["rmse"],
            "mae": base_test_metrics["mae"],
        },
        "Optimized RF": {
            "mape": opt_test_mape,
            "r2": opt_test_metrics["r2"],
            "rmse": opt_test_metrics["rmse"],
            "mae": opt_test_metrics["mae"],
        },
    }

    plot_rf_results(metrics_dict)

    logger.info("Model training and evaluation completed successfully")
    print(
        f"\n{Fore.GREEN}✓ Model training and evaluation completed successfully!{Style.RESET_ALL}"
    )
    print(
        f"{Fore.GREEN}✓ Model saved to {os.path.join(parent_dir, 'model')}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.GREEN}✓ Performance plots generated in {os.path.join(parent_dir, 'model/evaluation')}{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    try:
        logger.info("Starting House Price Prediction System")
        main()
        logger.info("House Price Prediction System completed successfully")
    except DataValidationError as e:
        logger.error(f"Data validation error: {str(e)}")
        print(f"{Fore.RED}Error validating data: {str(e)}{Style.RESET_ALL}")
    except FeatureEngineeringError as e:
        logger.error(f"Feature engineering error: {str(e)}")
        print(f"{Fore.RED}Error in feature engineering: {str(e)}{Style.RESET_ALL}")
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {str(e)}")
        print(f"{Fore.RED}Error during preprocessing: {str(e)}{Style.RESET_ALL}")
    except DataScalingError as e:
        logger.error(f"Data scaling error: {str(e)}")
        print(f"{Fore.RED}Error during data scaling: {str(e)}{Style.RESET_ALL}")
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        print(f"{Fore.RED}An unexpected error occurred: {str(e)}{Style.RESET_ALL}")
