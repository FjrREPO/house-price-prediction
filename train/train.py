import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
from colorama import Fore, Back, Style, init
from sklearn.model_selection import train_test_split

init(autoreset=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

from utils.helper import preprocess_data, remove_outliers
from rf_model import (
    train_base_rf_model,
    train_optimized_rf_model,
    evaluate_rf_model,
)
from ga_optimizer import GAOptimizer

output_dirs = [
    os.path.join(current_dir, "../model"),
    os.path.join(current_dir, "../model/evaluation"),
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

def plot_rf_results(base_mape: float, optimized_mape: float) -> None:
    """
    Generate and save a bar chart comparing base and optimized Random Forest model performance.
    
    Args:
        base_mape: Mean Absolute Percentage Error of the base Random Forest model on test data
        optimized_mape: Mean Absolute Percentage Error of the optimized Random Forest model on test data
    
    Returns:
        None
    """
    models = ["Base RF", "Optimized RF"]
    mapes = [base_mape, optimized_mape]

    plt.figure(figsize=(10, 6))
    plt.bar(models, [m * 100 for m in mapes], color=['#3498db', '#2ecc71'])
    plt.ylabel("MAPE (%)")
    plt.title("Random Forest Model Performance Comparison (Test Data)")

    for i, v in enumerate(mapes):
        plt.text(i, v * 100 + 0.5, f"{v * 100:.2f}%", ha="center")

    plt.tight_layout()
    plt.savefig(
        os.path.join(current_dir, "../model/evaluation/rf_model_comparison.png")
    )
    plt.close()


def save_models(optimized_rf_model, output_dir: str = "../model") -> None:
    """
    Save the optimized Random Forest model to disk.
    
    Args:
        optimized_rf_model: The trained and optimized Random Forest model
        output_dir: Directory path where the model will be saved
        
    Returns:
        None
    """
    model_path = os.path.join(current_dir, output_dir)
    os.makedirs(model_path, exist_ok=True)

    # Save the RF model
    with open(os.path.join(model_path, "optimized_rf_model.pkl"), "wb") as f:
        pickle.dump(optimized_rf_model, f)
    
    # Save model parameters as text for reference
    params = {
        'n_estimators': optimized_rf_model.n_estimators,
        'max_depth': optimized_rf_model.max_depth,
        'min_samples_split': optimized_rf_model.min_samples_split,
        'min_samples_leaf': optimized_rf_model.min_samples_leaf,
        'max_features': optimized_rf_model.max_features,
        'bootstrap': optimized_rf_model.bootstrap
    }
    
    with open(os.path.join(model_path, "optimized_rf_params.txt"), "w") as f:
        for param, value in params.items():
            f.write(f"{param}: {value}\n")

    print(f"{Fore.GREEN}Optimized RF model saved to {output_dir}{Style.RESET_ALL}")


def main() -> None:
    """
    Main function to execute the house price prediction workflow:
    1. Load and preprocess data
    2. Train a base Random Forest model
    3. Optimize the model with Genetic Algorithm
    4. Evaluate and save the optimized model
    
    Returns:
        None
    """
    print(f"{Back.BLUE}{Fore.WHITE} House Price Prediction System {Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}Loading and preprocessing data...{Style.RESET_ALL}")
    df = pd.read_csv(os.path.join(current_dir, "../dataset/houses-cleaned.csv"))
    df_preprocessed = preprocess_data(df)

    df_yogyakarta = df_preprocessed[df_preprocessed["kabupaten_kota"] == "Yogyakarta"]
    print(
        f"{Fore.YELLOW}Number of houses in Yogyakarta: {len(df_yogyakarta)}{Style.RESET_ALL}"
    )

    columns_to_check = ["price", "bedroom", "bathroom", "LT", "LB"]
    df_yogyakarta_cleaned = remove_outliers(df_yogyakarta, columns_to_check)

    X = df_yogyakarta_cleaned[["bedroom", "bathroom", "LT", "LB"]]
    y = df_yogyakarta_cleaned["price"]
    
    # Split data into train and test sets for proper evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"{Fore.CYAN}Data split: {len(X_train)} training samples, {len(X_test)} test samples{Style.RESET_ALL}")

    print(
        f"\n{Back.GREEN}{Fore.BLACK} Training and Evaluating Base Random Forest Model {Style.RESET_ALL}"
    )
    base_rf_model = train_base_rf_model(X_train, y_train)
    
    # Evaluate with cross-validation on training data
    base_cv_mape, _, base_cv_metrics = evaluate_rf_model(base_rf_model, X_train, y_train, cross_validate=True, cv=5)
    
    # Evaluate on test data
    base_test_mape, base_predictions, base_test_metrics = evaluate_rf_model(base_rf_model, X_test, y_test)
    
    print(f"{Fore.GREEN}Base Random Forest Metrics:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Cross-Validation MAPE: {Fore.WHITE}{base_cv_mape * 100:.2f}%{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Test MAPE: {Fore.WHITE}{base_test_mape * 100:.2f}%{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Test R²: {Fore.WHITE}{base_test_metrics['r2']:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Test RMSE: {Fore.WHITE}{base_test_metrics['rmse']:.2f}{Style.RESET_ALL}")

    print(
        f"\n{Back.YELLOW}{Fore.BLACK} Optimizing Random Forest with Genetic Algorithm {Style.RESET_ALL}"
    )
    ga = GAOptimizer()

    # Enhanced GA parameters
    population_size = 100  # Larger population for better exploration
    generations = 10       # More generations for better convergence
    mutation_rate = 0.35   # Initial mutation rate (will adapt during evolution)
    object_bounds = [
        (50, 300),      # n_estimators
        (5, 50),        # max_depth
        (2, 20),        # min_samples_split
        (1, 10),        # min_samples_leaf
        (0.0, 1.0),     # max_features (will be mapped to 'sqrt', 'log2', or None)
        (0.0, 1.0),     # bootstrap (boolean, will be mapped based on threshold)
    ]
    
    # Log GA parameters
    print(f"{Fore.YELLOW}GA Parameters:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Population Size: {population_size}")
    print(f"{Fore.YELLOW}Generations: {generations}")
    print(f"{Fore.YELLOW}Initial Mutation Rate: {mutation_rate}{Style.RESET_ALL}")

    best_solution, best_performers = ga.genetic_algorithm(
        population_size, object_bounds, generations, mutation_rate, X_train, price_target=y_train
    )

    n_estimators = int(best_solution[0])
    max_depth = int(best_solution[1])
    min_samples_split = int(best_solution[2])
    min_samples_leaf = int(best_solution[3])
    # Extract all parameters from best solution
    max_features_param = best_solution[4] if len(best_solution) > 4 else 'sqrt'
    bootstrap_param = best_solution[5] > 0.5 if len(best_solution) > 5 else True
    
    # Map max_features float to string for display
    max_features_display = max_features_param
    if isinstance(max_features_param, float):
        if max_features_param < 0.33:
            max_features_display = 'sqrt'
        elif max_features_param < 0.66:
            max_features_display = 'log2'
        else:
            max_features_display = 'None'

    print(f"\n{Fore.YELLOW}Best Random Forest Parameters:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}n_estimators: {n_estimators}")
    print(f"{Fore.YELLOW}max_depth: {max_depth}")
    print(f"{Fore.YELLOW}min_samples_split: {min_samples_split}")
    print(f"{Fore.YELLOW}min_samples_leaf: {min_samples_leaf}")
    print(f"{Fore.YELLOW}max_features: {max_features_display}")
    print(f"{Fore.YELLOW}bootstrap: {bootstrap_param}{Style.RESET_ALL}")

    # Train model on all training data with best parameters
    optimized_rf_model = train_optimized_rf_model(X_train, y_train, best_solution)
    
    # Evaluate with cross-validation
    opt_cv_mape, _, opt_cv_metrics = evaluate_rf_model(
        optimized_rf_model, X_train, y_train, cross_validate=True, cv=5
    )
    
    # Evaluate on test data
    opt_test_mape, opt_predictions, opt_test_metrics = evaluate_rf_model(
        optimized_rf_model, X_test, y_test
    )
    
    print(f"\n{Fore.YELLOW}Optimized Random Forest Metrics:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Cross-Validation MAPE: {Fore.WHITE}{opt_cv_mape * 100:.2f}%{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Test MAPE: {Fore.WHITE}{opt_test_mape * 100:.2f}%{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Test R²: {Fore.WHITE}{opt_test_metrics['r2']:.4f}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Test RMSE: {Fore.WHITE}{opt_test_metrics['rmse']:.2f}{Style.RESET_ALL}")

    # Calculate improvement using test metrics
    improvement_percentage = ((base_test_mape - opt_test_mape) / base_test_mape) * 100
    print(
        f"{Fore.YELLOW}Random Forest improvement on test data: {Fore.WHITE}{improvement_percentage:.2f}%{Style.RESET_ALL}"
    )
    
    # Calculate improvement on R² (higher is better, so formula is reversed)
    r2_improvement = ((opt_test_metrics['r2'] - base_test_metrics['r2']) / abs(base_test_metrics['r2'])) * 100 if base_test_metrics['r2'] != 0 else float('inf')
    print(
        f"{Fore.YELLOW}R² improvement: {Fore.WHITE}{r2_improvement:.2f}%{Style.RESET_ALL}"
    )

    # Save the optimized model
    save_models(optimized_rf_model)
    
    # Create and save performance comparison plot using test metrics
    plot_rf_results(base_test_mape, opt_test_mape)
    print(
        f"{Fore.GREEN}Random Forest model results plotted and saved.{Style.RESET_ALL}"
    )

    print(
        f"{Fore.GREEN}House Price Prediction System completed successfully!{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()
