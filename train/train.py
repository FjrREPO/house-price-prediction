import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style, init

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


def plot_rf_results(base_mape, optimized_mape):
    models = ["Base RF", "Optimized RF"]
    mapes = [base_mape, optimized_mape]

    plt.figure(figsize=(10, 6))
    plt.bar(models, [m * 100 for m in mapes])
    plt.ylabel("MAPE (%)")
    plt.title("Random Forest Model Performance Comparison")

    for i, v in enumerate(mapes):
        plt.text(i, v * 100 + 0.5, f"{v * 100:.2f}%", ha="center")

    plt.tight_layout()
    plt.savefig(
        os.path.join(current_dir, "../model/evaluation/rf_model_comparison.png")
    )
    plt.close()


def save_models(optimized_rf_model, output_dir="../model"):
    model_path = os.path.join(current_dir, output_dir)
    os.makedirs(model_path, exist_ok=True)

    # Save the RF model
    with open(os.path.join(model_path, "optimized_rf_model.pkl"), "wb") as f:
        pickle.dump(optimized_rf_model, f)

    print(f"{Fore.GREEN}Optimized RF model saved to {output_dir}{Style.RESET_ALL}")


def main():
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

    print(
        f"\n{Back.GREEN}{Fore.BLACK} Training and Evaluating Base Random Forest Model {Style.RESET_ALL}"
    )
    base_rf_model = train_base_rf_model(X, y)
    base_mape, base_predictions = evaluate_rf_model(base_rf_model, X, y)
    print(
        f"{Fore.GREEN}Base Random Forest MAPE: {Fore.WHITE}{base_mape * 100:.2f}%{Style.RESET_ALL}"
    )

    print(
        f"\n{Back.YELLOW}{Fore.BLACK} Optimizing Random Forest with Genetic Algorithm {Style.RESET_ALL}"
    )
    ga = GAOptimizer()

    population_size = 80
    generations = 5
    mutation_rate = 0.35
    object_bounds = [
        (50, 200),  # n_estimators
        (5, 30),  # max_depth
        (2, 10),  # min_samples_split
        (1, 5),  # min_samples_leaf
    ]

    best_solution, best_performers = ga.genetic_algorithm(
        population_size, object_bounds, generations, mutation_rate, X, price_target=y
    )

    n_estimators = int(best_solution[0])
    max_depth = int(best_solution[1])
    min_samples_split = int(best_solution[2])
    min_samples_leaf = int(best_solution[3])

    print(f"\n{Fore.YELLOW}Best Random Forest Parameters:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}n_estimators: {n_estimators}")
    print(f"max_depth: {max_depth}")
    print(f"min_samples_split: {min_samples_split}")
    print(f"min_samples_leaf: {min_samples_leaf}{Style.RESET_ALL}")

    optimized_rf_model = train_optimized_rf_model(X, y, best_solution)
    optimized_rf_mape, optimized_predictions = evaluate_rf_model(
        optimized_rf_model, X, y
    )
    print(
        f"{Fore.YELLOW}Optimized Random Forest MAPE: {Fore.WHITE}{optimized_rf_mape * 100:.2f}%{Style.RESET_ALL}"
    )

    improvement_percentage = ((base_mape - optimized_rf_mape) / base_mape) * 100
    print(
        f"{Fore.YELLOW}Random Forest improvement: {Fore.WHITE}{improvement_percentage:.2f}%{Style.RESET_ALL}"
    )

    save_models(optimized_rf_model)

    plot_rf_results(base_mape, optimized_rf_mape)
    print(
        f"{Fore.GREEN}Random Forest model results plotted and saved.{Style.RESET_ALL}"
    )

    print(
        f"{Fore.GREEN}House Price Prediction System completed successfully!{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()
