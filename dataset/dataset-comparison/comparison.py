import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style, init
from typing import Dict, Tuple, Any
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))

init(autoreset=True)


class DatasetComparator:
    def __init__(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name1: str = "Dataset 1",
        name2: str = "Dataset 2",
    ):
        self.df1 = df1.copy()
        self.df2 = df2.copy()
        self.name1 = name1
        self.name2 = name2

        if set(df1.columns) != set(df2.columns):
            print(
                f"{Fore.RED}Warning: Datasets have different columns.{Style.RESET_ALL}"
            )
            print(f"Columns only in {name1}: {set(df1.columns) - set(df2.columns)}")
            print(f"Columns only in {name2}: {set(df2.columns) - set(df1.columns)}")
            print(f"Will only compare common columns.")

        self.common_columns = list(set(df1.columns).intersection(set(df2.columns)))

        self.results = {
            "completeness": {},
            "uniqueness": {},
            "consistency": {},
            "accuracy": {},
            "overall": {},
        }

    def check_completeness(self) -> Dict[str, float]:
        print(f"{Back.BLUE}{Fore.WHITE} CHECKING COMPLETENESS {Style.RESET_ALL}")

        missing1 = self.df1[self.common_columns].isna().mean() * 100
        missing2 = self.df2[self.common_columns].isna().mean() * 100

        completeness1 = 100 - missing1.mean()
        completeness2 = 100 - missing2.mean()

        print(
            f"{Fore.CYAN}Missing values in {self.name1}: {missing1.mean():.2f}% (Completeness: {completeness1:.2f}%)"
        )
        print(
            f"{Fore.CYAN}Missing values in {self.name2}: {missing2.mean():.2f}% (Completeness: {completeness2:.2f}%)"
        )

        top_missing1 = missing1.sort_values(ascending=False).head(5)
        top_missing2 = missing2.sort_values(ascending=False).head(5)

        print(f"\n{Fore.YELLOW}Top 5 columns with most missing values in {self.name1}:")
        for col, pct in top_missing1.items():
            print(f"  - {col}: {pct:.2f}%")

        print(f"\n{Fore.YELLOW}Top 5 columns with most missing values in {self.name2}:")
        for col, pct in top_missing2.items():
            print(f"  - {col}: {pct:.2f}%")

        # Store results
        self.results["completeness"] = {
            self.name1: completeness1,
            self.name2: completeness2,
            "winner": self.name1 if completeness1 > completeness2 else self.name2,
            "difference": abs(completeness1 - completeness2),
        }

        return self.results["completeness"]

    def check_uniqueness(self) -> Dict[str, float]:
        print(f"\n{Back.BLUE}{Fore.WHITE} CHECKING UNIQUENESS {Style.RESET_ALL}")

        dup1 = self.df1.duplicated().sum()
        dup2 = self.df2.duplicated().sum()

        uniqueness1 = 100 - (dup1 / len(self.df1) * 100) if len(self.df1) > 0 else 0
        uniqueness2 = 100 - (dup2 / len(self.df2) * 100) if len(self.df2) > 0 else 0

        print(
            f"{Fore.CYAN}Duplicates in {self.name1}: {dup1} rows ({100-uniqueness1:.2f}%)"
        )
        print(
            f"{Fore.CYAN}Duplicates in {self.name2}: {dup2} rows ({100-uniqueness2:.2f}%)"
        )

        categorical_cols = [
            col
            for col in self.common_columns
            if self.df1[col].dtype == "object" and self.df2[col].dtype == "object"
        ]

        if categorical_cols:
            print(f"\n{Fore.YELLOW}Unique values in categorical columns:")

            for col in categorical_cols[:5]:
                uniq1 = self.df1[col].nunique()
                uniq2 = self.df2[col].nunique()

                print(
                    f"  - Column '{col}': {self.name1}: {uniq1} unique values, {self.name2}: {uniq2} unique values"
                )

                if abs(uniq1 - uniq2) > max(uniq1, uniq2) * 0.2:
                    vals1 = set(self.df1[col].dropna().unique())
                    vals2 = set(self.df2[col].dropna().unique())

                    only_in_1 = vals1 - vals2
                    only_in_2 = vals2 - vals1

                    if only_in_1 and len(only_in_1) < 5:
                        print(f"    Values only in {self.name1}: {only_in_1}")
                    if only_in_2 and len(only_in_2) < 5:
                        print(f"    Values only in {self.name2}: {only_in_2}")

        self.results["uniqueness"] = {
            self.name1: uniqueness1,
            self.name2: uniqueness2,
            "winner": self.name1 if uniqueness1 > uniqueness2 else self.name2,
            "difference": abs(uniqueness1 - uniqueness2),
        }

        return self.results["uniqueness"]

    def check_consistency(self) -> Dict[str, float]:
        """
        Check for data consistency (outliers, standard deviations)

        Returns:
            Dictionary with consistency scores
        """
        print(f"\n{Back.BLUE}{Fore.WHITE} CHECKING CONSISTENCY {Style.RESET_ALL}")

        # Check numeric columns for outliers and consistency
        numeric_cols = [
            col
            for col in self.common_columns
            if pd.api.types.is_numeric_dtype(self.df1[col])
            and pd.api.types.is_numeric_dtype(self.df2[col])
        ]

        if not numeric_cols:
            print(f"{Fore.RED}No common numeric columns to compare!{Style.RESET_ALL}")
            self.results["consistency"] = {
                self.name1: 50,
                self.name2: 50,
                "winner": "Tie",
                "difference": 0,
            }
            return self.results["consistency"]

        # Initialize outlier counts
        outliers1 = 0
        outliers2 = 0
        total_checked1 = 0
        total_checked2 = 0

        print(f"{Fore.YELLOW}Checking for outliers in numeric columns:")

        for col in numeric_cols:
            # Skip columns with all missing values
            if self.df1[col].isna().all() or self.df2[col].isna().all():
                continue

            # Calculate IQR for both datasets
            Q1_1 = self.df1[col].quantile(0.25)
            Q3_1 = self.df1[col].quantile(0.75)
            IQR_1 = Q3_1 - Q1_1

            Q1_2 = self.df2[col].quantile(0.25)
            Q3_2 = self.df2[col].quantile(0.75)
            IQR_2 = Q3_2 - Q1_2

            # Define outlier boundaries
            lower_bound1 = Q1_1 - 1.5 * IQR_1
            upper_bound1 = Q3_1 + 1.5 * IQR_1

            lower_bound2 = Q1_2 - 1.5 * IQR_2
            upper_bound2 = Q3_2 + 1.5 * IQR_2

            # Count outliers
            outlier_count1 = (
                (self.df1[col] < lower_bound1) | (self.df1[col] > upper_bound1)
            ).sum()
            outlier_count2 = (
                (self.df2[col] < lower_bound2) | (self.df2[col] > upper_bound2)
            ).sum()

            # Add to total
            total_checked1 += len(self.df1[col].dropna())
            total_checked2 += len(self.df2[col].dropna())
            outliers1 += outlier_count1
            outliers2 += outlier_count2

            # Print details for columns with significant differences
            outlier_pct1 = (
                outlier_count1 / len(self.df1[col].dropna()) * 100
                if len(self.df1[col].dropna()) > 0
                else 0
            )
            outlier_pct2 = (
                outlier_count2 / len(self.df2[col].dropna()) * 100
                if len(self.df2[col].dropna()) > 0
                else 0
            )

            if abs(outlier_pct1 - outlier_pct2) > 5:  # 5% difference threshold
                print(f"  - Column '{col}':")
                print(
                    f"    {self.name1}: {outlier_count1} outliers ({outlier_pct1:.2f}%)"
                )
                print(
                    f"    {self.name2}: {outlier_count2} outliers ({outlier_pct2:.2f}%)"
                )
                print(
                    f"    Range in {self.name1}: [{self.df1[col].min():.2f}, {self.df1[col].max():.2f}]"
                )
                print(
                    f"    Range in {self.name2}: [{self.df2[col].min():.2f}, {self.df2[col].max():.2f}]"
                )

        # Calculate consistency percentage (100% - outlier%)
        consistency1 = (
            100 - (outliers1 / total_checked1 * 100) if total_checked1 > 0 else 0
        )
        consistency2 = (
            100 - (outliers2 / total_checked2 * 100) if total_checked2 > 0 else 0
        )

        print(
            f"\n{Fore.CYAN}Overall outliers in {self.name1}: {outliers1} values ({100-consistency1:.2f}%)"
        )
        print(
            f"{Fore.CYAN}Overall outliers in {self.name2}: {outliers2} values ({100-consistency2:.2f}%)"
        )

        # Store results
        self.results["consistency"] = {
            self.name1: consistency1,
            self.name2: consistency2,
            "winner": self.name1 if consistency1 > consistency2 else self.name2,
            "difference": abs(consistency1 - consistency2),
        }

        return self.results["consistency"]

    def check_accuracy(self) -> Dict[str, float]:
        """
        Check data accuracy and valid values

        Returns:
            Dictionary with accuracy scores
        """
        print(f"\n{Back.BLUE}{Fore.WHITE} CHECKING ACCURACY {Style.RESET_ALL}")

        accuracy_issues1 = 0
        accuracy_issues2 = 0
        total_checked1 = 0
        total_checked2 = 0

        # Check numerical columns for extreme values
        numeric_cols = [
            col
            for col in self.common_columns
            if pd.api.types.is_numeric_dtype(self.df1[col])
            and pd.api.types.is_numeric_dtype(self.df2[col])
        ]

        # Check for negative values in columns that should be positive
        likely_positive_cols = [
            col
            for col in numeric_cols
            if any(
                term in col.lower()
                for term in [
                    "price",
                    "cost",
                    "amount",
                    "age",
                    "count",
                    "area",
                    "size",
                    "length",
                    "width",
                    "height",
                    "weight",
                    "time",
                ]
            )
        ]

        for col in likely_positive_cols:
            # Count negative values
            neg_count1 = (self.df1[col] < 0).sum()
            neg_count2 = (self.df2[col] < 0).sum()

            total_checked1 += len(self.df1[col].dropna())
            total_checked2 += len(self.df2[col].dropna())
            accuracy_issues1 += neg_count1
            accuracy_issues2 += neg_count2

            # Print details for columns with issues
            if neg_count1 > 0 or neg_count2 > 0:
                print(f"{Fore.YELLOW}Column '{col}' (should be positive):")
                print(f"  - {self.name1}: {neg_count1} negative values")
                print(f"  - {self.name2}: {neg_count2} negative values")

        # Check text columns for potential format issues
        text_cols = [
            col
            for col in self.common_columns
            if self.df1[col].dtype == "object" and self.df2[col].dtype == "object"
        ]

        # Check date columns (identified by name)
        date_cols = [
            col
            for col in text_cols
            if any(
                term in col.lower()
                for term in [
                    "date",
                    "time",
                    "day",
                    "month",
                    "year",
                    "updated",
                    "created",
                ]
            )
        ]

        for col in date_cols:
            # Try to parse as date and count failures
            valid_dates1 = 0
            valid_dates2 = 0

            # Sample data for performance if datasets are large
            sample_size = min(1000, len(self.df1), len(self.df2))
            sample1 = (
                self.df1[col]
                .dropna()
                .sample(min(sample_size, len(self.df1[col].dropna())))
                if not self.df1[col].dropna().empty
                else pd.Series()
            )
            sample2 = (
                self.df2[col]
                .dropna()
                .sample(min(sample_size, len(self.df2[col].dropna())))
                if not self.df2[col].dropna().empty
                else pd.Series()
            )

            for val in sample1:
                try:
                    pd.to_datetime(val)
                    valid_dates1 += 1
                except:
                    pass

            for val in sample2:
                try:
                    pd.to_datetime(val)
                    valid_dates2 += 1
                except:
                    pass

            invalid_dates1 = len(sample1) - valid_dates1
            invalid_dates2 = len(sample2) - valid_dates2

            # Scale up to estimate total invalid dates
            if len(sample1) > 0:
                total_checked1 += len(self.df1[col].dropna())
                accuracy_issues1 += int(
                    invalid_dates1 / len(sample1) * len(self.df1[col].dropna())
                )

            if len(sample2) > 0:
                total_checked2 += len(self.df2[col].dropna())
                accuracy_issues2 += int(
                    invalid_dates2 / len(sample2) * len(self.df2[col].dropna())
                )

            # Print details for columns with issues
            if invalid_dates1 > 0 or invalid_dates2 > 0:
                print(f"{Fore.YELLOW}Column '{col}' (date format issues):")
                print(
                    f"  - {self.name1}: ~{invalid_dates1/len(sample1)*100 if len(sample1)>0 else 0:.1f}% invalid format"
                )
                print(
                    f"  - {self.name2}: ~{invalid_dates2/len(sample2)*100 if len(sample2)>0 else 0:.1f}% invalid format"
                )

                # Show examples of potentially invalid dates
                if invalid_dates1 > 0 and len(sample1) > 0:
                    invalid_examples1 = [
                        val for val in sample1[:5] if not self._is_valid_date(val)
                    ]
                    if invalid_examples1:
                        print(f"    Examples from {self.name1}: {invalid_examples1}")

                if invalid_dates2 > 0 and len(sample2) > 0:
                    invalid_examples2 = [
                        val for val in sample2[:5] if not self._is_valid_date(val)
                    ]
                    if invalid_examples2:
                        print(f"    Examples from {self.name2}: {invalid_examples2}")

        # Calculate accuracy percentage
        accuracy1 = (
            100 - (accuracy_issues1 / total_checked1 * 100) if total_checked1 > 0 else 0
        )
        accuracy2 = (
            100 - (accuracy_issues2 / total_checked2 * 100) if total_checked2 > 0 else 0
        )

        print(
            f"\n{Fore.CYAN}Overall accuracy issues in {self.name1}: {accuracy_issues1} values ({100-accuracy1:.2f}%)"
        )
        print(
            f"{Fore.CYAN}Overall accuracy issues in {self.name2}: {accuracy_issues2} values ({100-accuracy2:.2f}%)"
        )

        # Store results
        self.results["accuracy"] = {
            self.name1: accuracy1,
            self.name2: accuracy2,
            "winner": self.name1 if accuracy1 > accuracy2 else self.name2,
            "difference": abs(accuracy1 - accuracy2),
        }

        return self.results["accuracy"]

    def _is_valid_date(self, val):
        try:
            pd.to_datetime(val)
            return True
        except:
            return False

    def get_overlap_percentage(self) -> Tuple[float, float]:
        print(f"\n{Back.BLUE}{Fore.WHITE} CHECKING DATA OVERLAP {Style.RESET_ALL}")

        # Try to find common key columns for matching
        potential_key_cols = [
            col
            for col in self.common_columns
            if col.lower() in ["id", "key", "code", "identifier", "name", "title"]
        ]

        if not potential_key_cols:
            # If no obvious key columns, use all columns
            print(
                f"{Fore.YELLOW}No obvious key columns found. Using all columns to identify similar rows."
            )
            key_cols = self.common_columns
        else:
            key_cols = potential_key_cols
            print(f"{Fore.YELLOW}Using key columns: {key_cols}")

        # Count matches (using a sample if datasets are large)
        sample_size = min(5000, len(self.df1), len(self.df2))
        if len(self.df1) > sample_size or len(self.df2) > sample_size:
            print(
                f"{Fore.YELLOW}Datasets are large, using a sample of {sample_size} rows for overlap analysis."
            )
            df1_sample = (
                self.df1.sample(sample_size)
                if len(self.df1) > sample_size
                else self.df1
            )
            df2_sample = (
                self.df2.sample(sample_size)
                if len(self.df2) > sample_size
                else self.df2
            )
        else:
            df1_sample = self.df1
            df2_sample = self.df2

        matches_in_df1 = 0
        for _, row1 in df1_sample.iterrows():
            if any((df2_sample[key_cols] == row1[key_cols]).all(axis=1)):
                matches_in_df1 += 1

        matches_in_df2 = 0
        for _, row2 in df2_sample.iterrows():
            if any((df1_sample[key_cols] == row2[key_cols]).all(axis=1)):
                matches_in_df2 += 1

        # Calculate percentages
        pct_of_df1_in_df2 = matches_in_df1 / len(df1_sample) * 100
        pct_of_df2_in_df1 = matches_in_df2 / len(df2_sample) * 100

        print(
            f"{Fore.CYAN}{pct_of_df1_in_df2:.1f}% of rows in {self.name1} seem to be present in {self.name2}"
        )
        print(
            f"{Fore.CYAN}{pct_of_df2_in_df1:.1f}% of rows in {self.name2} seem to be present in {self.name1}"
        )

        if pct_of_df1_in_df2 > 95 and pct_of_df2_in_df1 > 95:
            print(
                f"{Fore.MAGENTA}Datasets appear to be nearly identical!{Style.RESET_ALL}"
            )
        elif pct_of_df1_in_df2 > 90 or pct_of_df2_in_df1 > 90:
            print(f"{Fore.MAGENTA}Datasets have significant overlap!{Style.RESET_ALL}")

        return pct_of_df1_in_df2, pct_of_df2_in_df1

    def summarize_results(self) -> Dict[str, Any]:
        print(
            f"\n{Back.GREEN}{Fore.BLACK} OVERALL COMPARISON RESULTS {Style.RESET_ALL}"
        )

        weights = {
            "completeness": 0.35,
            "consistency": 0.30,
            "accuracy": 0.25,
            "uniqueness": 0.10,
        }

        overall_score1 = sum(
            self.results[metric][self.name1] * weights[metric]
            for metric in weights.keys()
        )
        overall_score2 = sum(
            self.results[metric][self.name2] * weights[metric]
            for metric in weights.keys()
        )

        # Determine overall winner
        winner = self.name1 if overall_score1 > overall_score2 else self.name2
        score_diff = abs(overall_score1 - overall_score2)

        # Store overall results
        self.results["overall"] = {
            self.name1: overall_score1,
            self.name2: overall_score2,
            "winner": winner,
            "difference": score_diff,
        }

        # Print summary table
        print(f"{Fore.CYAN}Dataset Quality Scores (higher is better):{Style.RESET_ALL}")
        print(
            f"{'Metric':<15} | {self.name1:<10} | {self.name2:<10} | {'Winner':<10} | {'Difference':<10}"
        )
        print("-" * 62)

        for metric in ["completeness", "uniqueness", "consistency", "accuracy"]:
            winner_name = self.results[metric]["winner"]
            diff = self.results[metric]["difference"]
            score1 = self.results[metric][self.name1]
            score2 = self.results[metric][self.name2]

            print(
                f"{metric.capitalize():<15} | {score1:<10.2f} | {score2:<10.2f} | {winner_name:<10} | {diff:<10.2f}"
            )

        print("-" * 62)
        print(
            f"{'OVERALL':<15} | {overall_score1:<10.2f} | {overall_score2:<10.2f} | {winner:<10} | {score_diff:<10.2f}"
        )

        # Print interpretation
        print(f"\n{Back.BLUE}{Fore.WHITE} CLEANER DATASET: {winner} {Style.RESET_ALL}")

        if score_diff < 5:
            cleanness_desc = "slightly cleaner"
        elif score_diff < 10:
            cleanness_desc = "moderately cleaner"
        else:
            cleanness_desc = "significantly cleaner"

        print(
            f"\n{Fore.GREEN}The {winner} is {cleanness_desc} than the other dataset.{Style.RESET_ALL}"
        )

        # Summarize key issues
        print(f"\n{Fore.YELLOW}Key reasons why {winner} is cleaner:{Style.RESET_ALL}")

        for metric in ["completeness", "uniqueness", "consistency", "accuracy"]:
            if (
                self.results[metric]["winner"] == winner
                and self.results[metric]["difference"] > 2
            ):
                if metric == "completeness":
                    print(
                        f"- Has {self.results[metric]['difference']:.1f}% fewer missing values"
                    )
                elif metric == "uniqueness":
                    print(
                        f"- Has {self.results[metric]['difference']:.1f}% fewer duplicate entries"
                    )
                elif metric == "consistency":
                    print(
                        f"- Has {self.results[metric]['difference']:.1f}% fewer outliers/inconsistencies"
                    )
                elif metric == "accuracy":
                    print(
                        f"- Has {self.results[metric]['difference']:.1f}% fewer data format/value issues"
                    )

        return self.results["overall"]

    def run_full_comparison(self) -> Dict[str, Any]:
        self.check_completeness()
        self.check_uniqueness()
        self.check_consistency()
        self.check_accuracy()
        self.get_overlap_percentage()
        return self.summarize_results()

    def generate_report(self, output_dir="./"):
        try:
            # Create plots
            plt.figure(figsize=(10, 6))

            metrics = [
                "completeness",
                "uniqueness",
                "consistency",
                "accuracy",
                "overall",
            ]
            scores1 = [self.results[m][self.name1] for m in metrics]
            scores2 = [self.results[m][self.name2] for m in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 8))
            rects1 = ax.bar(x - width / 2, scores1, width, label=self.name1)
            rects2 = ax.bar(x + width / 2, scores2, width, label=self.name2)

            ax.set_title("Dataset Quality Comparison", fontsize=16)
            ax.set_ylabel("Score (higher is better)", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=12)
            ax.legend()

            # Add score labels on bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(
                        f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

            autolabel(rects1)
            autolabel(rects2)

            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(output_dir, f"dataset_comparison_{timestamp}")
            plt.tight_layout()
            plt.savefig(f"{report_file}.png")

            # Build HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dataset Comparison Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; }}
                    .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #3498db; color: white; }}
                    .winner {{ font-weight: bold; color: #27ae60; }}
                    .chart {{ margin: 30px 0; text-align: center; }}
                    .footer {{ margin-top: 30px; font-size: 0.8em; color: #7f8c8d; }}
                </style>
            </head>
            <body>
                <h1>Dataset Comparison Report</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>The <span class="winner">{self.results["overall"]["winner"]}</span> is cleaner with an overall score of 
                    {self.results["overall"][self.results["overall"]["winner"]]:.2f}/100 
                    (difference: {self.results["overall"]["difference"]:.2f} points).</p>
                </div>
                
                <h2>Detailed Scores</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>{self.name1}</th>
                        <th>{self.name2}</th>
                        <th>Winner</th>
                        <th>Difference</th>
                    </tr>
                    <tr>
                        <td>Completeness</td>
                        <td>{self.results["completeness"][self.name1]:.2f}</td>
                        <td>{self.results["completeness"][self.name2]:.2f}</td>
                        <td>{self.results["completeness"]["winner"]}</td>
                        <td>{self.results["completeness"]["difference"]:.2f}</td>
                    </tr>
                    <tr>
                        <td>Uniqueness</td>
                        <td>{self.results["uniqueness"][self.name1]:.2f}</td>
                        <td>{self.results["uniqueness"][self.name2]:.2f}</td>
                        <td>{self.results["uniqueness"]["winner"]}</td>
                        <td>{self.results["uniqueness"]["difference"]:.2f}</td>
                    </tr>
                    <tr>
                        <td>Consistency</td>
                        <td>{self.results["consistency"][self.name1]:.2f}</td>
                        <td>{self.results["consistency"][self.name2]:.2f}</td>
                        <td>{self.results["consistency"]["winner"]}</td>
                        <td>{self.results["consistency"]["difference"]:.2f}</td>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>{self.results["accuracy"][self.name1]:.2f}</td>
                        <td>{self.results["accuracy"][self.name2]:.2f}</td>
                        <td>{self.results["accuracy"]["winner"]}</td>
                        <td>{self.results["accuracy"]["difference"]:.2f}</td>
                    </tr>
                    <tr>
                        <td>Overall</td>
                        <td>{self.results["overall"][self.name1]:.2f}</td>
                        <td>{self.results["overall"][self.name2]:.2f}</td>
                        <td>{self.results["overall"]["winner"]}</td>
                        <td>{self.results["overall"]["difference"]:.2f}</td>
                    </tr>
                </table>
                <div class="chart">
                    <h2>Comparison Chart</h2>
                    <img src="{report_file}.png" alt="Dataset Comparison Chart">
                </div>
                <div class="footer">
                    <p>Report generated by DatasetComparator</p>
                </div>
            </body>
            </html>
            """
            with open(f"{report_file}.html", "w") as f:
                f.write(html_content)
            print(f"{Fore.GREEN}Report generated: {report_file}.html{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error generating report: {e}{Style.RESET_ALL}")
            raise
        finally:
            plt.close()
            if os.path.exists(f"{report_file}.png"):
                os.remove(f"{report_file}.png")
            print(
                f"{Fore.YELLOW}Temporary chart file removed: {report_file}.png{Style.RESET_ALL}"
            )
        return f"{report_file}.html"


def main():
    current_dir = os.path.dirname(__file__)

    df1_path = os.path.join(current_dir, "../houses.csv")
    df2_path = os.path.join(current_dir, "../houses-cleaned.csv")
    output_dir = os.path.join(current_dir, "./reports")

    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    os.makedirs(output_dir, exist_ok=True)

    comparator = DatasetComparator(df1, df2, name1="Dataset 1", name2="Dataset 2")

    results = comparator.run_full_comparison()
    comparator.generate_report(output_dir)

    print(
        f"\n{Fore.GREEN}Comparison completed! Report saved to {output_dir}{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()
