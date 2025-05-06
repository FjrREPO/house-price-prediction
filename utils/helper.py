import re
import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import stats


# Custom exceptions for data preprocessing
class PreprocessingError(Exception):
    

    pass


class DataValidationError(PreprocessingError):
    

    pass


class FeatureEngineeringError(PreprocessingError):
    

    pass


class DataScalingError(PreprocessingError):
    

    pass


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("house_price_prediction")

# Data quality thresholds
QUALITY_THRESHOLDS = {
    "price": {
        "min": 100e6,  # 100 million rupiah minimum reasonable price
        "max": 20e9,  # 20 billion rupiah maximum reasonable price
    },
    "LT": {
        "min": 15,  # 15 m² minimum land area
        "max": 5000,  # 5000 m² maximum land area
    },
    "LB": {
        "min": 10,  # 10 m² minimum building area
        "max": 2000,  # 2000 m² maximum building area
    },
    "bedroom": {"min": 1, "max": 20},  # Minimum 1 bedroom  # Maximum 20 bedrooms
    "bathroom": {"min": 1, "max": 15},  # Minimum 1 bathroom  # Maximum 15 bathrooms
    "carport": {"min": 0, "max": 10},  # Minimum 0 carports  # Maximum 10 carports
}

# Thresholds for outlier detection by feature
IQR_THRESHOLDS = {
    "price": 2.0,  # More lenient for price
    "LT": 1.75,  # Land area can vary widely
    "LB": 1.75,  # Building area can vary widely
    "bedroom": 1.5,  # Standard
    "bathroom": 1.5,  # Standard
    "carport": 2.0,  # More lenient
}

# Z-score threshold for each feature
Z_SCORE_THRESHOLDS = {
    "price": 3.0,
    "LT": 3.0,
    "LB": 3.0,
    "bedroom": 2.5,
    "bathroom": 2.5,
    "carport": 3.0,
}


# Function to perform feature selection
def select_features(
    X: pd.DataFrame, y: pd.Series, method: str = "f_regression", k: int = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the most important features based on statistical tests.

    Args:
        X: Feature DataFrame
        y: Target variable
        method: Feature selection method ('f_regression' or 'mutual_info')
        k: Number of features to select (if None, selects half of available features)

    Returns:
        Tuple of (DataFrame with selected features, list of selected feature names)
    """
    if k is None:
        k = max(1, X.shape[1] // 2)  # Select half of features by default, minimum 1

    # Choose the scoring function
    if method == "f_regression":
        score_func = f_regression
    elif method == "mutual_info":
        score_func = mutual_info_regression
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    try:
        # Apply feature selection
        selector = SelectKBest(score_func=score_func, k=k)
        X_new = selector.fit_transform(X, y)

        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()

        # Create DataFrame with selected features
        X_selected = pd.DataFrame(X_new, columns=selected_features, index=X.index)

        # Log feature importance scores
        feature_scores = pd.DataFrame(
            {"Feature": X.columns, "Score": selector.scores_}
        ).sort_values(by="Score", ascending=False)

        logger.info(
            f"Feature selection completed using {method}. Selected {len(selected_features)} features: {', '.join(selected_features)}"
        )
        logger.info(
            f"Top 5 features: {', '.join(feature_scores['Feature'].head(5).tolist())}"
        )

        return X_selected, selected_features

    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise ValueError(f"Feature selection failed: {str(e)}")


# Function to scale features
def scale_features(
    df: pd.DataFrame,
    price_cols: List[str] = None,
    numeric_cols: List[str] = None,
    fit_scalers: bool = True,
    scalers: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Scale features using appropriate scalers.

    Args:
        df: DataFrame to scale
        price_cols: List of price-related columns to scale with RobustScaler
        numeric_cols: List of numeric columns to scale with StandardScaler
        fit_scalers: Whether to fit new scalers or use provided ones
        scalers: Dictionary of pre-fitted scalers (used when fit_scalers=False)

    Returns:
        Tuple of (DataFrame with scaled features, dictionary of fitted scalers)
    """
    df_scaled = df.copy()

    # Default columns if not specified
    if price_cols is None:
        price_cols = [col for col in df.columns if "price" in col.lower()]

    if numeric_cols is None:
        # Get numeric columns that aren't price-related
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        numeric_cols = [col for col in numeric_cols if col not in price_cols]

    # Initialize scalers dictionary if not provided
    if scalers is None:
        scalers = {}

    try:
        # Scale price-related features with RobustScaler (handles outliers better)
        if price_cols:
            price_data = df_scaled[price_cols]

            if fit_scalers:
                price_scaler = RobustScaler()
                scalers["price_scaler"] = price_scaler
            else:
                price_scaler = scalers.get("price_scaler")
                if price_scaler is None:
                    raise DataScalingError(
                        "Price scaler not provided but fit_scalers=False"
                    )

            # Fit and transform or just transform
            if fit_scalers:
                df_scaled[price_cols] = price_scaler.fit_transform(price_data)
            else:
                df_scaled[price_cols] = price_scaler.transform(price_data)

        # Scale other numeric features with StandardScaler
        if numeric_cols:
            numeric_data = df_scaled[numeric_cols]

            if fit_scalers:
                numeric_scaler = StandardScaler()
                scalers["numeric_scaler"] = numeric_scaler
            else:
                numeric_scaler = scalers.get("numeric_scaler")
                if numeric_scaler is None:
                    raise DataScalingError(
                        "Numeric scaler not provided but fit_scalers=False"
                    )

            # Fit and transform or just transform
            if fit_scalers:
                df_scaled[numeric_cols] = numeric_scaler.fit_transform(numeric_data)
            else:
                df_scaled[numeric_cols] = numeric_scaler.transform(numeric_data)

        logger.info(
            f"Data scaling completed. Scaled {len(price_cols)} price features and {len(numeric_cols)} numeric features"
        )
        return df_scaled, scalers

    except Exception as e:
        logger.error(f"Error during data scaling: {str(e)}")
        raise DataScalingError(f"Data scaling failed: {str(e)}")


def remove_outliers(
    df: pd.DataFrame, columns: List[str], method: str = "both"
) -> pd.DataFrame:
    """
    Remove outliers from the dataset using IQR and/or Z-score methods.

    Args:
        df: DataFrame to process
        columns: List of column names to check for outliers
        method: Outlier detection method ('iqr', 'zscore', or 'both')

    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    original_count = len(df_clean)

    if method in ["iqr", "both"]:
        for column in columns:
            if column in df_clean.columns:
                # Get the appropriate threshold for this feature
                threshold = IQR_THRESHOLDS.get(column, 1.5)

                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Count outliers before removal
                outliers_count = df_clean[
                    (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
                ].shape[0]

                # Remove outliers
                df_clean = df_clean[
                    (df_clean[column] >= lower_bound)
                    & (df_clean[column] <= upper_bound)
                ]

                logger.info(
                    f"IQR method: Removed {outliers_count} outliers from {column} "
                    f"(threshold: {threshold})"
                )

    if method in ["zscore", "both"]:
        for column in columns:
            if column in df_clean.columns:
                # Get the appropriate threshold for this feature
                threshold = Z_SCORE_THRESHOLDS.get(column, 3.0)

                # Calculate z-scores
                z_scores = np.abs(stats.zscore(df_clean[column], nan_policy="omit"))

                # Count outliers before removal
                outliers_count = df_clean[z_scores > threshold].shape[0]

                # Remove outliers
                df_clean = df_clean[z_scores <= threshold]

                logger.info(
                    f"Z-score method: Removed {outliers_count} outliers from {column} "
                    f"(threshold: {threshold})"
                )

    removed_count = original_count - len(df_clean)
    removed_percentage = (
        (removed_count / original_count) * 100 if original_count > 0 else 0
    )
    logger.info(
        f"Removed {removed_count} outliers in total ({removed_percentage:.2f}% of data)"
    )

    return df_clean


def validate_dataframe(
    df: pd.DataFrame, required_columns: List[str], raise_exception: bool = False
) -> Tuple[bool, str]:
    """
    Validate the input DataFrame for required columns and data quality.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if df.empty:
        error_msg = "DataFrame is empty"
        if raise_exception:
            raise DataValidationError(error_msg)
        return False, error_msg

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}"
        if raise_exception:
            raise DataValidationError(error_msg)
        return False, error_msg

    # Check for all-null columns
    null_columns = [
        col for col in required_columns if col in df.columns and df[col].isna().all()
    ]
    if null_columns:
        error_msg = f"Columns with all null values: {', '.join(null_columns)}"
        if raise_exception:
            raise DataValidationError(error_msg)
        return False, error_msg

    # Check for data type consistency
    type_issues = []
    for col in df.columns:
        if col in df.select_dtypes(include=["object"]).columns:
            # Check if column should be numeric but contains non-numeric strings
            if any(
                col.lower().endswith(suffix)
                for suffix in ["price", "area", "lt", "lb", "size"]
            ):
                try:
                    pd.to_numeric(df[col], errors="raise")
                except:
                    type_issues.append(col)

    if type_issues:
        error_msg = f"Columns with potential type issues: {', '.join(type_issues)}"
        if raise_exception:
            raise DataValidationError(error_msg)
        return False, error_msg

    return True, "Validation passed"


def convert_to_numeric(value: Union[str, int, float, None]) -> Optional[float]:
    """
    Convert price string to numeric value.

    Args:
        value: Price value to convert, which can be a formatted string or numeric

    Returns:
        Numeric price value or None if conversion fails
    """
    # If already numeric, return as is
    if isinstance(value, (int, float)):
        return float(value)

    # If None or NaN, return None
    if value is None or pd.isna(value):
        return None

    try:
        # Convert string to appropriate format
        value_str = str(value)
        value_numeric = re.sub(r"Rp\s?", "", value_str)

        # Handle "Miliar" (billions)
        if "Miliar" in value_numeric:
            cleaned_value = re.sub(r"\s?Miliar", "", value_numeric).replace(",", ".")
            return float(cleaned_value) * 1e9

        # Handle "Juta" (millions)
        elif "Juta" in value_numeric:
            cleaned_value = re.sub(r"\s?Juta", "", value_numeric).replace(",", ".")
            return float(cleaned_value) * 1e6

        # Try to convert numeric strings
        elif re.match(r"^\d+(\.\d+)?$", value_numeric.strip()):
            return float(value_numeric)

        # Can't determine format
        else:
            logger.warning(f"Couldn't parse price value: {value}")
            return None

    except (ValueError, TypeError) as e:
        logger.warning(f"Error converting price value {value}: {str(e)}")
        return None


def preprocess_updated(updated: Any) -> Optional[datetime]:
    """
    Convert relative time strings to datetime objects.

    Args:
        updated: String indicating when the listing was updated (e.g., "3 hari lalu")

    Returns:
        Datetime object or None if conversion fails
    """
    # Return None for missing values
    if updated is None or pd.isna(updated):
        return None

    # If already a datetime, return as is
    if isinstance(updated, datetime):
        return updated

    # Convert string to datetime
    try:
        updated_str = str(updated).lower()
        today = datetime.today()

        # Extract the number from the string
        number_match = re.search(r"(\d+)", updated_str)
        if not number_match:
            logger.warning(f"Couldn't extract number from updated value: {updated}")
            return None

        time_value = int(number_match.group(1))

        # Process based on time unit
        if "minggu" in updated_str:
            return today - timedelta(weeks=time_value)
        elif "bulan" in updated_str:
            # More accurate than 30 days per month
            return today - timedelta(days=time_value * 30.44)
        elif "hari" in updated_str:
            return today - timedelta(days=time_value)
        elif "jam" in updated_str:
            return today - timedelta(hours=time_value)
        elif "menit" in updated_str:
            return today - timedelta(minutes=time_value)
        else:
            # Try to parse as explicit date
            try:
                # Try common Indonesian date formats
                for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d %b %Y"]:
                    try:
                        return datetime.strptime(updated_str, fmt)
                    except ValueError:
                        continue

                logger.warning(f"Unknown time format in updated value: {updated}")
                return None
            except:
                logger.warning(f"Failed to parse date: {updated}")
                return None

    except Exception as e:
        logger.warning(f"Error processing updated value {updated}: {str(e)}")
        return None


def preprocess_area(area_str: Any) -> Optional[float]:
    """
    Extract and convert area values from strings.

    Args:
        area_str: String containing area information (e.g., "150 m²")

    Returns:
        Numeric area value in square meters or None if conversion fails
    """
    # If already numeric, return as is
    if isinstance(area_str, (int, float)):
        return float(area_str) if not pd.isna(area_str) else None

    # If None or NaN, return None
    if area_str is None or pd.isna(area_str):
        return None

    try:
        # Convert to string for processing
        area_text = str(area_str).strip()

        # Handle standard format: "150 m²"
        match = re.search(r"(\d+(\.\d+)?)\s*m²", area_text)
        if match:
            return float(match.group(1))

        # Handle format: "150m²"
        match = re.search(r"(\d+(\.\d+)?)m²", area_text)
        if match:
            return float(match.group(1))

        # Handle format with just numbers (assumes square meters)
        match = re.search(r"^(\d+(\.\d+)?)$", area_text)
        if match:
            return float(match.group(1))

        # Handle format with commas: "150,5"
        match = re.search(r"^(\d+,\d+)$", area_text)
        if match:
            return float(match.group(1).replace(",", "."))

        # Log when we can't parse the area
        logger.warning(f"Couldn't parse area value: {area_str}")
        return None

    except (ValueError, TypeError) as e:
        logger.warning(f"Error processing area value {area_str}: {str(e)}")
        return None


def preprocess_carport(carport_str: Any) -> int:
    """
    Preprocess carport data by converting it to numeric values.

    Args:
        carport_str: String or integer representing the number of carports

    Returns:
        Integer representing the number of carports or 0 if missing
    """
    try:
        if pd.isna(carport_str):
            return 0
        elif isinstance(carport_str, (int, float)) and not pd.isna(carport_str):
            return int(carport_str)
        elif isinstance(carport_str, str) and carport_str.strip().isdigit():
            return int(carport_str)
        else:
            # Try to extract numeric value using regex
            match = re.search(r"(\d+)", str(carport_str))
            if match:
                return int(match.group(1))
            # If all else fails, assume no carport
            logger.warning(
                f"Could not parse carport value: {carport_str}, defaulting to 0"
            )
            return 0
    except Exception as e:
        logger.warning(f"Error processing carport value {carport_str}: {str(e)}")
        return 0


def preprocess_location(location):
    """
    Extract kecamatan (subdistrict) and kabupaten/kota (regency/city) from location string.

    Args:
        location: String containing location information

    Returns:
        Series with kecamatan and kabupaten_kota
    """
    if pd.isna(location):
        return pd.Series([None, None])
    split_location = location.split(",")
    kecamatan = split_location[0].strip() if len(split_location) > 0 else None
    kabupaten_kota = split_location[1].strip() if len(split_location) > 1 else None
    return pd.Series([kecamatan, kabupaten_kota])


def preprocess_data(
    df: pd.DataFrame,
    validate: bool = True,
    scale_data: bool = False,
    feature_selection: bool = False,
    required_columns: List[str] = None,
    output_dir: str = "reports",
) -> pd.DataFrame:
    """
    Preprocess the housing data for model training:
    - Convert values to appropriate types
    - Extract and encode features
    - Handle missing values
    - Remove outliers and invalid entries
    - Perform feature engineering
    - Scale features (optional)
    - Select features (optional)

    Args:
        df: Raw housing data DataFrame
        validate: Whether to validate data before processing
        scale_data: Whether to scale numeric features
        feature_selection: Whether to perform feature selection
        required_columns: List of columns that must be present
        output_dir: Directory to save reports

    Returns:
        Preprocessed DataFrame ready for modeling

    Raises:
        DataValidationError: If validation fails and validate=True
    """
    # Define default required columns if not provided
    if required_columns is None:
        required_columns = ["price", "LT", "LB", "location", "bedroom", "bathroom"]

    # Validate input data
    if validate:
        is_valid, error_message = validate_dataframe(
            df, required_columns, raise_exception=False
        )
        if not is_valid:
            logger.error(f"Data validation failed: {error_message}")
            if validate:
                raise DataValidationError(error_message)

    logger.info(f"Starting preprocessing of {len(df)} rows of data")
    df_processed = df.copy()

    # Convert string values to appropriate numeric types
    df_processed["price"] = df_processed["price"].apply(convert_to_numeric)
    df_processed["LT"] = df_processed["LT"].apply(preprocess_area)
    df_processed["LB"] = df_processed["LB"].apply(preprocess_area)
    df_processed["updated"] = df_processed["updated"].apply(preprocess_updated)
    df_processed[["kecamatan", "kabupaten_kota"]] = df_processed["location"].apply(
        preprocess_location
    )
    df_processed["carport"] = df_processed["carport"].apply(preprocess_carport)

    # Convert updated to datetime
    df_processed["updated"] = pd.to_datetime(df_processed["updated"])

    # Remove non-residential properties (hotels, boarding houses, etc.)
    df_processed = df_processed[
        ~df_processed["title"].str.contains("hotel|kost|Kos", case=False, na=False)
    ]

    # Drop duplicates based on title
    df_processed = df_processed.drop_duplicates(subset="title", keep="first")

    # Filter out listings older than 60 days
    sixty_days_ago = datetime.today() - timedelta(days=60)
    df_processed = df_processed[df_processed["updated"] >= sixty_days_ago]

    # Handle missing values in required columns
    df_processed["price"] = df_processed["price"].fillna(df_processed["price"].median())
    df_processed["LT"] = df_processed["LT"].fillna(df_processed["LT"].median())
    df_processed["LB"] = df_processed["LB"].fillna(df_processed["LB"].median())
    df_processed["bedroom"] = df_processed["bedroom"].fillna(
        df_processed["bedroom"].median()
    )
    df_processed["bathroom"] = df_processed["bathroom"].fillna(
        df_processed["bathroom"].median()
    )

    # Drop rows with missing values in critical columns
    df_processed = df_processed.dropna(
        subset=["updated", "price", "LT", "LB", "bedroom", "bathroom"]
    )

    # Data quality checks
    try:
        logger.info("Performing data quality checks...")

        # Check for reasonable price values
        price_min = QUALITY_THRESHOLDS["price"]["min"]
        price_max = QUALITY_THRESHOLDS["price"]["max"]
        invalid_prices = df_processed[
            (df_processed["price"] < price_min) | (df_processed["price"] > price_max)
        ]

        if not invalid_prices.empty:
            logger.warning(
                f"Found {len(invalid_prices)} rows with price values outside reasonable range "
                f"({price_min:,.0f} - {price_max:,.0f})"
            )
            df_processed = df_processed[
                (df_processed["price"] >= price_min)
                & (df_processed["price"] <= price_max)
            ]

        # Check area ratios (LB shouldn't be larger than LT for normal houses)
        invalid_areas = df_processed[df_processed["LB"] > df_processed["LT"] * 1.1]
        if not invalid_areas.empty:
            logger.warning(
                f"Found {len(invalid_areas)} rows where building area (LB) > land area (LT)"
            )
            # We'll keep these rows, but flag them for review
            df_processed.loc[
                df_processed["LB"] > df_processed["LT"] * 1.1, "area_ratio_flag"
            ] = True
            df_processed["area_ratio_flag"] = df_processed["area_ratio_flag"].fillna(
                False
            )

        # Check logical constraints in room counts
        # Too many bedrooms/bathrooms for the building size
        avg_room_size = 9  # Assume average bedroom/bathroom needs about 9 m²
        df_processed["total_rooms"] = df_processed["bedroom"] + df_processed["bathroom"]
        df_processed["min_area_needed"] = df_processed["total_rooms"] * avg_room_size

        invalid_rooms = df_processed[
            df_processed["min_area_needed"] > df_processed["LB"]
        ]
        if not invalid_rooms.empty:
            logger.warning(
                f"Found {len(invalid_rooms)} rows with too many rooms for the building size"
            )
            # Flag these for review
            df_processed.loc[
                df_processed["min_area_needed"] > df_processed["LB"], "room_count_flag"
            ] = True
            df_processed["room_count_flag"] = df_processed["room_count_flag"].fillna(
                False
            )

        # Log data quality metrics
        logger.info(
            f"Data quality checks completed. Remaining rows: {len(df_processed)}"
        )

    except Exception as e:
        logger.error(f"Error during data quality checks: {str(e)}")

    # Encode categorical variables
    # Only encode kecamatan if it will be used in modeling
    if "kecamatan" in df_processed.columns and df_processed["kecamatan"].notna().any():
        # Fill NaN values with "Unknown"
        df_processed["kecamatan"] = df_processed["kecamatan"].fillna("Unknown")

        # Encode kecamatan using LabelEncoder
        le = LabelEncoder()
        df_processed["kecamatan_encoded"] = le.fit_transform(df_processed["kecamatan"])

        # Store the encoder mapping for reference
        kecamatan_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        # Save encoding mapping for reference
        os.makedirs(output_dir, exist_ok=True)
        try:
            with open(f"{output_dir}/kecamatan_encoding.json", "w") as f:
                json.dump(kecamatan_mapping, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save kecamatan encoding mapping: {str(e)}")

    # Feature Engineering
    try:
        logger.info("Performing feature engineering...")

        # Price per square meter (land)
        df_processed["price_per_m2_land"] = df_processed.apply(
            lambda row: row["price"] / row["LT"] if row["LT"] > 0 else np.nan, axis=1
        )

        # Price per square meter (building)
        df_processed["price_per_m2_building"] = df_processed.apply(
            lambda row: row["price"] / row["LB"] if row["LB"] > 0 else np.nan, axis=1
        )

        # Building efficiency ratio (LB/LT)
        df_processed["building_efficiency"] = df_processed.apply(
            lambda row: row["LB"] / row["LT"] if row["LT"] > 0 else np.nan, axis=1
        )

        # Age of listing in days
        df_processed["listing_age_days"] = (
            datetime.today() - df_processed["updated"]
        ).dt.days

        # Frequency encoding for kecamatan
        if "kecamatan" in df_processed.columns:
            freq_encoding = (
                df_processed["kecamatan"].value_counts(normalize=True).to_dict()
            )
            df_processed["kecamatan_freq"] = df_processed["kecamatan"].map(
                freq_encoding
            )

        # Fill any NaN values created during feature engineering
        numeric_cols = df_processed.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            if df_processed[col].isna().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())

        logger.info(
            f"Feature engineering completed, added {4 + ('kecamatan' in df_processed.columns)} new features"
        )

    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise FeatureEngineeringError(f"Feature engineering failed: {str(e)}")

    # Generate data quality report
    try:
        quality_metrics = {
            "total_rows": len(df_processed),
            "duplicate_count": len(df) - len(df.drop_duplicates()),
            "missing_values": {
                col: int(df_processed[col].isna().sum()) for col in df_processed.columns
            },
            "numeric_features": {
                col: {
                    "mean": float(df_processed[col].mean()),
                    "median": float(df_processed[col].median()),
                    "min": float(df_processed[col].min()),
                    "max": float(df_processed[col].max()),
                    "std": float(df_processed[col].std()),
                }
                for col in df_processed.select_dtypes(
                    include=["float64", "int64"]
                ).columns
            },
            "categorical_features": {
                col: {
                    "unique_values": int(df_processed[col].nunique()),
                    "most_common": (
                        df_processed[col].value_counts().index[0]
                        if not df_processed[col].value_counts().empty
                        else None
                    ),
                }
                for col in df_processed.select_dtypes(
                    include=["object", "category"]
                ).columns
            },
        }

        # Save quality metrics to file
        os.makedirs("reports", exist_ok=True)
        with open("reports/data_quality_metrics.json", "w") as f:
            json.dump(quality_metrics, f, indent=4, default=str)

        logger.info(
            f"Data quality report generated with {len(quality_metrics['numeric_features'])} numeric and {len(quality_metrics['categorical_features'])} categorical features"
        )

    except Exception as e:
        logger.warning(f"Failed to generate data quality report: {str(e)}")

    # Perform data scaling if requested
    scalers = {}
    if scale_data:
        try:
            logger.info("Scaling numeric features...")
            # Define price and numeric columns
            price_cols = [col for col in df_processed.columns if "price" in col.lower()]
            numeric_cols = df_processed.select_dtypes(
                include=["float64", "int64"]
            ).columns
            numeric_cols = [col for col in numeric_cols if col not in price_cols]

            # Scale the data
            df_processed, scalers = scale_features(
                df_processed, price_cols, numeric_cols
            )

            # Save scalers for future use
            try:
                import pickle

                os.makedirs(output_dir, exist_ok=True)
                with open(f"{output_dir}/scalers.pkl", "wb") as f:
                    pickle.dump(scalers, f)
                logger.info(f"Saved scalers to {output_dir}/scalers.pkl")
            except Exception as e:
                logger.warning(f"Failed to save scalers: {str(e)}")

        except Exception as e:
            logger.error(f"Error during data scaling: {str(e)}")

    # Log final dataset statistics
    logger.info(f"Preprocessing completed. Final dataset shape: {df_processed.shape}")
    logger.info(f"Columns in processed data: {', '.join(df_processed.columns)}")

    return df_processed
