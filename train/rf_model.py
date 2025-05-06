import logging
import numpy as np
import pandas as pd
import traceback
from typing import Tuple, List, Dict, Optional, Union, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from sklearn.model_selection import KFold
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

logging.basicConfig(
    level=logging.INFO, 
    format=f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rf_model")


def train_base_rf_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, float]:
    try:
        logger.info(f"{Fore.GREEN}Training base Random Forest model... [n_estimators=100]{Style.RESET_ALL}")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X, y)
        
        feature_importances = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        
        top_features = ', '.join(feature_importances.index[:5].tolist())
        logger.info(f"{Fore.GREEN}Base RF model trained successfully - Trees: {model.n_estimators}, Top features: {top_features}{Style.RESET_ALL}")
        
        # Evaluate model with cross-validation to get MAPE
        cv_mape, _, _ = evaluate_rf_model(model, X, y, cross_validate=True)
        
        return model, cv_mape

    except Exception as e:
        logger.error(f"{Fore.RED}Error training base Random Forest model: {str(e)}{Style.RESET_ALL}")
        raise RuntimeError(f"Failed to train base RF model: {str(e)}")


def evaluate_rf_model(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    cross_validate: bool = False,
    cv: int = 5,
) -> Tuple[float, List[float], Dict[str, float]]:
    try:
        if cross_validate:
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)

            mape_scores = []
            r2_scores = []
            mae_scores = []
            rmse_scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                fold_model = RandomForestRegressor(**model.get_params())
                fold_model.fit(X_train, y_train)

                predictions = fold_model.predict(X_val)

                mape = mean_absolute_percentage_error(y_val, predictions)
                r2 = r2_score(y_val, predictions)
                mae = mean_absolute_error(y_val, predictions)
                rmse = np.sqrt(mean_squared_error(y_val, predictions))

                mape_scores.append(mape)
                r2_scores.append(r2)
                mae_scores.append(mae)
                rmse_scores.append(rmse)

            avg_mape = np.mean(mape_scores)
            avg_r2 = np.mean(r2_scores)
            avg_mae = np.mean(mae_scores)
            avg_rmse = np.mean(rmse_scores)

            predictions = model.predict(X)

            metrics = {
                "mape": avg_mape,
                "r2": avg_r2,
                "mae": avg_mae,
                "rmse": avg_rmse,
                "mape_cv": mape_scores,
                "r2_cv": r2_scores,
                "mae_cv": mae_scores,
                "rmse_cv": rmse_scores,
            }

            logger.info(
                f"{Fore.YELLOW}Cross-validation results - MAPE: {avg_mape*100:.2f}%, R²: {avg_r2:.4f}, RMSE: {avg_rmse:.2f}{Style.RESET_ALL}"
            )
            return avg_mape, predictions.tolist(), metrics

        else:
            predictions = model.predict(X)

            mape = mean_absolute_percentage_error(y, predictions)
            r2 = r2_score(y, predictions)
            mae = mean_absolute_error(y, predictions)
            rmse = np.sqrt(mean_squared_error(y, predictions))

            metrics = {"mape": mape, "r2": r2, "mae": mae, "rmse": rmse}

            logger.info(
                f"{Fore.YELLOW}Evaluation results - MAPE: {mape*100:.2f}%, R²: {r2:.4f}, RMSE: {rmse:.2f}{Style.RESET_ALL}"
            )
            return mape, predictions.tolist(), metrics

    except Exception as e:
        logger.error(f"{Fore.RED}Error evaluating Random Forest model: {str(e)}{Style.RESET_ALL}")

        return 1.0, [], {"mape": 1.0, "r2": 0.0, "mae": 1e10, "rmse": 1e10}


def train_optimized_rf_model(
    X: pd.DataFrame, y: pd.Series, params: Union[Tuple, List, Dict[str, Any]]
) -> Tuple[RandomForestRegressor, float]:
    """
    Train an optimized Random Forest model using the hyperparameters found by the GA optimizer.
    
    Args:
        X: Feature dataframe
        y: Target series
        params: Parameters from GA optimization as a tuple, list, or dictionary
               If dictionary, must contain keys: n_estimators, max_depth, min_samples_split,
               min_samples_leaf, and optionally max_features and bootstrap
        
    Returns:
        Tuple of (trained model, cv_mape)
    """
    try:
        # Convert dictionary to tuple if necessary
        if isinstance(params, dict):
            logger.info(f"Converting dictionary params to tuple: {params}")
            param_tuple = (
                params.get('n_estimators', 100),
                params.get('max_depth', None),
                params.get('min_samples_split', 2),
                params.get('min_samples_leaf', 1),
                params.get('max_features', 'sqrt'),
                params.get('bootstrap', True)
            )
            params = param_tuple
            logger.info(f"Converted to tuple: {params}")
        
        # Validate params data type and structure
        if not isinstance(params, (list, tuple)):
            raise TypeError(f"Expected params to be a list, tuple, or dict, got {type(params).__name__}: {params}")
        
        if len(params) < 4:
            raise ValueError(f"Expected at least 4 parameters, got {len(params)}: {params}")
            
        # Parameter validation and conversion
        try:
            n_estimators = int(params[0])
            if n_estimators <= 0:
                raise ValueError(f"n_estimators must be positive, got {n_estimators}")
                
            max_depth = int(params[1]) if params[1] and params[1] > 0 else None
            
            min_samples_split = int(params[2])
            if min_samples_split < 2:
                raise ValueError(f"min_samples_split must be at least 2, got {min_samples_split}")
                
            min_samples_leaf = int(params[3])
            if min_samples_leaf < 1:
                raise ValueError(f"min_samples_leaf must be at least 1, got {min_samples_leaf}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error converting basic parameters: {e}") from e

        # Handle max_features parameter
        max_features = "sqrt"  # default
        if len(params) > 4:
            mf_val = params[4]
            logger.info(f"Processing max_features parameter with value: {mf_val} (type: {type(mf_val).__name__})")
            
            if isinstance(mf_val, str):
                if mf_val in ["auto", "sqrt", "log2", None]:
                    max_features = mf_val
                else:
                    raise ValueError(f"Invalid string value for max_features: {mf_val}")
                    
            elif isinstance(mf_val, (int, float)):
                # If max_features is a float between 0 and 1, use it directly
                if 0.0 < mf_val <= 1.0:
                    # Explicitly convert to float to ensure sklearn accepts it
                    max_features = float(mf_val)
                    logger.info(f"Using max_features as a float value: {max_features}")
                # Otherwise apply range mapping
                elif mf_val < 0.33:
                    max_features = "sqrt"
                elif mf_val < 0.66:
                    max_features = "log2"
                else:
                    max_features = None
            else:
                logger.warning(f"Unexpected type for max_features: {type(mf_val).__name__}, using default 'sqrt'")

        # Handle bootstrap parameter
        bootstrap = True  # default
        if len(params) > 5:
            bootstrap_val = params[5]
            if isinstance(bootstrap_val, bool):
                bootstrap = bootstrap_val
            elif isinstance(bootstrap_val, (int, float)):
                bootstrap = bootstrap_val > 0.5
            else:
                logger.warning(f"Unexpected type for bootstrap: {type(bootstrap_val).__name__}, using default True")

        # Log all parameters that will be used
        param_str = (f"n_estimators={n_estimators}, max_depth={max_depth}, "
                    f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                    f"max_features={max_features} (type: {type(max_features).__name__}), bootstrap={bootstrap}")
        logger.info(f"{Fore.GREEN}Training optimized RF model with parameters: {param_str}{Style.RESET_ALL}")

        # Create and train the model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1,
        )

        logger.info(f"Fitting model with X shape: {X.shape}, y shape: {y.shape}")
        try:
            model.fit(X, y)
            logger.info(f"{Fore.GREEN}Optimized Random Forest model trained successfully{Style.RESET_ALL}")
        except Exception as fit_error:
            logger.error(f"{Fore.RED}Error during model.fit(): {str(fit_error)}{Style.RESET_ALL}")
            raise
        
        # Evaluate model with cross-validation to get MAPE
        cv_mape, _, _ = evaluate_rf_model(model, X, y, cross_validate=True)

        return model, cv_mape

    except Exception as e:
        # Include traceback in the error message for better debugging
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"{Fore.RED}Error training optimized Random Forest model: {error_msg}{Style.RESET_ALL}")
        logger.error(f"{Fore.RED}Input params type: {type(params).__name__}, value: {params}{Style.RESET_ALL}")

        logger.warning(f"{Fore.YELLOW}Falling back to default Random Forest model{Style.RESET_ALL}")
        return train_base_rf_model(X, y)


def predict_price_rf(
    model: RandomForestRegressor,
    bedroom: Union[int, float],
    bathroom: Union[int, float],
    lt: Union[int, float],
    lb: Union[int, float],
    carport: Optional[Union[int, float]] = None,
    kecamatan_encoded: Optional[Union[int, float]] = None,
    fallback_value: Optional[float] = None,
) -> float:
    try:

        features = {
            "bedroom": [bedroom],
            "bathroom": [bathroom],
            "LT": [lt],
            "LB": [lb],
        }

        if carport is not None:
            features["carport"] = [carport]

        if kecamatan_encoded is not None:
            features["kecamatan_encoded"] = [kecamatan_encoded]

        if "price_per_m2_land" in model.feature_names_in_:

            pass

        if "building_efficiency" in model.feature_names_in_:
            features["building_efficiency"] = [lb / lt if lt > 0 else 0]

        X_pred = pd.DataFrame(features)

        required_columns = model.feature_names_in_
        missing_columns = set(required_columns) - set(X_pred.columns)

        if missing_columns:
            logger.warning(f"{Fore.YELLOW}Missing columns for prediction: {missing_columns}{Style.RESET_ALL}")
            for col in missing_columns:
                X_pred[col] = 0

        X_pred = X_pred[required_columns]

        prediction = model.predict(X_pred)[0]

        return prediction

    except Exception as e:
        logger.error(f"{Fore.RED}Error predicting price with RF model: {str(e)}{Style.RESET_ALL}")

        return fallback_value if fallback_value is not None else 5000000000
