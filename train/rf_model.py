import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold


def predict_price_rf(
    model, bedroom_val, bathroom_val, LT_val, LB_val, fallback_value=None
):
    input_data = pd.DataFrame(
        {
            "bedroom": [bedroom_val],
            "bathroom": [bathroom_val],
            "LT": [LT_val],
            "LB": [LB_val],
        }
    )

    try:
        prediction = model.predict(input_data)[0]
        return prediction
    except Exception as e:
        print(f"RF prediction error: {e}")
        return fallback_value if fallback_value is not None else 0


def train_base_rf_model(X, y):
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    base_model.fit(X, y)
    return base_model


def evaluate_rf_model(model, X, y, cross_validate=False, cv=5):
    if cross_validate:
        # Perform cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        mape_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_percentage_error')
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
        
        predictions = model.predict(X)
        mape = np.mean(mape_scores)
        
        # Return comprehensive evaluation metrics
        metrics = {
            'mape': mape,
            'mape_cv': mape_scores,
            'r2': np.mean(r2_scores),
            'r2_cv': r2_scores,
            'mae': np.mean(mae_scores),
            'mae_cv': mae_scores,
            'rmse': np.sqrt(mean_squared_error(y, predictions))
        }
        return mape, predictions, metrics
    else:
        predictions = []
        for i in range(len(X)):
            prediction = predict_price_rf(
                model,
                X.iloc[i]["bedroom"],
                X.iloc[i]["bathroom"],
                X.iloc[i]["LT"],
                X.iloc[i]["LB"],
                fallback_value=y.iloc[i],
            )
            predictions.append(prediction)

        mape = mean_absolute_percentage_error(y, predictions)
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        metrics = {
            'mape': mape,
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }
        
        return mape, predictions, metrics


def train_optimized_rf_model(X, y, params):
    n_estimators = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])
    max_features = params[4] if len(params) > 4 else 'sqrt'
    bootstrap = params[5] > 0.5 if len(params) > 5 else True
    
    # Convert string parameter if needed
    if isinstance(max_features, float):
        if max_features < 0.33:
            max_features = 'sqrt'
        elif max_features < 0.66:
            max_features = 'log2'
        else:
            max_features = None
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    model.fit(X, y)
    return model
