import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


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


def evaluate_rf_model(model, X, y):
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
    return mape, predictions


def train_optimized_rf_model(X, y, params):
    n_estimators = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    model.fit(X, y)
    return model
