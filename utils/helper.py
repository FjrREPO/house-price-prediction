import re
import pandas as pd
from datetime import datetime, timedelta


def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


def convert_to_numeric(value: str) -> float:
    try:
        value_numeric = re.sub(r"Rp\s?", "", value)

        if "Miliar" in value_numeric:
            value_numeric = (
                float(re.sub(r"\s?Miliar", "", value_numeric).replace(",", ".")) * 1e9
            )
        elif "Juta" in value_numeric:
            value_numeric = (
                float(re.sub(r"\s?Juta", "", value_numeric).replace(",", ".")) * 1e6
            )
        else:
            return None
        return value_numeric
    except ValueError:
        return None


def preprocess_updated(updated):
    today = datetime.today()

    if "minggu" in updated:
        weeks_ago = int(re.search(r"(\d+)", updated).group(1))
        return today - timedelta(weeks=weeks_ago)
    elif "bulan" in updated:
        months_ago = int(re.search(r"(\d+)", updated).group(1))
        return today - timedelta(days=months_ago * 30)
    elif "hari" in updated:
        days_ago = int(re.search(r"(\d+)", updated).group(1))
        return today - timedelta(days=days_ago)
    elif "jam" in updated:
        hours_ago = int(re.search(r"(\d+)", updated).group(1))
        return today - timedelta(hours=hours_ago)


def preprocess_area(area_str):
    if isinstance(area_str, str):
        match = re.search(r"(\d+(\.\d+)?)\s*m²", area_str)
        if match:
            return float(match.group(1))
    return None


def preprocess_location(location):
    if pd.isna(location):
        return pd.Series([None, None])
    split_location = location.split(",")
    kecamatan = split_location[0].strip() if len(split_location) > 0 else None
    kabupaten_kota = split_location[1].strip() if len(split_location) > 1 else None
    return pd.Series([kecamatan, kabupaten_kota])


def preprocess_data(df):

    df_processed = df.copy()

    df_processed["price"] = df_processed["price"].apply(convert_to_numeric)
    df_processed["LT"] = df_processed["LT"].apply(preprocess_area)
    df_processed["LB"] = df_processed["LB"].apply(preprocess_area)
    df_processed["updated"] = df_processed["updated"].apply(preprocess_updated)
    df_processed[["kecamatan", "kabupaten_kota"]] = df_processed["location"].apply(
        preprocess_location
    )

    df_processed["updated"] = pd.to_datetime(df_processed["updated"])

    df_processed["price"] = df_processed["price"].fillna(df_processed["price"].median())
    df_processed["LT"] = df_processed["LT"].fillna(df_processed["LT"].median())
    df_processed["LB"] = df_processed["LB"].fillna(df_processed["LB"].median())

    df_processed = df_processed.dropna(subset=["updated"])

    df_processed = df_processed.dropna(subset=["LT", "LB", "price"])

    df_processed = df_processed[
        ~df_processed["title"].str.contains("hotel|kost|Kos", case=False, na=False)
    ]

    df_processed.drop(columns=["carport"], inplace=True)

    df_processed.dropna(subset=["bedroom", "bathroom"], inplace=True)
    df_processed.dropna(subset=["LT", "LB"], inplace=True)

    sixty_days_ago = datetime.today() - timedelta(days=60)
    df_processed = df_processed[df_processed["updated"] >= sixty_days_ago]

    df_processed = df_processed.drop_duplicates(subset="title", keep="first")

    return df_processed
