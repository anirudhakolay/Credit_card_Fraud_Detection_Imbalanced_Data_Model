"""
Data Preprocessing and Feature Engineering Module
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees) using the Haversine formula.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c  # Earth radius in kilometers
    return km


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering:
    - Extracts temporal features (hour, day of week, night indicator)
    - Computes cardholder age from date of birth (dob)
    - Calculates geographical distance (km) between cardholder and merchant
    - Drops non-generalizable identifier columns
    """
    df = df.copy()

    # 1. Parse timestamps
    if "trans_date_trans_time" in df.columns:
        trans_dt = pd.to_datetime(df["trans_date_trans_time"])
        df["trans_hour"] = trans_dt.dt.hour
        df["trans_day_of_week"] = trans_dt.dt.dayofweek
        # Fraud spikes heavily late at night (10 PM to 4 AM)
        df["is_night"] = df["trans_hour"].apply(lambda h: 1 if (h >= 22 or h <= 4) else 0)

    # 2. Extract cardholder age
    if "dob" in df.columns:
        dob_dt = pd.to_datetime(df["dob"])
        # Reference year 2020 based on dataset transaction period
        df["age"] = 2020 - dob_dt.dt.year

    # 3. Calculate distance between cardholder residence and merchant location
    coord_cols = ["lat", "long", "merch_lat", "merch_long"]
    if all(col in df.columns for col in coord_cols):
        df["distance_km"] = calculate_haversine_distance(
            df["lat"], df["long"], df["merch_lat"], df["merch_long"]
        )

    # 4. Amount log transformation (useful for linear models to handle extreme skew)
    if "amt" in df.columns:
        df["log_amt"] = np.log1p(df["amt"])

    # 5. Drop high-cardinality and unique identifier columns that cause data leakage
    drop_cols = [
        "Unnamed: 0",
        "trans_date_trans_time",
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "state",
        "zip",
        "lat",
        "long",
        "job",
        "dob",
        "trans_num",
        "unix_time",
        "merch_lat",
        "merch_long",
        "merchant",
    ]
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


def prepare_data(df: pd.DataFrame, target_col: str = "is_fraud", label_encoders: dict = None):
    """
    Full preparation pipeline:
    Extracts features, encodes categoricals, and splits into X and y.
    """
    # Feature extraction
    processed_df = extract_features(df)

    # Separate target if present
    y = None
    if target_col in processed_df.columns:
        y = processed_df[target_col]
        X = processed_df.drop(columns=[target_col])
    else:
        X = processed_df

    # Encode categorical columns
    categorical_cols = ["category", "gender"]
    if label_encoders is None:
        label_encoders = {}
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
    else:
        for col in categorical_cols:
            if col in X.columns and col in label_encoders:
                le = label_encoders[col]
                # Handle unseen categories gracefully
                X[col] = X[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else -1
                )

    return X, y, label_encoders
