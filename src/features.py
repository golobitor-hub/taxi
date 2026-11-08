import argparse

import numpy as np
import pandas as pd

from src.utils import PROJECT_ROOT, ensure_parent, load_params

def manhattan_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dy = R * np.abs(np.radians(lat2 - lat1))
    mean_lat = (np.radians(lat1) + np.radians(lat2)) / 2
    dx = R * np.cos(mean_lat) * np.abs(np.radians(lon2 - lon1))
    return dx + dy

def bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(y, x))

def make_features(df: pd.DataFrame, destination_known: bool = True) -> pd.DataFrame:
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    X = pd.DataFrame(index=df.index)
    X["vendor_id"] = df["vendor_id"].astype("float32")
    X["passenger_count"] = df["passenger_count"].astype("float32")
    X["pickup_longitude"] = df["pickup_longitude"].astype("float32")
    X["pickup_latitude"] = df["pickup_latitude"].astype("float32")
    X["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"N": 0, "Y": 1}).fillna(-1).astype("int8")

    X["pickup_hour"] = dt.dt.hour.astype("float32")
    X["pickup_dow"] = dt.dt.dayofweek.astype("float32")
    X["pickup_month"] = dt.dt.month.astype("float32")
    X["pickup_day"] = dt.dt.day.astype("float32")
    X["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    X["is_rush_hour"] = ((dt.dt.hour.between(7, 9) | dt.dt.hour.between(16, 19))
                         & (dt.dt.dayofweek < 5)).astype("int8")
    X["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24).astype("float32")
    X["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24).astype("float32")
    X["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7).astype("float32")
    X["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7).astype("float32")

    if destination_known:
        X["dropoff_longitude"] = df["dropoff_longitude"].astype("float32")
        X["dropoff_latitude"] = df["dropoff_latitude"].astype("float32")
        X["distance_km"] = manhattan_distance_km(
            df["pickup_latitude"], df["pickup_longitude"],
            df["dropoff_latitude"], df["dropoff_longitude"]).astype("float32")
        X["longitude_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs().astype("float32")
        X["latitude_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs().astype("float32")
        X["bearing"] = bearing(df["pickup_longitude"], df["pickup_latitude"],
                               df["dropoff_longitude"], df["dropoff_latitude"]).astype("float32")
    return X.replace([np.inf, -np.inf], np.nan)

def main(params_path: str) -> None:
    params = load_params(params_path)
    d, split = params["data"], params["split"]
    dest = params["features"]["destination_known"]
    out_dir = PROJECT_ROOT / d["features_dir"]

    train = pd.read_csv(PROJECT_ROOT / d["clean_train"], parse_dates=["pickup_datetime", "dropoff_datetime"])
    train["is_anomaly"] = (train["trip_duration"] > params["target"]["anomaly_threshold_sec"]).astype("int8")

    X = make_features(train, dest)
    assert not {"trip_duration", "dropoff_datetime", "id"} & set(X.columns), "утечка целевой переменной"
    y = train["is_anomaly"]

    order = np.argsort(train["pickup_datetime"].to_numpy(), kind="stable")
    cut_train = int(len(order) * split["train_frac"])
    cut_cal = int(len(order) * split["calibration_frac"])
    parts = {"train": order[:cut_train], "cal": order[cut_train:cut_cal], "holdout": order[cut_cal:]}

    meta_cols = ["id", "pickup_datetime", "dropoff_datetime", "trip_duration", "is_anomaly"]
    for name, idx in parts.items():
        X.iloc[idx].to_parquet(ensure_parent(out_dir / f"X_{name}.parquet"))
        train.iloc[idx][meta_cols].to_parquet(out_dir / f"meta_{name}.parquet")
        print(f"{name:8s} {len(idx):>8d}  positive={y.iloc[idx].mean():.4%}")

    test = pd.read_csv(PROJECT_ROOT / d["raw_test"], parse_dates=["pickup_datetime"])
    make_features(test, dest).to_parquet(out_dir / "X_test.parquet")
    test.to_parquet(out_dir / "test_raw.parquet")
    print("Признаков:", X.shape[1], "| test:", len(test))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    main(ap.parse_args().params)
