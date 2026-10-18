"""Стадия prepare: валидация сырых данных -> очистка -> валидация + quality gate.

Запуск: python -m src.clean
"""
import argparse

import pandas as pd

from src.data_validation import print_validation_summary, quality_gate, validate_dataframe
from src.utils import PROJECT_ROOT, ensure_parent, load_params


def clean_taxi_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Удаляет дубликаты, пропуски и строки, нарушающие контракт данных."""
    out = df.drop_duplicates()
    out = out.drop_duplicates(subset="id", keep="first")
    out = out.dropna(subset=cfg["expected_columns"])

    out = out[out["vendor_id"].isin(cfg["vendor_ids"]) & out["store_and_fwd_flag"].isin(cfg["store_and_fwd_flags"])]
    out = out[out["passenger_count"].between(cfg["passenger_min"], cfg["passenger_max"])]
    out = out[out["trip_duration"].between(cfg["duration_min_sec"], cfg["duration_max_sec"])]
    for p in ("pickup", "dropoff"):
        out = out[out[f"{p}_longitude"].between(cfg["lon_min"], cfg["lon_max"])
                  & out[f"{p}_latitude"].between(cfg["lat_min"], cfg["lat_max"])]
    return out.copy()


def data_profile(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.Series({
        "dataset": name, "rows": len(df), "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "passenger_min": df["passenger_count"].min(), "passenger_max": df["passenger_count"].max(),
        "duration_min_sec": df["trip_duration"].min(),
        "duration_median_sec": df["trip_duration"].median(),
        "duration_max_sec": df["trip_duration"].max(),
        "pickup_date_min": df["pickup_datetime"].min(), "pickup_date_max": df["pickup_datetime"].max(),
    })


def main(params_path: str) -> None:
    params = load_params(params_path)
    cfg = params["validation"]
    d = params["data"]
    report_dir = PROJECT_ROOT / "reports" / "data_quality"
    report_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(PROJECT_ROOT / d["raw_train"], parse_dates=["pickup_datetime", "dropoff_datetime"])
    print("Размер сырого датасета:", raw.shape)

    raw_report = validate_dataframe(raw, "raw_train", cfg)
    print_validation_summary(raw_report, "РЕЗУЛЬТАТ ВАЛИДАЦИИ СЫРОГО ДАТАСЕТА")

    clean = clean_taxi_data(raw, cfg)
    removed = len(raw) - len(clean)
    print(f"Очищенный датасет: {clean.shape}, удалено {removed} строк ({removed / len(raw):.4%})")

    clean_report = validate_dataframe(clean, "clean_train", cfg)
    print_validation_summary(clean_report, "РЕЗУЛЬТАТ ВАЛИДАЦИИ ОЧИЩЕННОГО ДАТАСЕТА")
    quality_gate(clean_report, "CLEAN TRAIN")  # упадёт -> стадия DVC не завершится

    comparison = raw_report.merge(clean_report, on=["expectation", "column"], how="outer",
                                  suffixes=("_raw", "_clean"))
    profiles = pd.DataFrame([data_profile(raw, "raw"), data_profile(clean, "clean")]).set_index("dataset")

    clean.to_csv(ensure_parent(PROJECT_ROOT / d["clean_train"]), index=False)
    raw_report.to_csv(report_dir / "validation_raw.csv", index=False)
    clean_report.to_csv(report_dir / "validation_clean.csv", index=False)
    comparison.to_csv(report_dir / "validation_comparison.csv", index=False)
    profiles.T.to_csv(report_dir / "data_profiles.csv")
    print("Сохранено:", PROJECT_ROOT / d["clean_train"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    main(ap.parse_args().params)
