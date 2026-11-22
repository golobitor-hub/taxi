"""Юнит-тесты очистки данных (src/clean.py)."""
import numpy as np
import pandas as pd
import pytest

from src.clean import clean_taxi_data, data_profile


def test_valid_rows_are_kept(trips, vcfg):
    df = trips({}, {"passenger_count": 6}, {"trip_duration": 1})
    assert len(clean_taxi_data(df, vcfg)) == 3


@pytest.mark.parametrize("bad", [
    {"vendor_id": 3},
    {"store_and_fwd_flag": "X"},
    {"passenger_count": 0},
    {"passenger_count": 7},
    {"trip_duration": 0},
    {"trip_duration": 86401},
    {"pickup_longitude": -75.0},
    {"pickup_latitude": 42.0},
    {"dropoff_longitude": -73.0},
    {"dropoff_latitude": 40.0},
    {"passenger_count": np.nan},
], ids=lambda d: next(iter(d.items())).__repr__())
def test_invalid_row_is_removed(trips, vcfg, bad):
    out = clean_taxi_data(trips({}, bad), vcfg)
    assert len(out) == 1
    assert out.iloc[0]["id"] == "id0000000"  # осталась только корректная строка


def test_boundaries_are_inclusive(trips, vcfg):
    df = trips(
        {"passenger_count": vcfg["passenger_min"], "trip_duration": vcfg["duration_min_sec"]},
        {"passenger_count": vcfg["passenger_max"], "trip_duration": vcfg["duration_max_sec"]},
        {"pickup_longitude": vcfg["lon_min"], "pickup_latitude": vcfg["lat_max"]},
    )
    assert len(clean_taxi_data(df, vcfg)) == 3


def test_full_duplicates_and_duplicate_ids_removed(trips, vcfg):
    df = trips({}, {})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)       # полный дубликат строки
    dup_id = df.iloc[[1]].assign(passenger_count=2)              # тот же id, другие значения
    df = pd.concat([df, dup_id], ignore_index=True)
    out = clean_taxi_data(df, vcfg)
    assert len(out) == 2
    assert out["id"].is_unique
    assert out.loc[out["id"] == "id0000001", "passenger_count"].item() == 1  # оставлена первая запись


def test_input_not_mutated(trips, vcfg):
    df = trips({}, {"passenger_count": 9})
    before = df.copy()
    clean_taxi_data(df, vcfg)
    pd.testing.assert_frame_equal(df, before)


def test_idempotent(raw_sample, vcfg):
    once = clean_taxi_data(raw_sample, vcfg)
    twice = clean_taxi_data(once, vcfg)
    pd.testing.assert_frame_equal(once, twice)


def test_columns_preserved(raw_sample, vcfg):
    assert list(clean_taxi_data(raw_sample, vcfg).columns) == list(raw_sample.columns)


def test_data_profile_counts(trips):
    df = trips({}, {"passenger_count": 4, "trip_duration": 100})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.loc[1, "vendor_id"] = np.nan
    prof = data_profile(df, "test")
    assert prof["rows"] == 3
    assert prof["missing_cells"] == 1
    assert prof["duplicate_rows"] == 1
    assert prof["duplicate_ids"] == 1
    assert prof["passenger_max"] == 4
    assert prof["duration_min_sec"] == 100
