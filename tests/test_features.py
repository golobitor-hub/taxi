"""Юнит-тесты признаков, целевой переменной и временного разбиения (src/features.py)."""
import numpy as np
import pandas as pd
import pytest

from src.features import LEAKAGE_COLUMNS, bearing, make_features, make_target, manhattan_distance_km, time_split

DEST_COLUMNS = {"dropoff_longitude", "dropoff_latitude", "distance_km", "longitude_diff", "latitude_diff", "bearing"}


# --- геометрия ---------------------------------------------------------------------------------

def test_distance_zero_for_same_point():
    assert manhattan_distance_km(40.75, -73.98, 40.75, -73.98) == pytest.approx(0.0)


def test_distance_one_degree_latitude():
    # 1° широты = R * pi / 180 ≈ 111.19 км
    assert manhattan_distance_km(40.0, -74.0, 41.0, -74.0) == pytest.approx(111.19, abs=0.01)


def test_distance_is_symmetric_and_manhattan_ge_straight():
    a = manhattan_distance_km(40.70, -74.01, 40.80, -73.95)
    b = manhattan_distance_km(40.80, -73.95, 40.70, -74.01)
    assert a == pytest.approx(b)
    dy, dx = 0.10 * 111.19, 0.06 * 111.19 * np.cos(np.radians(40.75))
    assert a == pytest.approx(dx + dy, rel=1e-3)
    assert a >= np.hypot(dx, dy)


@pytest.mark.parametrize("dlon, dlat, expected", [(0, 0.1, 0), (0.1, 0, 90), (0, -0.1, 180), (-0.1, 0, -90)])
def test_bearing_cardinal_directions(dlon, dlat, expected):
    lon, lat = -73.98, 40.75
    assert bearing(lon, lat, lon + dlon, lat + dlat) == pytest.approx(expected, abs=0.1)


# --- признаки ----------------------------------------------------------------------------------

def test_no_target_leakage(raw_sample):
    X = make_features(raw_sample)
    assert not LEAKAGE_COLUMNS & set(X.columns)
    assert len(X) == len(raw_sample)
    assert X.index.equals(raw_sample.index)


def test_destination_flag_controls_columns(raw_sample):
    with_dest = set(make_features(raw_sample, destination_known=True).columns)
    without = set(make_features(raw_sample, destination_known=False).columns)
    assert DEST_COLUMNS <= with_dest
    assert not DEST_COLUMNS & without
    assert with_dest - without == DEST_COLUMNS


def test_works_on_test_schema(raw_sample):
    """В test.csv нет dropoff_datetime и trip_duration — признаки должны строиться без них."""
    test_like = raw_sample.drop(columns=["dropoff_datetime", "trip_duration"])
    pd.testing.assert_frame_equal(make_features(test_like), make_features(raw_sample))


@pytest.mark.parametrize("ts, hour, dow, weekend, rush", [
    ("2016-03-14 08:30:00", 8, 0, 0, 1),   # понедельник, утренний час пик
    ("2016-03-14 13:00:00", 13, 0, 0, 0),  # понедельник, день
    ("2016-03-18 17:15:00", 17, 4, 0, 1),  # пятница, вечерний час пик
    ("2016-03-19 08:30:00", 8, 5, 1, 0),   # суббота: часа пик нет
    ("2016-03-20 23:59:00", 23, 6, 1, 0),  # воскресенье
])
def test_time_features(trips, ts, hour, dow, weekend, rush):
    X = make_features(trips({"pickup_datetime": pd.Timestamp(ts)}))
    row = X.iloc[0]
    assert (row["pickup_hour"], row["pickup_dow"]) == (hour, dow)
    assert (row["is_weekend"], row["is_rush_hour"]) == (weekend, rush)
    assert row["hour_sin"] == pytest.approx(np.sin(2 * np.pi * hour / 24), abs=1e-6)


def test_cyclic_features_bounded(raw_sample):
    X = make_features(raw_sample)
    for c in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert X[c].between(-1, 1).all()
    np.testing.assert_allclose(X["hour_sin"] ** 2 + X["hour_cos"] ** 2, 1, atol=1e-5)


def test_store_flag_encoding(trips):
    X = make_features(trips({"store_and_fwd_flag": "N"}, {"store_and_fwd_flag": "Y"}, {"store_and_fwd_flag": None}))
    assert X["store_and_fwd_flag"].tolist() == [0, 1, -1]


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_no_infinite_values(trips):
    X = make_features(trips({"pickup_longitude": np.inf}))
    assert not np.isinf(X.select_dtypes("number").to_numpy(dtype=float)).any()


def test_dtypes_are_compact(raw_sample):
    X = make_features(raw_sample)
    assert set(X.dtypes.astype(str)) <= {"float32", "int8"}


# --- целевая переменная и разбиение --------------------------------------------------------------

def test_target_is_strictly_greater():
    y = make_target(pd.Series([3599, 3600, 3601, 86400]), 3600)
    assert y.tolist() == [0, 0, 1, 1]
    assert y.dtype == "int8"


def test_time_split_sizes_and_disjoint():
    dt = pd.Series(pd.date_range("2016-01-01", periods=100, freq="h")).sample(frac=1, random_state=0)
    parts = time_split(dt, 0.7, 0.85)
    assert [len(parts[k]) for k in ("train", "cal", "holdout")] == [70, 15, 15]
    all_idx = np.concatenate(list(parts.values()))
    assert sorted(all_idx) == list(range(100))


def test_time_split_is_chronological(raw_sample):
    dt = raw_sample["pickup_datetime"].reset_index(drop=True)
    parts = time_split(dt, 0.7, 0.85)
    assert dt.iloc[parts["train"]].max() <= dt.iloc[parts["cal"]].min()
    assert dt.iloc[parts["cal"]].max() <= dt.iloc[parts["holdout"]].min()


@pytest.mark.parametrize("train_frac, cal_frac", [(0.85, 0.7), (0, 0.5), (0.7, 1.0)])
def test_time_split_rejects_bad_fractions(train_frac, cal_frac):
    with pytest.raises(ValueError):
        time_split(pd.Series(pd.date_range("2016-01-01", periods=10)), train_frac, cal_frac)
