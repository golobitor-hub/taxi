"""Общие фикстуры тестов.

tests/fixtures/train_sample.csv — 400 реальных строк из data/raw/train.csv (random_state=42).
Полный датасет в CI недоступен (лежит в DVC-хранилище), поэтому тесты данных в CI идут на этой
выборке, а тесты с маркером `data` запускаются на полном датасете только локально.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.utils import PROJECT_ROOT, load_params

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def params() -> dict:
    return load_params(PROJECT_ROOT / "params.yaml")


@pytest.fixture(scope="session")
def vcfg(params) -> dict:
    return params["validation"]


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "train_sample.csv", parse_dates=["pickup_datetime", "dropoff_datetime"])


def make_trip(**overrides) -> dict:
    """Одна корректная поездка по Манхэттену; поля можно переопределить."""
    row = {
        "id": "id0000001", "vendor_id": 1,
        "pickup_datetime": pd.Timestamp("2016-03-14 08:30:00"),
        "dropoff_datetime": pd.Timestamp("2016-03-14 08:45:00"),
        "passenger_count": 1,
        "pickup_longitude": -73.98, "pickup_latitude": 40.75,
        "dropoff_longitude": -73.96, "dropoff_latitude": 40.77,
        "store_and_fwd_flag": "N", "trip_duration": 900,
    }
    row.update(overrides)
    return row


@pytest.fixture
def trips():
    """Фабрика DataFrame из списка переопределений: trips({}, {"passenger_count": 9}, ...)."""
    def _make(*rows: dict) -> pd.DataFrame:
        return pd.DataFrame([make_trip(id=f"id{i:07d}", **r) for i, r in enumerate(rows)])
    return _make
