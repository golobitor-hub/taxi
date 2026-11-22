"""Тесты данных: контракт Great Expectations и quality gate из ЛР2 (src/data_validation.py).

В CI проверки идут на реальной выборке tests/fixtures/train_sample.csv. Тесты с маркером `data`
проверяют полный датасет и запускаются только там, где он есть (после `dvc pull` / `dvc repro`).
"""
import pandas as pd
import pytest

from src.clean import clean_taxi_data
from src.data_validation import build_expectations, quality_gate, validate_dataframe
from src.utils import PROJECT_ROOT


def failed_rules(report: pd.DataFrame) -> set:
    failed = report[~report["success"]]
    return set(zip(failed["expectation"], failed["column"], strict=True))


def set_first(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Копия df, где в первой строке column заменён на value."""
    out = df.copy()
    out.loc[out.index[0], column] = value
    return out


def test_contract_covers_every_column(vcfg):
    rules = build_expectations(vcfg)
    # схема + уникальность id + 2 множества + 2 диапазона + 4 координаты + not-null на каждую колонку
    assert len(rules) == 10 + len(vcfg["expected_columns"])
    not_null = {r.column for r in rules if type(r).__name__ == "ExpectColumnValuesToNotBeNull"}
    assert not_null == set(vcfg["expected_columns"])


def test_real_sample_passes_contract(raw_sample, vcfg):
    report = validate_dataframe(raw_sample, "sample", vcfg)
    assert report["success"].all(), failed_rules(report)
    assert (report["element_count"] == len(raw_sample)).all()


def test_cleaned_sample_passes_quality_gate(raw_sample, vcfg):
    clean = clean_taxi_data(raw_sample, vcfg)
    quality_gate(validate_dataframe(clean, "clean_sample", vcfg), "CLEAN SAMPLE")  # не бросает исключение


CORRUPTIONS = {
    "duplicate_id": (lambda df: df.assign(id=df["id"].where(df.index != df.index[1], df["id"].iloc[0])),
                     ("expect_column_values_to_be_unique", "id")),
    "bad_vendor": (lambda df: set_first(df, "vendor_id", 7),
                   ("expect_column_values_to_be_in_set", "vendor_id")),
    "bad_flag": (lambda df: df.assign(store_and_fwd_flag="?"),
                 ("expect_column_values_to_be_in_set", "store_and_fwd_flag")),
    "too_many_passengers": (lambda df: set_first(df, "passenger_count", 9),
                            ("expect_column_values_to_be_between", "passenger_count")),
    "negative_duration": (lambda df: set_first(df, "trip_duration", -5),
                          ("expect_column_values_to_be_between", "trip_duration")),
    "outside_nyc": (lambda df: set_first(df, "pickup_longitude", 0.0),
                    ("expect_column_values_to_be_between", "pickup_longitude")),
    "missing_value": (lambda df: set_first(df, "dropoff_latitude", None),
                      ("expect_column_values_to_not_be_null", "dropoff_latitude")),
    "extra_column": (lambda df: df.assign(tip_amount=1.0), ("expect_table_columns_to_match_set", "TABLE")),
}


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_contract_detects_corruption(raw_sample, vcfg, name):
    corrupt, expected_failure = CORRUPTIONS[name]
    report = validate_dataframe(corrupt(raw_sample), name, vcfg)
    assert expected_failure in failed_rules(report)


def test_quality_gate_blocks_bad_data(raw_sample, vcfg):
    bad = set_first(raw_sample, "passenger_count", 0)
    report = validate_dataframe(bad, "bad", vcfg)
    with pytest.raises(ValueError, match="DATA QUALITY GATE FAILED"):
        quality_gate(report, "BAD SAMPLE")


def test_cleaning_repairs_corrupted_sample(raw_sample, vcfg):
    """Очистка + контракт работают вместе: испорченные строки удаляются, и выборка снова проходит gate."""
    bad = raw_sample.copy()
    bad.loc[bad.index[:3], "passenger_count"] = 0
    bad.loc[bad.index[3:5], "trip_duration"] = 100_000
    bad = pd.concat([bad, bad.iloc[[10]]], ignore_index=True)
    clean = clean_taxi_data(bad, vcfg)
    assert len(clean) == len(raw_sample) - 5
    quality_gate(validate_dataframe(clean, "repaired", vcfg), "REPAIRED")


# --- полный датасет (только локально) -------------------------------------------------------------

CLEAN_TRAIN = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"


@pytest.mark.data
@pytest.mark.skipif(not CLEAN_TRAIN.exists(), reason="нет data/processed/train_clean.csv (нужен dvc pull)")
def test_full_clean_dataset_passes_contract(vcfg):
    clean = pd.read_csv(CLEAN_TRAIN, parse_dates=["pickup_datetime", "dropoff_datetime"])
    assert len(clean) > 1_000_000
    quality_gate(validate_dataframe(clean, "full_clean", vcfg), "FULL CLEAN TRAIN")
