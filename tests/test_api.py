"""Тесты REST API (app/): валидация входных данных, инференс, служебные эндпоинты.

Настоящая модель хранится в DVC и в CI недоступна, поэтому здесь на реальной выборке
tests/fixtures/train_sample.csv обучается маленькая модель того же типа и подставляется через MODEL_PATH.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier

from app.main import app
from app.schemas import EXAMPLE_TRIP, MAX_BATCH
from src.features import make_features

THRESHOLD = 0.5


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    sample = pd.read_csv("tests/fixtures/train_sample.csv", parse_dates=["pickup_datetime", "dropoff_datetime"])
    X = make_features(sample)
    y = (sample["trip_duration"] > sample["trip_duration"].median()).astype(int)  # сбалансированная цель
    model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, y)
    d = tmp_path_factory.mktemp("model")
    joblib.dump(model, d / "model.joblib")
    (d / "threshold.json").write_text(json.dumps({"threshold": THRESHOLD, "target_recall": 0.9, "n_iter": 20}))
    return d / "model.joblib", d / "threshold.json", model


@pytest.fixture(scope="module")
def client(model_files):
    model_path, thr_path, _ = model_files
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(model_path))
    mp.setenv("THRESHOLD_PATH", str(thr_path))
    with TestClient(app) as c:  # with — чтобы выполнился lifespan (загрузка модели)
        yield c
    mp.undo()


def trip(**overrides) -> dict:
    return {**EXAMPLE_TRIP, **overrides}


# --- служебные эндпоинты ---------------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_model_info(client, model_files):
    info = client.get("/model/info").json()
    assert info["model_type"] == "HistGradientBoostingClassifier"
    assert info["threshold"] == THRESHOLD
    assert info["features"] == list(model_files[2].feature_names_in_)
    assert len(info["model_sha256"]) == 12


def test_openapi_schema_has_constraints(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["Trip"]["properties"]
    assert schema["passenger_count"]["minimum"] == 1 and schema["passenger_count"]["maximum"] == 6
    assert schema["vendor_id"]["enum"] == [1, 2]


# --- инференс --------------------------------------------------------------------------------------

def test_predict_valid_trip(client):
    r = client.post("/predict", json=EXAMPLE_TRIP)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == EXAMPLE_TRIP["id"]
    assert 0 <= body["anomaly_probability"] <= 1
    assert body["is_anomaly"] == (body["anomaly_probability"] >= THRESHOLD)


def test_prediction_matches_offline_model(client, model_files):
    """API должен давать ту же вероятность, что и модель на признаках из make_features (нет skew)."""
    model = model_files[2]
    t = trip(pickup_datetime="2016-06-30T23:10:00", passenger_count=3, dropoff_longitude=-73.79,
             dropoff_latitude=40.64)  # поездка в аэропорт JFK
    api_p = client.post("/predict", json=t).json()["anomaly_probability"]
    df = pd.DataFrame([{**t, "pickup_datetime": pd.Timestamp(t["pickup_datetime"])}])
    offline_p = model.predict_proba(make_features(df)[model.feature_names_in_])[0, 1]
    assert api_p == pytest.approx(offline_p, abs=1e-9)


def test_optional_fields_have_defaults(client):
    t = {k: v for k, v in EXAMPLE_TRIP.items() if k not in ("id", "store_and_fwd_flag")}
    r = client.post("/predict", json=t)
    assert r.status_code == 200 and r.json()["id"] is None


def test_batch(client):
    trips = [trip(id=f"t{i}", passenger_count=i) for i in range(1, 6)]
    r = client.post("/predict/batch", json={"trips": trips})
    assert r.status_code == 200
    body = r.json()
    assert [p["id"] for p in body["predictions"]] == [f"t{i}" for i in range(1, 6)]
    assert body["n_anomalies"] == sum(p["is_anomaly"] for p in body["predictions"])


def test_batch_equals_single(client):
    trips = [trip(id="a"), trip(id="b", pickup_datetime="2016-01-02T03:04:05")]
    batch = client.post("/predict/batch", json={"trips": trips}).json()["predictions"]
    single = [client.post("/predict", json=t).json() for t in trips]
    np.testing.assert_allclose([p["anomaly_probability"] for p in batch], [p["anomaly_probability"] for p in single])


# --- валидация (Pydantic) ----------------------------------------------------------------------------

@pytest.mark.parametrize("field, value", [
    ("passenger_count", 0),
    ("passenger_count", 7),
    ("passenger_count", "two"),          # неверный тип
    ("passenger_count", 1.5),            # дробное число пассажиров
    ("vendor_id", 3),
    ("store_and_fwd_flag", "X"),
    ("pickup_longitude", -75.0),         # вне Нью-Йорка
    ("dropoff_latitude", 42.0),
    ("pickup_latitude", "north"),
    ("pickup_datetime", "вчера"),
    ("pickup_datetime", "2016-03-14T17:24:55+03:00"),  # с часовым поясом
])
def test_invalid_field_rejected(client, field, value):
    r = client.post("/predict", json=trip(**{field: value}))
    assert r.status_code == 422
    assert any(err["loc"][-1] == field for err in r.json()["detail"])


@pytest.mark.parametrize("field", ["vendor_id", "pickup_datetime", "passenger_count", "pickup_longitude"])
def test_missing_required_field(client, field):
    r = client.post("/predict", json={k: v for k, v in EXAMPLE_TRIP.items() if k != field})
    assert r.status_code == 422
    assert r.json()["detail"][0]["type"] == "missing"


@pytest.mark.parametrize("leak", ["trip_duration", "dropoff_datetime"])
def test_target_leakage_fields_forbidden(client, leak):
    """Поля, которых нет в момент посадки, запрещены: их нельзя случайно передать модели."""
    r = client.post("/predict", json=trip(**{leak: 900}))
    assert r.status_code == 422
    assert r.json()["detail"][0]["type"] == "extra_forbidden"


def test_not_json(client):
    r = client.post("/predict", content="vendor_id=1", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


@pytest.mark.parametrize("trips", [[], [EXAMPLE_TRIP] * (MAX_BATCH + 1)])
def test_batch_size_limits(client, trips):
    assert client.post("/predict/batch", json={"trips": trips}).status_code == 422


def test_batch_reports_index_of_bad_trip(client):
    r = client.post("/predict/batch", json={"trips": [EXAMPLE_TRIP, trip(passenger_count=9)]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "trips", 1, "passenger_count"]


# --- сервис без модели ----------------------------------------------------------------------------

def test_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=EXAMPLE_TRIP).status_code == 503
