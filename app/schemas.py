"""Схемы запросов и ответов (Pydantic).

Границы допустимых значений берутся из params.yaml (раздел validation) — это тот же контракт данных,
по которому в ЛР2 проверяется обучающая выборка. Поэтому модель не получит на вход того, чего не видела
при обучении: поездку вне Нью-Йорка, 9 пассажиров или неизвестного перевозчика.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils import load_params

_V = load_params()["validation"]
MAX_BATCH = 1000

EXAMPLE_TRIP = {
    "id": "id2875421",
    "vendor_id": 2,
    "pickup_datetime": "2016-03-14T17:24:55",
    "passenger_count": 1,
    "pickup_longitude": -73.982155,
    "pickup_latitude": 40.767937,
    "dropoff_longitude": -73.964630,
    "dropoff_latitude": 40.765602,
    "store_and_fwd_flag": "N",
}


class Trip(BaseModel):
    """Поездка в момент посадки пассажира. Длительности и времени высадки нет: их и предсказываем."""

    # extra="forbid": лишние поля (например, trip_duration) — ошибка, а не тихое игнорирование
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_TRIP]})

    id: str | None = Field(None, max_length=64, description="Идентификатор поездки (возвращается в ответе)")
    vendor_id: Literal[1, 2] = Field(description="Перевозчик")
    pickup_datetime: datetime = Field(description="Время посадки, местное время Нью-Йорка без часового пояса")
    passenger_count: int = Field(ge=_V["passenger_min"], le=_V["passenger_max"])
    pickup_longitude: float = Field(ge=_V["lon_min"], le=_V["lon_max"])
    pickup_latitude: float = Field(ge=_V["lat_min"], le=_V["lat_max"])
    dropoff_longitude: float = Field(ge=_V["lon_min"], le=_V["lon_max"])
    dropoff_latitude: float = Field(ge=_V["lat_min"], le=_V["lat_max"])
    store_and_fwd_flag: Literal["N", "Y"] = Field("N", description="Запись сохранялась в памяти машины")

    @field_validator("pickup_datetime")
    @classmethod
    def naive_local_time(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            raise ValueError("укажите местное время Нью-Йорка без часового пояса (как в обучающих данных)")
        return v


class Prediction(BaseModel):
    id: str | None
    is_anomaly: bool = Field(description="Поездка будет дольше порога аномалии")
    anomaly_probability: float = Field(ge=0, le=1)
    threshold: float = Field(description="Порог вероятности, подобранный на calibration (recall ≥ target_recall)")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trips: list[Trip] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    n_anomalies: int


class Health(BaseModel):
    status: Literal["ok", "model_not_loaded"]
    model_loaded: bool
    version: str


class ModelInfo(BaseModel):
    model_type: str
    model_sha256: str
    n_iter: int
    n_features: int
    features: list[str]
    threshold: float
    target_recall: float
    anomaly_threshold_sec: int
