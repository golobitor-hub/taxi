"""FastAPI-приложение: инференс модели аномальных поездок.

Запуск локально:  uvicorn app.main:app --reload       (документация: http://127.0.0.1:8000/docs)
Пути к модели можно переопределить переменными окружения MODEL_PATH и THRESHOLD_PATH.
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app import __version__
from app.predictor import Predictor
from app.schemas import BatchRequest, BatchResponse, Health, ModelInfo, Prediction, Trip
from src.utils import PROJECT_ROOT

log = logging.getLogger("uvicorn.error")


def _paths() -> tuple[Path, Path]:
    model = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models" / "final_model.joblib"))
    threshold = Path(os.environ.get("THRESHOLD_PATH", PROJECT_ROOT / "models" / "threshold.json"))
    return model, threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path, threshold_path = _paths()
    try:
        app.state.predictor = Predictor(model_path, threshold_path)
        log.info("Модель загружена: %s (sha256 %s)", model_path, app.state.predictor.info.model_sha256)
    except (FileNotFoundError, KeyError, ValueError) as e:
        # Сервис стартует, но /health сообщит о проблеме, а /predict вернёт 503
        app.state.predictor = None
        log.error("Модель не загружена: %s", e)
    yield


app = FastAPI(
    title="Taxi anomaly API",
    version=__version__,
    description="Предсказание аномально долгих (> 1 часа) поездок такси Нью-Йорка в момент посадки.",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return predictor


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=Health, tags=["service"])
def health(request: Request):
    loaded = request.app.state.predictor is not None
    body = Health(status="ok" if loaded else "model_not_loaded", model_loaded=loaded, version=__version__)
    return JSONResponse(body.model_dump(), status_code=200 if loaded else 503)


@app.get("/model/info", response_model=ModelInfo, tags=["service"])
def model_info(request: Request) -> ModelInfo:
    return get_predictor(request).info


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(trip: Trip, request: Request) -> Prediction:
    return get_predictor(request).predict([trip])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    preds = get_predictor(request).predict(batch.trips)
    return BatchResponse(predictions=preds, n_anomalies=sum(p.is_anomaly for p in preds))
