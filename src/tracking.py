"""Общие функции трекинга экспериментов (MLflow) для стадий baseline и evaluate."""
import os
import subprocess

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import f1_score

from src.utils import PROJECT_ROOT


def setup_mlflow(mcfg: dict) -> str:
    """Tracking URI берётся из env MLFLOW_TRACKING_URI, иначе из params.yaml.

    Относительный sqlite-путь превращаем в абсолютный от корня проекта, чтобы
    запуск из любой папки (и из dvc repro) писал в одну и ту же БД.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", mcfg["tracking_uri"])
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + (PROJECT_ROOT / uri[len("sqlite:///"):]).as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(mcfg["experiment_name"])
    return uri


def git_commit() -> str:
    """Короткий хэш коммита; суффикс -dirty, если код в рабочей копии не закоммичен."""
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--abbrev=7", "--exclude=*"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def log_model(model, name: str = "model"):
    """MLflow>=3 принимает name=, 2.x — artifact_path=.

    serialization_format="cloudpickle": по умолчанию MLflow 3 сохраняет через skops, который
    отвергает типы numpy внутри Pipeline (UntrustedTypesFoundException: numpy.dtype).
    """
    try:
        mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path=name, serialization_format="cloudpickle")


def best_f1_threshold(y_true, proba) -> float:
    """Порог, максимизирующий F1 на calibration (на holdout порог не подбирается)."""
    grid = np.unique(np.quantile(proba, np.linspace(0.01, 0.99, 197)))
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])
