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


CODE_PATHS = ["src", "params.yaml", "dvc.yaml", "requirements.txt"]


def git_commit() -> str:
    """Короткий хэш коммита; суффикс -dirty, если код/конфиг в рабочей копии не закоммичены.

    dvc.lock и метрики не учитываются: их перезаписывает сам dvc repro посреди прогона.
    """
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    try:
        sha = git("rev-parse", "--short=7", "HEAD")
        return sha + ("-dirty" if git("status", "--porcelain", "--", *CODE_PATHS) else "")
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
