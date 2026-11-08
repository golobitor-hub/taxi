"""Стадия baseline: простые базовые модели + трекинг экспериментов в MLflow.

Два эксперимента-baseline на одних и тех же признаках (data/features):
  * классификатор «аномальная поездка» (LogisticRegression) -> ROC-AUC, PR-AUC, F1
  * регрессор длительности поездки (Ridge)                   -> RMSE (в секундах и в log-шкале)
Для сравнения логируется и «тупой» бейзлайн (Dummy): любая модель обязана его побить.

В MLflow уходят: гиперпараметры, метрики (train / calibration / holdout), веса модели,
графики, params.yaml и список признаков; в теги — коммит Git и размеры выборок.

Запуск:  python -m src.baseline
Просмотр: mlflow ui --backend-store-uri sqlite:///mlflow.db   (http://127.0.0.1:5000)
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay, average_precision_score,
                             f1_score, mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import PROJECT_ROOT, load_params


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def load_split(fdir: Path, name: str):
    X = pd.read_parquet(fdir / f"X_{name}.parquet")
    meta = pd.read_parquet(fdir / f"meta_{name}.parquet")
    return X, meta["is_anomaly"].to_numpy(), meta["trip_duration"].to_numpy(dtype="float64")


def best_f1_threshold(y_true, proba) -> float:
    """Порог, максимизирующий F1 на calibration (на holdout порог не подбирается)."""
    grid = np.unique(np.quantile(proba, np.linspace(0.01, 0.99, 197)))
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def clf_metrics(prefix: str, y, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        f"{prefix}_roc_auc": roc_auc_score(y, proba),
        f"{prefix}_pr_auc": average_precision_score(y, proba),
        f"{prefix}_f1": f1_score(y, pred, zero_division=0),
        f"{prefix}_precision": precision_score(y, pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, pred, zero_division=0),
    }


def reg_metrics(prefix: str, dur_true, log_pred) -> dict:
    """RMSE считаем в двух шкалах: log1p(секунды) — на ней учим, и секунды — для интерпретации."""
    log_true = np.log1p(dur_true)
    sec_pred = np.expm1(log_pred)
    return {
        f"{prefix}_rmse_log": float(np.sqrt(np.mean((log_true - log_pred) ** 2))),
        f"{prefix}_rmse_sec": float(np.sqrt(np.mean((dur_true - sec_pred) ** 2))),
        f"{prefix}_mae_sec": float(mean_absolute_error(dur_true, sec_pred)),
        f"{prefix}_r2_log": float(r2_score(log_true, log_pred)),
    }


def log_model(model, name: str):
    """MLflow>=3 принимает name=, 2.x — artifact_path=.

    serialization_format="cloudpickle": по умолчанию MLflow 3 сохраняет через skops, который
    отвергает типы numpy внутри Pipeline (UntrustedTypesFoundException: numpy.dtype).
    """
    try:
        mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path=name, serialization_format="cloudpickle")


def run_classifier(run_name, model, params, data, thr_params, params_path, features, tags):
    (Xtr, ytr, _), (Xcal, ycal, _), (Xho, yho, _) = data
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({**tags, "task": "classification", "model_family": type(model[-1]).__name__})
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(features))
        model.fit(Xtr, ytr)

        p = {n: model.predict_proba(X)[:, 1] for n, X in [("train", Xtr), ("cal", Xcal), ("holdout", Xho)]}
        thr = best_f1_threshold(ycal, p["cal"])
        mlflow.log_param("decision_threshold", round(thr, 6))
        metrics = {}
        for n, y in [("train", ytr), ("cal", ycal), ("holdout", yho)]:
            metrics.update(clf_metrics(n, y, p[n], thr))
        mlflow.log_metrics(metrics)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        RocCurveDisplay.from_predictions(yho, p["holdout"], ax=axes[0])
        PrecisionRecallDisplay.from_predictions(yho, p["holdout"], ax=axes[1])
        fig.suptitle(f"{run_name}: holdout")
        mlflow.log_figure(fig, "holdout_curves.png")
        plt.close(fig)

        mlflow.log_artifact(str(params_path))
        mlflow.log_dict({"features": list(features)}, "features.json")
        log_model(model, "model")
        print(f"[{run_name}] holdout ROC-AUC={metrics['holdout_roc_auc']:.4f} "
              f"PR-AUC={metrics['holdout_pr_auc']:.4f} F1={metrics['holdout_f1']:.4f}")
        return metrics


def run_regressor(run_name, model, params, data, params_path, features, tags):
    (Xtr, _, dtr), (Xcal, _, dcal), (Xho, _, dho) = data
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({**tags, "task": "regression", "model_family": type(model[-1]).__name__})
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("target", "log1p(trip_duration)")
        model.fit(Xtr, np.log1p(dtr))

        metrics = {}
        for n, X, d in [("train", Xtr, dtr), ("cal", Xcal, dcal), ("holdout", Xho, dho)]:
            metrics.update(reg_metrics(n, d, model.predict(X)))
        mlflow.log_metrics(metrics)

        pred = np.expm1(model.predict(Xho))
        fig, ax = plt.subplots(figsize=(5.5, 5))
        lim = float(np.quantile(dho, 0.99))
        ax.scatter(dho, pred, s=4, alpha=0.3)
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="факт, с", ylabel="прогноз, с",
               title=f"{run_name}: holdout")
        mlflow.log_figure(fig, "holdout_pred_vs_true.png")
        plt.close(fig)

        mlflow.log_artifact(str(params_path))
        mlflow.log_dict({"features": list(features)}, "features.json")
        log_model(model, "model")
        print(f"[{run_name}] holdout RMSE={metrics['holdout_rmse_sec']:.1f} с "
              f"(log: {metrics['holdout_rmse_log']:.4f}) R2(log)={metrics['holdout_r2_log']:.4f}")
        return metrics


def main(params_path: str) -> None:
    params = load_params(params_path)
    seed, mcfg, bcfg = params["seed"], params["mlflow"], params["baseline"]
    fdir = PROJECT_ROOT / params["data"]["features_dir"]

    uri = os.environ.get("MLFLOW_TRACKING_URI", mcfg["tracking_uri"])
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(PROJECT_ROOT / uri[len("sqlite:///"):])  # абсолютный путь к БД
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(mcfg["experiment_name"])

    data = [load_split(fdir, n) for n in ("train", "cal", "holdout")]
    features = list(data[0][0].columns)
    tags = {"git_commit": git_commit(), "stage": "baseline",
            "rows_train": str(len(data[0][0])), "rows_cal": str(len(data[1][0])),
            "rows_holdout": str(len(data[2][0]))}

    def prep():  # пропуски (inf -> NaN в features.py) + масштаб для линейных моделей
        return [SimpleImputer(strategy="median"), StandardScaler()]

    lr_p = bcfg["logreg"]
    ridge_p = bcfg["ridge"]
    results = {}
    results["dummy_classifier"] = run_classifier(
        "dummy_classifier", make_pipeline(DummyClassifier(strategy="prior")),
        {"strategy": "prior"}, data, None, params_path, features, tags)
    results["logreg_classifier"] = run_classifier(
        "logreg_classifier",
        make_pipeline(*prep(), LogisticRegression(C=lr_p["C"], class_weight="balanced",
                                                  max_iter=lr_p["max_iter"], random_state=seed)),
        {**lr_p, "class_weight": "balanced", "seed": seed}, data, None, params_path, features, tags)
    results["dummy_regressor"] = run_regressor(
        "dummy_regressor", make_pipeline(DummyRegressor(strategy="mean")),
        {"strategy": "mean"}, data, params_path, features, tags)
    results["ridge_regressor"] = run_regressor(
        "ridge_regressor", make_pipeline(*prep(), Ridge(alpha=ridge_p["alpha"], random_state=seed)),
        {**ridge_p, "seed": seed}, data, params_path, features, tags)

    out = PROJECT_ROOT / "metrics" / "baseline_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: {m: float(v) for m, v in r.items() if m.startswith("holdout")}
                   for k, r in results.items()}, f, indent=2)
    print("Метрики сохранены:", out)
    print("Tracking URI:", uri)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    main(ap.parse_args().params)