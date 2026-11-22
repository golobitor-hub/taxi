"""Стадия evaluate: метрики и графики основной модели на holdout + запись run в MLflow.

Run "hgb_classifier" попадает в тот же эксперимент, что и baseline-модели, поэтому в MLflow UI
их можно сравнить напрямую. holdout_f1/precision/recall считаются так же, как у baseline
(порог с лучшим F1 на calibration); метрики рабочего порога (target recall) — с суффиксом
_at_target_recall.
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.tracking import best_f1_threshold, git_commit, log_model, setup_mlflow
from src.utils import PROJECT_ROOT, ensure_parent, load_params


def log_run(params, params_path, model, metrics, y, proba, fdir):
    """Логирует основную модель в MLflow: гиперпараметры, метрики, веса, графики."""
    setup_mlflow(params["mlflow"])
    X_cal = pd.read_parquet(fdir / "X_cal.parquet")
    y_cal = pd.read_parquet(fdir / "meta_cal.parquet")["is_anomaly"].to_numpy()
    thr_f1 = best_f1_threshold(y_cal, model.predict_proba(X_cal)[:, 1])
    pred_f1 = (proba >= thr_f1).astype("int8")

    with mlflow.start_run(run_name="hgb_classifier"):
        mlflow.set_tags({"git_commit": git_commit(), "stage": "evaluate", "task": "classification",
                         "model_family": type(model).__name__, "rows_holdout": str(len(y))})
        mlflow.log_params({**params["model"], "seed": params["seed"], "class_weight": "balanced",
                           "n_iter": int(model.n_iter_), "n_features": int(model.n_features_in_),
                           "target_recall": params["threshold"]["target_recall"],
                           "decision_threshold": round(thr_f1, 6),
                           "decision_threshold_target_recall": round(metrics["threshold"], 6)})
        mlflow.log_metrics({
            "holdout_roc_auc": metrics["roc_auc"],
            "holdout_pr_auc": metrics["pr_auc"],
            "holdout_f1": f1_score(y, pred_f1, zero_division=0),
            "holdout_precision": precision_score(y, pred_f1, zero_division=0),
            "holdout_recall": recall_score(y, pred_f1, zero_division=0),
            "holdout_f1_at_target_recall": metrics["f1"],
            "holdout_precision_at_target_recall": metrics["precision"],
            "holdout_recall_at_target_recall": metrics["recall"],
            "holdout_missed_anomaly_percent": metrics["missed_anomaly_percent"],
        })
        mlflow.log_artifact(str(PROJECT_ROOT / "metrics" / "holdout_plots.png"))
        mlflow.log_artifact(str(PROJECT_ROOT / "models" / "threshold.json"))
        mlflow.log_artifact(str(params_path))
        log_model(model, "model")
    print(f"[hgb_classifier] записан в MLflow; F1 при F1-оптимальном пороге: "
          f"{f1_score(y, pred_f1):.4f}")


def main(params_path: str) -> None:
    params = load_params(params_path)
    fdir = PROJECT_ROOT / params["data"]["features_dir"]

    X = pd.read_parquet(fdir / "X_holdout.parquet")
    meta = pd.read_parquet(fdir / "meta_holdout.parquet")
    y = meta["is_anomaly"].to_numpy()

    model = joblib.load(PROJECT_ROOT / "models" / "model.joblib")
    with open(PROJECT_ROOT / "models" / "threshold.json", encoding="utf-8") as f:
        thr = json.load(f)["threshold"]

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype("int8")
    print(classification_report(y, pred, target_names=["normal", "anomaly"], digits=4))

    res = meta.copy()
    res["predicted_probability"] = proba
    res["predicted_anomaly"] = pred
    errors = res[res["is_anomaly"] != res["predicted_anomaly"]]
    fp = int(((res.is_anomaly == 0) & (res.predicted_anomaly == 1)).sum())
    fn = int(((res.is_anomaly == 1) & (res.predicted_anomaly == 0)).sum())
    total, real = len(res), int((res.is_anomaly == 1).sum())

    metrics = {
        "threshold": thr,
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "baseline_pr_auc": float(y.mean()),
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "holdout_rows": total,
        "errors": len(errors),
        "false_positive": fp,
        "false_negative": fn,
        "missed_anomaly_percent": fn / real * 100 if real else 0.0,
    }
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(ensure_parent(PROJECT_ROOT / "metrics" / "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    errors.to_csv(PROJECT_ROOT / "metrics" / "holdout_errors.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=axes[0], cmap="Blues")
    PrecisionRecallDisplay.from_predictions(y, proba, ax=axes[1])
    RocCurveDisplay.from_predictions(y, proba, ax=axes[2])
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "metrics" / "holdout_plots.png", dpi=120)
    plt.close(fig)

    log_run(params, params_path, model, metrics, y, proba, fdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    main(ap.parse_args().params)
