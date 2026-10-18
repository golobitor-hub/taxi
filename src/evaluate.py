"""Стадия evaluate: метрики на holdout, графики, таблица ошибок.

Запуск: python -m src.evaluate
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.utils import PROJECT_ROOT, ensure_parent, load_params


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    main(ap.parse_args().params)
