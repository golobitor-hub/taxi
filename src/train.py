import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

from src.utils import PROJECT_ROOT, ensure_parent, load_params


def balanced_weights(y: pd.Series) -> np.ndarray:
    w = (y == 0).sum() / (y == 1).sum()
    return np.where(y.to_numpy() == 1, w, 1.0)

def pick_threshold(y_true, proba, target_recall: float):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    valid = np.where(recall[:-1] >= target_recall)[0]
    if len(valid) > 0:
        i = valid[np.argmax(precision[:-1][valid])]
    else:
        i = int(np.argmax(recall[:-1]))
    return float(thresholds[i]), float(recall[i]), float(precision[i])

def main(params_path: str) -> None:
    params = load_params(params_path)
    seed, mp = params["seed"], params["model"]
    fdir = PROJECT_ROOT / params["data"]["features_dir"]

    def load(name):
        X = pd.read_parquet(fdir / f"X_{name}.parquet")
        y = pd.read_parquet(fdir / f"meta_{name}.parquet")["is_anomaly"]
        y.index = X.index
        return X, y

    X_train, y_train = load("train")
    X_cal, y_cal = load("cal")

    model = HistGradientBoostingClassifier(
        learning_rate=mp["learning_rate"], max_iter=mp["max_iter"],
        max_leaf_nodes=mp["max_leaf_nodes"], min_samples_leaf=mp["min_samples_leaf"],
        l2_regularization=mp["l2_regularization"], early_stopping=True,
        validation_fraction=mp["validation_fraction"], random_state=seed)
    model.fit(X_train, y_train, sample_weight=balanced_weights(y_train))
    print("Фактическое число итераций:", model.n_iter_)

    p_cal = model.predict_proba(X_cal)[:, 1]
    target = params["threshold"]["target_recall"]
    thr, rec, prec = pick_threshold(y_cal, p_cal, target)
    print(f"Целевой recall {target:.0%} | порог {thr:.6f} | recall {rec:.2%} | precision {prec:.2%}")

    X_dev, y_dev = pd.concat([X_train, X_cal]), pd.concat([y_train, y_cal])
    final_model = HistGradientBoostingClassifier(
        learning_rate=mp["learning_rate"], max_iter=model.n_iter_,
        max_leaf_nodes=mp["max_leaf_nodes"], min_samples_leaf=mp["min_samples_leaf"],
        l2_regularization=mp["l2_regularization"], early_stopping=False, random_state=seed)
    final_model.fit(X_dev, y_dev, sample_weight=balanced_weights(y_dev))

    mdir = PROJECT_ROOT / "models"
    joblib.dump(model, ensure_parent(mdir / "model.joblib"))
    joblib.dump(final_model, mdir / "final_model.joblib")
    with open(mdir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "target_recall": target, "calibration_recall": rec,
                   "calibration_precision": prec, "n_iter": int(model.n_iter_)}, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    main(ap.parse_args().params)
