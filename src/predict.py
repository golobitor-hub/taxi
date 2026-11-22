import argparse
import json

import joblib
import pandas as pd

from src.utils import PROJECT_ROOT, ensure_parent, load_params


def main(params_path: str) -> None:
    params = load_params(params_path)
    fdir = PROJECT_ROOT / params["data"]["features_dir"]

    X_test = pd.read_parquet(fdir / "X_test.parquet")
    test = pd.read_parquet(fdir / "test_raw.parquet")
    model = joblib.load(PROJECT_ROOT / "models" / "final_model.joblib")
    with open(PROJECT_ROOT / "models" / "threshold.json", encoding="utf-8") as f:
        thr = json.load(f)["threshold"]

    proba = model.predict_proba(X_test)[:, 1]
    out = pd.DataFrame({"id": test["id"], "is_anomaly": (proba >= thr).astype("int8"),
                        "anomaly_probability": proba})
    path = ensure_parent(PROJECT_ROOT / "data" / "predictions" / "anomaly_predictions.csv")
    out.to_csv(path, index=False)

    n_anom = int(out["is_anomaly"].sum())
    print(f"Сохранено: {path} {out.shape}")
    print(f"Найдено аномальных поездок: {n_anom} ({n_anom / len(out):.4%})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    main(ap.parse_args().params)
