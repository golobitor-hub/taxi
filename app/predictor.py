"""Загрузка модели и инференс.

Признаки строятся той же функцией make_features, что и при обучении (src/features.py), —
так исключается расхождение между обучением и инференсом (training-serving skew).
"""
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from app.schemas import ModelInfo, Prediction, Trip
from src.features import make_features
from src.utils import load_params


class Predictor:
    def __init__(self, model_path: Path, threshold_path: Path):
        self.model = joblib.load(model_path)
        meta = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
        self.threshold = float(meta["threshold"])
        params = load_params()
        self.destination_known = params["features"]["destination_known"]
        self.features = list(self.model.feature_names_in_)
        self.info = ModelInfo(
            model_type=type(self.model).__name__,
            model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:12],
            n_iter=int(getattr(self.model, "n_iter_", meta.get("n_iter", 0))),
            n_features=len(self.features),
            features=self.features,
            threshold=self.threshold,
            target_recall=float(meta.get("target_recall", 0)),
            anomaly_threshold_sec=params["target"]["anomaly_threshold_sec"],
        )

    def predict(self, trips: list[Trip]) -> list[Prediction]:
        df = pd.DataFrame([t.model_dump() for t in trips])
        X = make_features(df, self.destination_known)[self.features]
        proba = self.model.predict_proba(X)[:, 1]
        return [Prediction(id=t.id, is_anomaly=bool(p >= self.threshold), anomaly_probability=float(p),
                           threshold=self.threshold)
                for t, p in zip(trips, proba, strict=True)]
