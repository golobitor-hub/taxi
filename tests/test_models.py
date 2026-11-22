"""Юнит-тесты вспомогательных функций обучения и метрик (train, tracking, baseline)."""
import numpy as np
import pandas as pd
import pytest

from src.baseline import ClippedLogRegressor, clf_metrics, reg_metrics
from src.tracking import best_f1_threshold, git_commit
from src.train import balanced_weights, pick_threshold


def test_balanced_weights_equalize_classes():
    y = pd.Series([0] * 90 + [1] * 10)
    w = balanced_weights(y)
    assert w[y == 0].sum() == pytest.approx(w[y == 1].sum())
    assert set(np.unique(w[y == 0])) == {1.0}


def test_pick_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(900), np.ones(100)]
    proba = np.clip(y * 0.5 + rng.normal(0.25, 0.15, size=y.size), 0, 1)
    thr, rec, prec = pick_threshold(y, proba, target_recall=0.9)
    pred = proba >= thr
    assert rec >= 0.9
    assert pred[y == 1].mean() == pytest.approx(rec)
    assert pred[y == 1].sum() / pred.sum() == pytest.approx(prec)


def test_best_f1_threshold_separable():
    y = np.r_[np.zeros(50), np.ones(50)]
    proba = np.r_[np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)]
    thr = best_f1_threshold(y, proba)
    assert 0.4 < thr <= 0.6
    assert clf_metrics("t", y, proba, thr)["t_f1"] == pytest.approx(1.0)


def test_clf_metrics_keys_and_ranges():
    y = np.array([0, 0, 1, 1])
    m = clf_metrics("holdout", y, np.array([0.1, 0.6, 0.4, 0.9]), 0.5)
    assert set(m) == {f"holdout_{k}" for k in ("roc_auc", "pr_auc", "f1", "precision", "recall")}
    assert all(0 <= v <= 1 for v in m.values())


def test_reg_metrics_perfect_prediction():
    dur = np.array([60.0, 600.0, 7200.0])
    m = reg_metrics("h", dur, np.log1p(dur), normal_max_sec=3600)
    assert m["h_rmse_log"] == pytest.approx(0) and m["h_rmse_sec"] == pytest.approx(0, abs=1e-6)
    assert m["h_r2_log"] == pytest.approx(1)


def test_reg_metrics_normal_rmse_ignores_long_trips():
    dur = np.array([600.0, 600.0, 80000.0])
    pred = np.log1p(np.array([600.0, 600.0, 600.0]))
    m = reg_metrics("h", dur, pred, normal_max_sec=3600)
    assert m["h_rmse_sec_normal"] == pytest.approx(0, abs=1e-6)
    assert m["h_rmse_sec"] > 40000


def test_clipped_regressor_respects_bounds():
    class Raw:
        def predict(self, X):
            return np.array([-5.0, np.log1p(600), 50.0])
    pred = np.expm1(ClippedLogRegressor(Raw(), 1, 86400).predict(None))
    np.testing.assert_allclose(pred, [1, 600, 86400])


def test_git_commit_format():
    tag = git_commit()
    assert tag == "unknown" or (len(tag.removesuffix("-dirty")) == 7 and tag[:7].isalnum())
