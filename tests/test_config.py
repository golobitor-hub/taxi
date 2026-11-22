"""Проверки конфигурации конвейера: params.yaml и dvc.yaml согласованы с кодом."""
import yaml

from src.utils import PROJECT_ROOT


def test_split_fractions_valid(params):
    s = params["split"]
    assert 0 < s["train_frac"] < s["calibration_frac"] < 1


def test_validation_bounds_consistent(params):
    v = params["validation"]
    assert v["passenger_min"] <= v["passenger_max"]
    assert v["duration_min_sec"] < params["target"]["anomaly_threshold_sec"] < v["duration_max_sec"]
    assert v["lon_min"] < v["lon_max"] and v["lat_min"] < v["lat_max"]


def test_threshold_and_model_params(params):
    assert 0 < params["threshold"]["target_recall"] <= 1
    assert params["model"]["learning_rate"] > 0
    assert params["baseline"]["logreg"]["C"] > 0 and params["baseline"]["ridge"]["alpha"] >= 0


def test_dvc_stages_reference_existing_code_and_params(params):
    dvc = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    for name, stage in dvc["stages"].items():
        module = stage["cmd"].split("python -m ")[1].split()[0]
        assert (PROJECT_ROOT / (module.replace(".", "/") + ".py")).exists(), name
        for dep in stage.get("deps", []):
            if dep.startswith("src/"):
                assert (PROJECT_ROOT / dep).exists(), f"{name}: {dep}"
        for p in stage.get("params", []):
            node = params
            for key in p.split("."):
                assert key in node, f"{name}: параметр {p} отсутствует в params.yaml"
                node = node[key]
