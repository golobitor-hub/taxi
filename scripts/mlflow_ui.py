"""Запуск MLflow UI с исправлением MIME-типов (белый экран на Windows).

    python scripts/mlflow_ui.py            # http://127.0.0.1:5000
    python scripts/mlflow_ui.py --port 5001
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX_DIR = Path(__file__).resolve().parent / "_mime_fix"

env = {**os.environ,
       "PYTHONPATH": os.pathsep.join(filter(None, [str(FIX_DIR), os.environ.get("PYTHONPATH")])),
       "MLFLOW_DISABLE_AGENT_HINT": "1"}
cmd = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", *sys.argv[1:]]
sys.exit(subprocess.call(cmd, cwd=ROOT, env=env))
