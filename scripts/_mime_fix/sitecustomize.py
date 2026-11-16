"""Подгружается Python автоматически (через PYTHONPATH из scripts/mlflow_ui.py).

На Windows mimetypes читает типы из реестра, и если там .js -> text/plain,
MLflow UI отдаёт свой JS как text/plain: браузер его не выполняет -> белый экран.
Возвращаем правильные типы только для процессов, запущенных этим скриптом.
"""
import mimetypes

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
