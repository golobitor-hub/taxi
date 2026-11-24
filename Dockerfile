# Образ сервиса инференса. Перед сборкой модель должна быть на диске: dvc pull (или dvc repro).
#   docker build -t taxi-anomaly-api:0.6.0 .
#   docker run --rm -p 8000:8000 taxi-anomaly-api:0.6.0      ->  http://127.0.0.1:8000/docs
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Зависимости отдельным слоем: при правке кода они не переустанавливаются
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Только то, что нужно для инференса: признаки, параметры, API и обученная модель
COPY params.yaml .
COPY src/__init__.py src/utils.py src/features.py src/
COPY app/ app/
COPY models/final_model.joblib models/threshold.json models/

# Сервис не работает от root
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
