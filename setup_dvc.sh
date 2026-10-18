``#!/usr/bin/env bash
# Первичная настройка Git + DVC. Запускать один раз из корня проекта.
# Использование: bash setup_dvc.sh [путь_к_локальному_remote]
set -euo pipefail

REMOTE_DIR="${1:-../dvc_storage}"

# 1. Git и DVC
[ -d .git ] || git init
dvc init
git add .dvc .dvcignore .gitignore

# 2. Remote-хранилище (локальная папка; для Google Drive / S3 замените url)
#    dvc remote add -d storage gdrive://<folder_id>
#    dvc remote add -d storage s3://bucket/path
dvc remote add -d storage "$REMOTE_DIR"

# 3. Версионирование исходных данных (кладём train.csv и test.csv в data/raw/)
dvc add data/raw/train.csv data/raw/test.csv
git add data/raw/train.csv.dvc data/raw/test.csv.dvc data/raw/.gitignore

# 4. Запуск конвейера и фиксация результата
dvc repro
git add dvc.yaml dvc.lock params.yaml src requirements.txt README.md .dvc/config
git commit -m "Pipeline: DVC + модульные скрипты"

# 5. Отправка данных в remote
dvc push
echo "Готово. Данные в $REMOTE_DIR, в Git только .dvc-файлы и код."
