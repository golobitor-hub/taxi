# Лабораторная работа №3. Модульный конвейер данных и версионирование артефактов (DVC)

Задача: обнаружение аномально долгих поездок такси (`trip_duration > 3600 с`).
Код из ноутбуков lab1 (модель) и lab2 (контракт данных) вынесен в скрипты `src/`,
а тяжёлые файлы (данные, признаки, модели) версионируются через DVC и не попадают в Git.

## Структура

```
taxi_pipeline/
├── dvc.yaml            # описание стадий конвейера (граф зависимостей)
├── params.yaml         # все параметры: границы контракта, split, гиперпараметры, порог
├── requirements.txt
├── setup_dvc.sh        # git init + dvc init + remote + dvc add + dvc push
├── src/
│   ├── utils.py            # загрузка params.yaml, пути
│   ├── data_validation.py  # контракт Great Expectations + quality gate   (lab2, п. 3-6)
│   ├── clean.py            # стадия prepare: валидация -> очистка -> gate (lab2, п. 5-10)
│   ├── features.py         # стадия featurize: признаки + временной split (lab1, п. 8-9)
│   ├── train.py            # стадия train: модель, порог, финальная модель (lab1, п. 10-11, 13)
│   ├── evaluate.py         # стадия evaluate: метрики holdout, графики     (lab1, п. 12)
│   └── predict.py          # стадия predict: предсказания на test.csv
├── data/raw/           # train.csv, test.csv — под контролем DVC
├── models/ metrics/ reports/
```

## Что куда делегировано

| Ноутбук | Что было | Где теперь |
|---|---|---|
| lab2 п. 3-6 | `EXPECTATIONS`, `validate_dataframe`, `quality_gate` | `src/data_validation.py` |
| lab2 п. 7-10 | `clean_taxi_data`, профили, сохранение | `src/clean.py` |
| lab1 п. 8-9 | `make_features`, split 70/15/15 по времени | `src/features.py` |
| lab1 п. 10-11, 13 | HistGradientBoosting, порог по recall 90%, финальная модель | `src/train.py` |
| lab1 п. 12 + итоговые ячейки | holdout-метрики, таблица ошибок | `src/evaluate.py` |
| lab1 п. 13 | предсказания на test | `src/predict.py` |

Что изменилось по сравнению с ноутбуками:

- константы (`3600`, границы NYC, доли split, гиперпараметры, `TARGET_RECALL`) вынесены в `params.yaml`;
- нет `google.colab`, `display`, `!pip` — скрипты работают из консоли и в CI;
- каждая стадия — отдельный CLI-модуль (`python -m src.<stage>`), читает вход с диска и пишет выход на диск;
- quality gate по-настоящему останавливает конвейер: при нарушении контракта `quality_gate` бросает `ValueError`, стадия `prepare` падает, следующие стадии не запускаются;
- EDA (графики распределений, VIF, корреляции из lab1 п. 3-6) в конвейер не включён: это разведка, а не воспроизводимый шаг.

## Быстрый старт

```bash
pip install -r requirements.txt
# положить train.csv и test.csv в data/raw/
bash setup_dvc.sh ../dvc_storage     # путь к локальному remote
```

Вручную то же самое:

```bash
git init && dvc init
dvc remote add -d storage ../dvc_storage
dvc add data/raw/train.csv data/raw/test.csv   # создаст *.dvc и .gitignore для CSV
git add data/raw/*.dvc data/raw/.gitignore .dvc
dvc repro                                      # запуск всех стадий
git add dvc.yaml dvc.lock params.yaml src
git commit -m "pipeline"
dvc push                                       # данные и артефакты -> remote
```

## Демонстрация версионирования

```bash
# 1. изменить параметр, например threshold.target_recall: 0.90 -> 0.95
dvc repro            # пересчитаются только train, evaluate, predict (prepare и featurize из кэша)
dvc metrics diff     # сравнение метрик с предыдущим коммитом
git commit -am "target_recall 0.95" && dvc push

# 2. вернуться к прошлой версии
git checkout HEAD~1
dvc checkout         # файлы данных и моделей возвращаются к состоянию того коммита

# 3. на чистой машине
git clone <repo> && cd <repo>
dvc pull             # скачает данные и артефакты из remote
dvc repro            # ничего не пересчитывает, если всё актуально
```

Полезно для отчёта: `dvc dag` (граф стадий), `dvc status`, `dvc params diff`, `git log -- data/raw/train.csv.dvc`.

## Что лежит в Git, а что в DVC

| В Git | В DVC (remote) |
|---|---|
| `src/`, `params.yaml`, `dvc.yaml`, `dvc.lock` | `data/raw/*.csv` |
| `data/raw/*.csv.dvc` (хэш и размер файла) | `data/processed/`, `data/features/` |
| `.dvc/config`, `.dvcignore`, `.gitignore` | `models/*.joblib`, `data/predictions/` |
| `models/threshold.json`, `metrics/metrics.json` (маленькие, читаются как метрики) | |
