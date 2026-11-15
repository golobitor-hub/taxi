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
│   ├── evaluate.py         # стадия evaluate: метрики holdout, графики + run в MLflow (lab1, п. 12; лаба 4)
│   ├── predict.py          # стадия predict: предсказания на test.csv
│   ├── baseline.py         # стадия baseline: Dummy / LogReg / Ridge + MLflow    (лаба 4)
│   └── tracking.py         # общие функции MLflow: URI, git-тег, логирование модели (лаба 4)
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


---

# Лабораторная работа №4. Baseline-модель и трекинг экспериментов (MLflow)

Цель: обучить первую итерацию модели и автоматически логировать гиперпараметры, метрики
(ROC-AUC, F1, RMSE) и веса моделей в MLflow.

## Что обучается

Все модели обучаются на одних и тех же признаках и временном split из стадии `featurize`.
Метрики считаются на holdout (последние 15% по времени).

| Run в MLflow | Задача | Модель | Стадия |
|---|---|---|---|
| `dummy_classifier` | аномалия (> 3600 с) | `DummyClassifier(prior)`: нижняя граница качества | `baseline` |
| `logreg_classifier` | аномалия | `Imputer → StandardScaler → LogisticRegression(class_weight=balanced)` | `baseline` |
| `dummy_regressor` | `log1p(trip_duration)` | `DummyRegressor(mean)` | `baseline` |
| `ridge_regressor` | `log1p(trip_duration)` | `Imputer → StandardScaler → Ridge` | `baseline` |
| `hgb_classifier` | аномалия | основная модель `HistGradientBoostingClassifier` из стадии `train` | `evaluate` |

Порог классификации для F1 подбирается на calibration (максимум F1) и затем применяется к holdout.
На holdout порог не подбирается. Для `hgb_classifier` отдельно логируются метрики рабочего порога
(recall ≥ 90%) с суффиксом `_at_target_recall`.

## Что логируется в каждый run

- **параметры:** гиперпараметры модели из `params.yaml` (`baseline.*`, `model.*`), `seed`, число признаков,
  выбранный порог, границы обрезки прогноза;
- **метрики:** для train / cal / holdout. Классификация: `roc_auc`, `pr_auc`, `f1`, `precision`, `recall`.
  Регрессия: `rmse_log`, `rmse_sec`, `rmse_sec_normal` (только поездки ≤ 1 ч), `mae_sec`, `r2_log`;
- **артефакты:** веса модели (`mlflow.sklearn`, `model/`), ROC/PR-кривые или scatter «прогноз–факт»,
  `params.yaml`, список признаков (`features.json`), для HGB ещё `threshold.json`;
- **теги:** `git_commit` (с суффиксом `-dirty`, если код не закоммичен), `stage`, `task`, `model_family`,
  размеры выборок.

## Запуск

```bash
pip install -r requirements.txt
dvc repro                       # стадии evaluate и baseline пишут runs в MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

Хранилище MLflow задаётся в `params.yaml → mlflow.tracking_uri` (по умолчанию локальная `mlflow.db`).
Переменная окружения `MLFLOW_TRACKING_URI` его перекрывает, например для удалённого сервера.
`mlflow.db` и `mlruns/` не версионируются ни в Git, ни в DVC (см. `.gitignore`).

Загрузка весов из MLflow:

```python
import mlflow
model = mlflow.sklearn.load_model("runs:/<run_id>/model")
```

## Результаты (holdout, 218 687 поездок, доля аномалий 1.15%)

| Модель | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| dummy_classifier | 0.500 | 0.011 | 0.023 |
| logreg_classifier | 0.922 | 0.272 | 0.368 |
| **hgb_classifier** | **0.936** | **0.528** | **0.540** |

| Модель | RMSE (log) | R² (log) | RMSE, с | RMSE ≤ 1 ч, с | MAE, с |
|---|---|---|---|---|---|
| dummy_regressor | 0.814 | -0.006 | 3277 | 655 | 601 |
| ridge_regressor | **0.614** | **0.427** | 3341 | 906 | **528** |

Выводы:

- Обе baseline-модели заметно лучше Dummy по ранжирующим метрикам. Основная модель HGB даёт
  почти вдвое больший PR-AUC, чем LogReg (0.53 против 0.27): нелинейность действительно нужна.
- Ridge выигрывает у Dummy по RMSE в log-шкале (на этой шкале модель и обучалась) и по MAE,
  но проигрывает по RMSE в секундах. Модель линейна по `log1p(duration)`, поэтому прогноз растёт
  экспоненциально с `distance_km`. На поездках длиннее 40 км (аэропорты) она даёт 30 000–86 000 с
  вместо ~2 000–3 500 с, и 1% таких ошибок даёт ~70% квадратичной ошибки. Прогнозы обрезаются
  границами контракта данных `[duration_min_sec, duration_max_sec]`. Без обрезки RMSE был 3913 с.
  Следующая итерация: `log(distance)` как признак или нелинейный регрессор.
- RMSE в секундах (~3300) в основном определяют поездки длиной ~86 000 с (ошибки таксометра).
  Поэтому дополнительно логируется `rmse_sec_normal` по поездкам ≤ 1 ч.

## Сравнение экспериментов

В MLflow UI: эксперимент `taxi-anomaly-baseline` → отметить runs → **Compare**
(параллельные координаты, scatter по метрикам). Из кода:

```python
mlflow.search_runs(experiment_names=["taxi-anomaly-baseline"],
                   order_by=["metrics.holdout_pr_auc DESC"])
```

Новый эксперимент (например, `baseline.logreg.C: 0.1` в `params.yaml`) запускается через
`dvc repro` и попадает в MLflow как новый run. Метрики можно сравнить и средствами DVC:
`dvc metrics diff`.
