import pandas as pd

def build_expectations(cfg: dict) -> list:
    import great_expectations as gx  # импорт здесь, чтобы clean_taxi_data работал без GX

    cols = cfg["expected_columns"]
    e = gx.expectations
    rules = [
        e.ExpectTableColumnsToMatchSet(column_set=cols, exact_match=True),
        e.ExpectColumnValuesToBeUnique(column="id"),
        e.ExpectColumnValuesToBeInSet(column="vendor_id", value_set=cfg["vendor_ids"]),
        e.ExpectColumnValuesToBeInSet(column="store_and_fwd_flag", value_set=cfg["store_and_fwd_flags"]),
        e.ExpectColumnValuesToBeBetween(column="passenger_count", min_value=cfg["passenger_min"], max_value=cfg["passenger_max"]),
        e.ExpectColumnValuesToBeBetween(column="trip_duration", min_value=cfg["duration_min_sec"], max_value=cfg["duration_max_sec"]),
    ]
    for prefix in ("pickup", "dropoff"):
        rules.append(e.ExpectColumnValuesToBeBetween(column=f"{prefix}_longitude", min_value=cfg["lon_min"], max_value=cfg["lon_max"]))
        rules.append(e.ExpectColumnValuesToBeBetween(column=f"{prefix}_latitude", min_value=cfg["lat_min"], max_value=cfg["lat_max"]))
    rules += [e.ExpectColumnValuesToNotBeNull(column=c) for c in cols]
    return rules


def validate_dataframe(df: pd.DataFrame, asset_name: str, cfg: dict) -> pd.DataFrame:
    import great_expectations as gx

    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("taxi_pandas_source")
    asset = source.add_dataframe_asset(name=asset_name)
    batch_def = asset.add_batch_definition_whole_dataframe(name=f"{asset_name}_whole_dataframe")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    rows = []
    for expectation in build_expectations(cfg):
        result = batch.validate(expectation, result_format={"result_format": "SUMMARY"})
        res = result.result or {}
        conf = result.expectation_config
        rows.append({
            "expectation": conf.type,
            "column": conf.kwargs.get("column", "TABLE"),
            "success": bool(result.success),
            "element_count": res.get("element_count", len(df)),
            "unexpected_count": res.get("unexpected_count", 0),
            "unexpected_percent": res.get("unexpected_percent", 0.0),
        })
    return pd.DataFrame(rows)


def print_validation_summary(report: pd.DataFrame, title: str) -> None:
    failed = report[~report["success"]]
    print("=" * 70)
    print(title)
    print("=" * 70)
    print("Всего проверок:", len(report))
    print("Успешно:", int(report["success"].sum()))
    print("Провалено:", len(failed))
    if failed.empty:
        print("КОНТРАКТ ПРОЙДЕН: выборка может быть передана дальше.")
    else:
        print("КОНТРАКТ НЕ ПРОЙДЕН: выборка должна быть заблокирована.")
        print(failed[["expectation", "column", "unexpected_count", "unexpected_percent"]]
              .sort_values("unexpected_count", ascending=False).to_string(index=False))


def quality_gate(report: pd.DataFrame, dataset_name: str = "dataset") -> None:
    failed = report[~report["success"]]
    if not failed.empty:
        names = (failed["column"].astype(str) + ": " + failed["expectation"].astype(str)).tolist()
        raise ValueError(
            f"{dataset_name}: DATA QUALITY GATE FAILED. Нарушено правил: {len(failed)}. "
            f"Выборка заблокирована. Проблемы: {names}"
        )
    print(f"{dataset_name}: DATA QUALITY GATE PASSED. Выборка разрешена для дальнейшей обработки.")
