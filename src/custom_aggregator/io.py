from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


def _coerce_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def read_dataframe(path: Path | str) -> pd.DataFrame:
    file_path = _coerce_path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported file extension '{suffix}' for {file_path}")


def write_dataframe(
    df: pd.DataFrame, path: Path | str, fmt: str, *, orient: str = "records"
) -> None:
    file_path = _coerce_path(path)
    fmt_normalised = fmt.lower()
    if fmt_normalised == "csv":
        df.to_csv(file_path, index=False)
        return
    if fmt_normalised == "parquet":
        df.to_parquet(file_path, index=False)
        return
    if fmt_normalised == "json":
        df.to_json(file_path, orient=orient)
        return
    raise ValueError(f"Unsupported output format '{fmt}'")


def select_columns(df: pd.DataFrame, columns: Optional[Iterable[str]]) -> pd.DataFrame:
    if not columns:
        return df
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Columns {sorted(missing)} not present in DataFrame.")
    return df.loc[:, list(columns)]


def aggregate_tracking(
    df: pd.DataFrame, join_on: Iterable[str], aggregations: Dict[str, str] | None
) -> pd.DataFrame:
    if not aggregations:
        return df.drop_duplicates(subset=list(join_on))
    grouped = df.groupby(list(join_on), dropna=False).agg(aggregations).reset_index()
    return grouped
