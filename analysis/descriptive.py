"""Type-aware descriptive tables with explicit missing-data denominators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cleaning import CATEGORICAL_TYPES


CONTINUOUS_COLUMNS = ["variable", "n", "missing", "missing_percent", "mean", "sd", "median", "q1", "q3", "min", "max"]
CATEGORICAL_COLUMNS = ["variable", "category", "n", "percent", "missing", "missing_percent"]


def category_levels(series: pd.Series, order: list[str] | None = None) -> list[str]:
    """Get reproducible text labels, honoring an explicitly supplied order."""
    observed = set(series.dropna().map(str))
    if order is not None:
        specified = list(dict.fromkeys(map(str, order)))
        return specified + sorted(observed - set(specified))
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(value) for value in series.cat.categories]
    return sorted(observed)


def _fmt(value: float) -> str:
    return "—" if pd.isna(value) or not np.isfinite(value) else f"{value:,.2f}"


def _n_percent(n: int, denominator: int) -> str:
    return f"{n} ({100 * n / denominator:.1f}%)" if denominator else f"{n} (—)"


def descriptive_statistics(
    data: pd.DataFrame,
    types: dict[str, str],
    columns: list[str] | None = None,
    category_orders: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return numeric detail tables plus a readable baseline-style table.

    Category percentages use nonmissing observations of that variable.
    Missing percentages use all observations in the supplied data. Dates and
    identifiers are omitted; they do not have clinical summary statistics here.
    """
    orders = category_orders or {}
    selected = list(data.columns) if columns is None else columns
    continuous: list[dict[str, Any]] = []
    categorical: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    total = len(data)
    for variable in selected:
        if variable not in data.columns:
            continue
        kind = types.get(variable, "nominal")
        if kind not in CATEGORICAL_TYPES | {"continuous"}:
            continue
        series = data[variable]
        missing = int(series.isna().sum())
        n = total - missing
        missing_percent = 100 * missing / total if total else np.nan
        if kind == "continuous":
            values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
            if len(values) != n or not np.isfinite(values).all():
                raise ValueError(f"{variable}: continuous values must be numeric and finite after cleaning.")
            record = {
                "variable": variable, "n": n, "missing": missing, "missing_percent": missing_percent,
                "mean": values.mean(), "sd": values.std(ddof=1), "median": values.median(),
                "q1": values.quantile(0.25), "q3": values.quantile(0.75),
                "min": values.min(), "max": values.max(),
            }
            continuous.append(record)
            baseline.extend([
                {"variable": variable, "level": "Nonmissing n", "Overall": str(n)},
                {"variable": variable, "level": "Mean (SD)", "Overall": f"{_fmt(record['mean'])} ({_fmt(record['sd'])})"},
                {"variable": variable, "level": "Median [Q1, Q3]", "Overall": f"{_fmt(record['median'])} [{_fmt(record['q1'])}, {_fmt(record['q3'])}]"},
                {"variable": variable, "level": "Range", "Overall": f"{_fmt(record['min'])}–{_fmt(record['max'])}"},
            ])
        else:
            present = series.dropna().map(str)
            counts = present.value_counts()
            levels = category_levels(series, orders.get(variable))
            for level in levels:
                count = int(counts.get(level, 0))
                categorical.append({
                    "variable": variable, "category": level, "n": count,
                    "percent": 100 * count / n if n else np.nan,
                    "missing": missing, "missing_percent": missing_percent,
                })
                baseline.append({"variable": variable, "level": level, "Overall": _n_percent(count, n)})
            if not levels:
                categorical.append({"variable": variable, "category": "(no observed categories)", "n": 0,
                                    "percent": np.nan, "missing": missing, "missing_percent": missing_percent})
                baseline.append({"variable": variable, "level": "(no observed categories)", "Overall": "0 (—)"})
        baseline.append({"variable": variable, "level": "Missing (all rows denominator)", "Overall": _n_percent(missing, total)})
    return {
        "continuous": pd.DataFrame(continuous, columns=CONTINUOUS_COLUMNS),
        "categorical": pd.DataFrame(categorical, columns=CATEGORICAL_COLUMNS),
        "table1": pd.DataFrame(baseline, columns=["variable", "level", "Overall"]),
    }
