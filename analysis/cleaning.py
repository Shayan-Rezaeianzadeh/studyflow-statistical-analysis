"""Read and preview explicit data-cleaning operations without mutating input data.

The caller must obtain user approval before replacing its analysis dataset with
``clean_data(...)["data"]``.  Reading a file does not reinterpret string missing
markers; those are handled by the separately previewable cleaning step.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


TYPE_LABELS = {
    "continuous": "Continuous numerical",
    "binary": "Binary categorical",
    "nominal": "Nominal categorical",
    "ordinal": "Ordinal categorical",
    "datetime": "Date/time",
    "identifier": "Identifier (excluded)",
}
DEFAULT_MISSING_TOKENS = ["", "NA", "N/A", "NaN", "null", "None", "?"]
CATEGORICAL_TYPES = {"binary", "nominal", "ordinal"}


def _as_string(series: pd.Series) -> pd.Series:
    """String category labels with genuine missing values preserved."""
    return series.map(lambda value: str(value) if pd.notna(value) else np.nan)


def _token_mask(series: pd.Series, tokens: list[str]) -> pd.Series:
    return series.map(lambda value: isinstance(value, str) and value in tokens)


def read_dataset(file: Any, filename: str | None = None, sheet_name: Any = 0) -> pd.DataFrame:
    """Read a UTF-8 CSV or an XLSX worksheet with duplicate-header validation.

    ``file`` may be a filesystem path, bytes, or a binary file-like object such
    as a Streamlit UploadedFile. An existing stream position is restored.
    """
    if isinstance(file, (str, Path)):
        source = Path(file)
        filename = filename or source.name
        payload = source.read_bytes()
    elif isinstance(file, bytes):
        payload = file
    else:
        filename = filename or getattr(file, "name", "dataset.csv")
        position = file.tell() if hasattr(file, "tell") else None
        if hasattr(file, "seek"):
            file.seek(0)
        payload = file.read()
        if position is not None and hasattr(file, "seek"):
            file.seek(position)
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

    suffix = Path(filename or "dataset.csv").suffix.lower()
    if suffix == ".csv":
        try:
            decoded = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV encoding is not UTF-8. Save the file as CSV UTF-8 and try again.") from exc
        reader = csv.reader(io.StringIO(decoded))
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise ValueError("The uploaded file is empty.") from exc
        for row_number, row in enumerate(reader, start=2):
            if row and len(row) != len(headers):
                raise ValueError(f"CSV record {row_number} has {len(row)} fields but the header has {len(headers)}. "
                                 "Check the delimiter and quoted fields in the source file.")
        data = pd.read_csv(io.StringIO(decoded), keep_default_na=False, dtype=object,
                           skip_blank_lines=False)
    elif suffix == ".xlsx":
        headers_frame = pd.read_excel(
            io.BytesIO(payload), sheet_name=sheet_name, header=None, nrows=1,
            keep_default_na=False, engine="openpyxl",
        )
        if not isinstance(headers_frame, pd.DataFrame) or headers_frame.empty:
            raise ValueError("Choose one nonempty Excel worksheet.")
        headers = headers_frame.iloc[0].tolist()
        data = pd.read_excel(
            io.BytesIO(payload), sheet_name=sheet_name, keep_default_na=False,
            engine="openpyxl", dtype=object,
        )
    else:
        raise ValueError("Choose a .csv or .xlsx file.")

    names = [str(value) for value in headers]
    if any(not name.strip() for name in names):
        raise ValueError("Every column must have a nonempty header in the first row.")
    if len(names) != len(set(names)):
        raise ValueError("Column names must be unique. Rename duplicate headers in the source file.")
    if len(names) != len(data.columns):
        raise ValueError("The header and data contain different numbers of columns.")
    if data.empty:
        raise ValueError("The dataset has no observations below the header row.")
    data.columns = names
    return data


def infer_types(data: pd.DataFrame) -> dict[str, str]:
    """Suggest conservative types; suggestions never change the input values.

    Dates are inferred only from an existing datetime dtype or a date-like
    column name whose nonmissing values all parse. Numeric codes with three or
    more levels remain numerical until the user changes their type.
    """
    result: dict[str, str] = {}
    for column in data.columns:
        series = data[column]
        name = str(column).lower().strip()
        candidate = series.mask(_token_mask(series, DEFAULT_MISSING_TOKENS)).dropna()
        if name in {"id", "identifier", "patientid", "subjectid", "participantid"} or name.endswith("_id"):
            kind = "identifier"
        elif is_datetime64_any_dtype(series):
            kind = "datetime"
        elif candidate.empty:
            kind = "nominal"
        elif candidate.nunique() == 2:
            kind = "binary"
        elif is_numeric_dtype(series) or pd.to_numeric(candidate, errors="coerce").notna().all():
            kind = "continuous"
        elif re.search(r"date|datetime|timestamp|(^|_)dob($|_)", name):
            parsed = pd.to_datetime(candidate, errors="coerce", format="mixed", utc=True)
            kind = "datetime" if parsed.notna().all() else "nominal"
        else:
            kind = "nominal"
        result[column] = kind
    return result


def profile_data(data: pd.DataFrame, types: dict[str, str] | None = None) -> pd.DataFrame:
    """Summarize actual missingness and flag possible inconsistencies for review."""
    types = types or infer_types(data)
    rows = []
    denominator = len(data)
    for column in data.columns:
        series = data[column]
        missing = int(series.isna().sum())
        present = series.dropna()
        marker_count = int(_token_mask(series, DEFAULT_MISSING_TOKENS).sum())
        issues = []
        if marker_count:
            issues.append(f"{marker_count} common missing markers; review before applying")
        text = present[present.map(lambda value: isinstance(value, str))]
        if not text.empty:
            if (text != text.str.strip()).any():
                issues.append("Leading or trailing whitespace")
            if text.nunique() > text.str.strip().str.casefold().nunique():
                issues.append("Labels differ only by case or whitespace")
            ordinary = text[~text.isin(DEFAULT_MISSING_TOKENS)]
            numeric_n = int(pd.to_numeric(ordinary, errors="coerce").notna().sum())
            if numeric_n:
                issues.append(f"{numeric_n} numeric values stored as text")
        if not present.empty and present.nunique() == 1:
            issues.append("Constant among nonmissing values")
        if present.empty:
            issues.append("All values missing")
        if types.get(column) in CATEGORICAL_TYPES and present.nunique() > 30:
            issues.append("Many categories; check whether this is an identifier")
        if types.get(column) == "continuous":
            converted = pd.to_numeric(series, errors="coerce")
            invalid_n = int((series.notna() & (converted.isna() | ~np.isfinite(converted))).sum())
            if invalid_n:
                issues.append(f"{invalid_n} nonnumeric or nonfinite values")
        rows.append({
            "variable": column, "type": types.get(column, "nominal"),
            "nonmissing": denominator - missing, "missing": missing,
            "missing_percent": 100.0 * missing / denominator if denominator else np.nan,
            "unique": int(present.nunique()), "issues": "; ".join(issues),
        })
    return pd.DataFrame(rows, columns=[
        "variable", "type", "nonmissing", "missing", "missing_percent", "unique", "issues",
    ])


def clean_data(
    data: pd.DataFrame,
    types: dict[str, str],
    options: dict[str, Any] | None = None,
    ordinal_orders: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Return a cleaning preview, action log, and validation errors.

    Numeric/date parse failures block a usable preview unless ``coerce_invalid``
    is explicitly true. No rows are removed unless ``drop_duplicates`` is true.
    There is no imputation, outlier removal, or implicit case deletion here.
    """
    settings = {
        "missing_tokens": DEFAULT_MISSING_TOKENS,
        "strip_whitespace": False,
        "drop_duplicates": False,
        "include_columns": None,
        "recodes": {},
        "coerce_invalid": False,
    }
    settings.update(options or {})
    orders = ordinal_orders or {}
    cleaned = data.copy(deep=True)
    log: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    changed_rows = np.zeros(len(data), dtype=bool)
    changed_cells_by_column: dict[str, np.ndarray] = {}
    coerced_count = 0

    def record(action: str, column: str, mask: pd.Series | np.ndarray, detail: str) -> None:
        affected = np.asarray(mask, dtype=bool)
        count = int(affected.sum())
        if count:
            changed_rows[:] |= affected
            if column not in changed_cells_by_column:
                changed_cells_by_column[column] = np.zeros(len(data), dtype=bool)
            changed_cells_by_column[column] |= affected
        log.append({
            "action": action, "variable": column, "affected_rows": count,
            "affected_cells": count, "detail": detail,
        })

    if not data.columns.is_unique:
        errors.append("Column names must be unique.")
        return {"data": cleaned, "types": types.copy(), "log": log, "warnings": warnings,
                "errors": errors, "impact": {}}
    include = settings["include_columns"]
    if include is not None:
        unknown = [column for column in include if column not in data.columns]
        if unknown:
            errors.append("The selected columns include columns not found in the source dataset.")
        include = [column for column in include if column in data.columns]
        include = list(dict.fromkeys(include))
        excluded = [column for column in data.columns if column not in include]
        cleaned = cleaned.loc[:, include].copy()
    else:
        excluded = []
    if excluded:
        log.append({
            "action": "exclude_columns", "variable": ", ".join(map(str, excluded)),
            "affected_rows": len(data), "affected_cells": len(data) * len(excluded),
            "detail": f"Excluded {len(excluded)} user-selected columns.",
        })
    if cleaned.shape[1] == 0:
        errors.append("Select at least one column to retain.")
    if len(cleaned) == 0:
        errors.append("The dataset contains no observations.")
    resolved_types = {column: types.get(column, "nominal") for column in cleaned.columns}
    tokens = settings["missing_tokens"]
    if not isinstance(tokens, (list, tuple)) or not all(isinstance(token, str) for token in tokens):
        errors.append("Missing-value markers must be a list of text strings.")
        tokens = []
    if not isinstance(settings["recodes"], dict):
        errors.append("Category recoding settings must be a dictionary keyed by column name.")
        settings["recodes"] = {}

    for column in cleaned.columns:
        series = cleaned[column]
        # Object storage allows replacement even when the source has categorical dtype.
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)
        if settings["strip_whitespace"]:
            stripped = series.map(lambda value: value.strip() if isinstance(value, str) else value)
            changed = series.notna() & (series.astype(str) != stripped.astype(str))
            record("strip_whitespace", column, changed, "Trimmed leading and trailing whitespace.")
            series = stripped
        marker_mask = _token_mask(series, list(tokens))
        record("standardize_missing", column, marker_mask,
               "Replaced the explicitly configured, case-sensitive missing markers.")
        series = series.mask(marker_mask, np.nan)

        recodes = settings["recodes"].get(column, {})
        if recodes:
            if resolved_types[column] not in CATEGORICAL_TYPES:
                errors.append(f"{column}: recoding is available for categorical variables only.")
            elif not isinstance(recodes, dict) or not all(isinstance(value, str) for value in recodes.values()):
                errors.append(f"{column}: recodes must map existing text labels to new text labels.")
            else:
                labels = _as_string(series)
                replacements = {str(old): new for old, new in recodes.items()}
                recoded = labels.map(lambda value: replacements.get(value, value) if pd.notna(value) else np.nan)
                changed = labels.notna() & (labels != recoded)
                record("recode_categories", column, changed, f"Applied {len(replacements)} label mappings.")
                series = recoded

        kind = resolved_types[column]
        if kind not in TYPE_LABELS:
            errors.append(f"{column}: unsupported variable type {kind!r}.")
        elif kind == "continuous":
            converted = pd.to_numeric(series, errors="coerce").astype(float)
            invalid = series.notna() & (converted.isna() | ~np.isfinite(converted))
            invalid_n = int(invalid.sum())
            if invalid_n and not settings["coerce_invalid"]:
                errors.append(f"{column}: {invalid_n} nonnumeric or nonfinite values cannot be converted; "
                              "correct them or explicitly allow conversion to missing.")
            else:
                if invalid_n:
                    converted = converted.mask(invalid, np.nan)
                    coerced_count += invalid_n
                    record("coerce_invalid_numeric", column, invalid,
                           "Converted invalid or infinite numeric values to missing by user request.")
                text_mask = series.notna() & series.map(lambda value: isinstance(value, str)) & ~invalid
                record("convert_numeric", column, text_mask, "Parsed numeric text without rounding.")
                series = converted
        elif kind == "datetime":
            converted = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
            invalid = series.notna() & converted.isna()
            invalid_n = int(invalid.sum())
            if invalid_n and not settings["coerce_invalid"]:
                errors.append(f"{column}: {invalid_n} unrecognized dates cannot be converted; "
                              "correct them or explicitly allow conversion to missing.")
            else:
                if invalid_n:
                    coerced_count += invalid_n
                    record("coerce_invalid_datetime", column, invalid,
                           "Converted unrecognized dates to missing by user request.")
                record("convert_datetime", column, series.notna() & ~invalid,
                       "Parsed dates; offset-aware times normalized to UTC, naive times interpreted as UTC.")
                series = converted
        elif kind in CATEGORICAL_TYPES:
            label_conversion = series.notna() & ~series.map(lambda value: isinstance(value, str))
            record("convert_category_labels", column, label_conversion,
                   "Represented nonmissing category labels as text; genuine missing values preserved.")
            series = _as_string(series)
            levels = series.dropna().unique().tolist()
            if kind == "binary" and len(levels) > 2:
                errors.append(f"{column}: a binary variable has {len(levels)} observed categories; "
                              "recode it or choose nominal/ordinal.")
            if kind == "ordinal":
                supplied_order = orders.get(column, [])
                order = [str(value) for value in supplied_order] if isinstance(supplied_order, (list, tuple)) else []
                if not isinstance(supplied_order, (list, tuple)) or len(order) != len(set(order)) or set(order) != set(levels):
                    errors.append(f"{column}: specify each observed category exactly once in its intended order.")
                else:
                    series = pd.Series(pd.Categorical(series, categories=order, ordered=True),
                                       index=series.index, name=series.name)
                    log.append({
                        "action": "set_category_order", "variable": column,
                        "affected_rows": int(series.notna().sum()), "affected_cells": 0,
                        "detail": "Applied category order: " + " → ".join(order),
                    })
        cleaned[column] = series
        log.append({
            "action": "assign_variable_type", "variable": column,
            "affected_rows": int(series.notna().sum()), "affected_cells": 0,
            "detail": f"Selected variable type: {kind}. This is metadata; value changes are logged separately.",
        })
        if kind != "identifier" and cleaned[column].nunique(dropna=True) < 2:
            warnings.append(f"{column}: fewer than two nonmissing distinct values; regression cannot use it as a predictor.")

    duplicates_removed = 0
    if settings["drop_duplicates"] and cleaned.shape[1] > 0:
        duplicate_mask = cleaned.duplicated(keep="first")
        duplicates_removed = int(duplicate_mask.sum())
        changed_rows[:] |= np.asarray(duplicate_mask)
        log.append({
            "action": "drop_duplicates", "variable": "All retained columns",
            "affected_rows": duplicates_removed,
            "affected_cells": duplicates_removed * cleaned.shape[1],
            "detail": "Removed exact duplicate rows after other cleaning operations; retained first occurrence.",
        })
        cleaned = cleaned.loc[~duplicate_mask].copy()
    elif cleaned.shape[1] > 0:
        duplicate_n = int(cleaned.duplicated().sum())
        if duplicate_n:
            warnings.append(f"{duplicate_n} exact duplicate rows are retained; removal requires an explicit choice.")

    if excluded:
        changed_rows[:] = True
    impact = {
        "rows_before": int(len(data)), "rows_after": int(len(cleaned)),
        "columns_before": int(data.shape[1]), "columns_after": int(cleaned.shape[1]),
        "duplicate_rows_removed": duplicates_removed,
        "missing_values_before": int(data.isna().sum().sum()),
        "missing_values_after": int(cleaned.isna().sum().sum()),
        "cells_changed": int(sum(mask.sum() for mask in changed_cells_by_column.values())),
        "rows_affected": int(changed_rows.sum()),
        "invalid_values_coerced": coerced_count, "excluded_columns": excluded,
    }
    return {"data": cleaned, "types": resolved_types, "log": log,
            "warnings": warnings, "errors": errors, "impact": impact}
