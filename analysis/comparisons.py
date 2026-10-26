"""Optional independent-group comparisons, distinct from regression screening."""

from __future__ import annotations

import warnings as pywarnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from .cleaning import CATEGORICAL_TYPES
from .descriptive import category_levels, descriptive_statistics


TEST_COLUMNS = ["variable", "p_value", "method", "n", "note"]


def _test_continuous(samples: list[np.ndarray], mode: str) -> tuple[float, str, str]:
    if len(samples) < 2 or any(len(values) == 0 for values in samples):
        return np.nan, "Not estimated", "At least two groups with nonmissing observations are required."
    if any(not np.isfinite(values).all() for values in samples):
        return np.nan, "Not estimated", "Continuous values must be numeric and finite after cleaning."
    if mode in {"welch", "pooled"} and any(len(values) < 2 for values in samples):
        return np.nan, "Not estimated", "Mean comparisons require at least two observations per group."
    if np.ptp(np.concatenate(samples)) == 0:
        return np.nan, "Not estimated", "All observed values are identical; no meaningful comparison can be estimated."
    if mode in {"welch", "pooled"} and all(np.var(values, ddof=1) == 0 for values in samples):
        return np.nan, "Not estimated", "All groups have zero within-group variance; a mean-comparison standard error cannot be estimated."

    note = "Two-sided; independent observations assumed."
    with pywarnings.catch_warnings(record=True) as caught:
        pywarnings.simplefilter("always")
        if mode == "rank":
            if len(samples) == 2:
                method = "Mann–Whitney U"
                p = stats.mannwhitneyu(*samples, alternative="two-sided", method="auto").pvalue
            else:
                method = "Kruskal–Wallis"
                if any(len(values) < 5 for values in samples):
                    return np.nan, method, "Too few observations in at least one group for the chi-square approximation (require at least 5 per group)."
                p = stats.kruskal(*samples).pvalue
            note += " Compares distributions/ranks; a median-only interpretation requires additional assumptions."
        elif len(samples) == 2:
            method = "Welch t-test" if mode == "welch" else "Independent t-test (equal variances)"
            p = stats.ttest_ind(*samples, equal_var=(mode == "pooled")).pvalue
            note += " Compares group means."
        elif mode == "welch":
            method = "Welch ANOVA"
            if any(np.var(values, ddof=1) == 0 for values in samples):
                return np.nan, method, "Welch ANOVA is undefined when a group has zero sample variance."
            p = anova_oneway(samples, use_var="unequal", welch_correction=True).pvalue
            note += " Omnibus comparison of group means; no post-hoc comparisons performed."
        else:
            method = "One-way ANOVA (equal variances)"
            p = stats.f_oneway(*samples).pvalue
            note += " Omnibus comparison of group means; no post-hoc comparisons performed."
    if not np.isfinite(p):
        return np.nan, method, "The test is undefined for these data (check group sizes and within-group variation)."
    if caught:
        note += " Numerical warning: " + "; ".join(dict.fromkeys(str(item.message) for item in caught))
    return float(p), method, note


def _test_categorical(table: pd.DataFrame) -> tuple[float, str, str]:
    observed = table.to_numpy(dtype=int)
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        return np.nan, "Not estimated", "At least two observed categories and two groups are required."
    _, p, _, expected = stats.chi2_contingency(observed, correction=False)
    sparse = bool((expected < 5).any())
    if sparse and observed.shape == (2, 2):
        p = stats.fisher_exact(observed, alternative="two-sided").pvalue
        return float(p), "Fisher exact (two-sided)", "At least one expected cell count is below 5."
    # Conservative: do not present an asymptotic test when any expected count is small.
    if sparse:
        return np.nan, "Chi-square not estimated", (
            "Sparse contingency table: at least one expected count is below 5. "
            "An exact or Monte Carlo RxC test is not implemented; inspect the descriptive counts."
        )
    return float(p), "Pearson chi-square", "All expected cell counts are at least 5; no continuity correction."


def compare_groups(
    data: pd.DataFrame,
    types: dict[str, str],
    group: str,
    variables: list[str] | None = None,
    include_tests: bool = True,
    continuous_test: str = "welch",
    category_orders: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Create a baseline table and, optionally, unadjusted descriptive tests.

    Overall summaries include every supplied row. Group columns exclude rows
    with missing group membership. Every p-value uses only rows observed for
    that variable and group. These tests do not feed regression screening.
    """
    orders = category_orders or {}
    warnings: list[str] = []
    info: dict[str, Any] = {
        "group": group, "overall_n": len(data), "continuous_test": continuous_test,
        "include_tests": bool(include_tests),
        "denominator_note": "Category percentages use nonmissing observations within each column; missing percentages use all rows in that column. Overall includes rows with missing group membership.",
    }
    empty = {"table": pd.DataFrame(), "tests": pd.DataFrame(columns=TEST_COLUMNS), "warnings": warnings, "info": info}
    if group not in data.columns:
        warnings.append("Choose a grouping variable present in the cleaned dataset.")
        return empty
    if types.get(group) not in CATEGORICAL_TYPES:
        warnings.append("The grouping variable must be categorical; review its variable type first.")
        return empty
    if continuous_test not in {"welch", "rank", "pooled"}:
        warnings.append("Choose continuous_test='welch', 'rank', or 'pooled'.")
        return empty
    selected = [column for column in (data.columns if variables is None else variables)
                if column != group and column in data.columns and types.get(column) in CATEGORICAL_TYPES | {"continuous"}]
    selected = list(dict.fromkeys(selected))
    observed_group = data[group].notna()
    labels = data[group].map(lambda value: str(value) if pd.notna(value) else np.nan)
    levels = [level for level in category_levels(data[group], orders.get(group)) if (labels == level).any()]
    group_sizes = {level: int((labels == level).sum()) for level in levels}
    info.update({"groups": group_sizes, "missing_group_n": int((~observed_group).sum()),
                 "comparison_n": int(observed_group.sum())})
    if len(levels) < 2:
        warnings.append("Fewer than two nonmissing groups are available; group hypothesis tests cannot be estimated.")
    if info["missing_group_n"]:
        warnings.append(f"{info['missing_group_n']} rows have missing {group}; they remain in Overall but are excluded from group columns and tests.")
    if not selected:
        warnings.append("Select at least one numerical or categorical variable other than the grouping variable.")
        return empty

    # Build every group with the same category universe so absent categories show 0.
    consistent_orders = dict(orders)
    for variable in selected:
        if types[variable] in CATEGORICAL_TYPES:
            consistent_orders[variable] = category_levels(data[variable], orders.get(variable))
    try:
        summary = descriptive_statistics(data, types, selected, consistent_orders)["table1"]
        for level in levels:
            name = f"Group: {level} (n={group_sizes[level]})"
            sub = descriptive_statistics(data.loc[labels == level], types, selected, consistent_orders)["table1"]
            # Explicitly shared category universes yield the same row order,
            # including groups with zero observations of an individual category.
            # Positional alignment also permits a literal category named "Range"
            # or "Missing (all rows denominator)" without a duplicate-key join.
            if not summary[["variable", "level"]].equals(sub[["variable", "level"]]):
                raise ValueError("Group summary rows could not be aligned with the overall category order.")
            summary[name] = sub["Overall"].to_numpy()
    except ValueError as exc:
        warnings.append(str(exc))
        return empty

    tests = []
    for variable in selected:
        complete = observed_group & data[variable].notna()
        row: dict[str, Any] = {"variable": variable, "p_value": np.nan, "method": "Not requested", "n": int(complete.sum()), "note": ""}
        if include_tests:
            try:
                if types[variable] == "continuous":
                    samples = [pd.to_numeric(data.loc[complete & (labels == level), variable], errors="coerce").to_numpy(dtype=float) for level in levels]
                    row["p_value"], row["method"], row["note"] = _test_continuous(samples, continuous_test)
                else:
                    contingency = pd.crosstab(labels[complete], data.loc[complete, variable].map(str))
                    if len(contingency) != len(levels):
                        row["method"] = "Not estimated"
                        row["note"] = "At least one group has no nonmissing values for this variable."
                    else:
                        row["p_value"], row["method"], row["note"] = _test_categorical(contingency)
            except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
                row["method"] = "Not estimated"
                row["note"] = f"The requested comparison could not be estimated: {exc}"
        tests.append(row)
    tests_frame = pd.DataFrame(tests, columns=TEST_COLUMNS)
    for column in ["p_value", "method", "n", "note"]:
        summary[column] = np.nan if column in {"p_value", "n"} else ""
    for row in tests:
        index = summary.index[summary["variable"] == row["variable"]][0]
        for column in ["p_value", "method", "n", "note"]:
            summary.at[index, column] = row[column]
    if include_tests:
        warnings.append("Group comparisons are descriptive and unadjusted. Multiple testing is not corrected; use the regression stages for adjusted associations.")
    return {"table": summary, "tests": tests_frame, "warnings": warnings, "info": info}
