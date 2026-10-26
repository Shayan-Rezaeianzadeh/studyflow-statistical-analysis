"""Separate unadjusted models and whole-variable exploratory screening."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regression import TERM_COLUMNS, fit_model


BIVARIATE_COLUMNS = [
    "variable", "type", "p_value", "estimate", "ci_lower", "ci_upper",
    "n", "method", "status", "reason",
]


def run_bivariate(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    types: dict[str, str],
    references: dict | None = None,
    event=None,
    family: str | None = None,
    missing: str = "complete_case",
    cov_type: str = "nonrobust",
    category_orders: dict | None = None,
    common_sample: bool = False,
) -> dict:
    """Return one overall p-value per predictor and all level-specific effects.

    Default samples are model-specific complete cases. ``common_sample=True``
    masks the outcome wherever *any candidate* is missing, preserving full-data
    predictor levels/reference choices and original row positions. This is a
    deliberate common complete-case restriction, recorded in every model and in
    the result metadata. ``missing='raise'`` always refuses any required missing
    data rather than silently carrying out that restriction.
    """
    predictors = list(dict.fromkeys(predictors))
    work = data
    common_n = None
    common_excluded = 0
    requested = [outcome] + predictors
    if common_sample and all(c in data.columns for c in requested):
        common_mask = data.loc[:, requested].notna().all(axis=1)
        common_n = int(common_mask.sum())
        common_excluded = len(data) - common_n
        if missing == "complete_case":
            work = data.copy()
            # Masking rather than subsetting preserves full-data level catalogs,
            # dataset-relative diagnostic row numbers, and exclusion denominators.
            if pd.api.types.is_numeric_dtype(work[outcome]):
                work[outcome] = work[outcome].astype(float)
            work.loc[~common_mask, outcome] = np.nan

    summaries = []
    effects = []
    models = {}
    for predictor in predictors:
        model = fit_model(
            work, outcome, [predictor], types,
            references=references, event=event, family=family, missing=missing,
            cov_type=cov_type, category_orders=category_orders,
        )
        model["info"]["sample_scope"] = "common complete cases" if common_sample else "model-specific complete cases"
        if common_sample:
            model["info"]["common_sample_candidates"] = list(predictors)
            model["info"]["common_sample_n"] = common_n
            model["info"]["common_sample_excluded"] = common_excluded
            model["info"]["original_missing_by_variable"] = {
                c: int(data[c].isna().sum()) for c in requested if c in data.columns
            }
            if missing == "raise" and common_excluded:
                model.update(success=False, error=(
                    f"{common_excluded} row(s) have missing values across the common candidate set. "
                    "The selected missing-data policy refuses exclusions."
                ))
                model["terms"] = pd.DataFrame(columns=TERM_COLUMNS)
                model["variables"] = model["variables"].iloc[:0]
        models[predictor] = model
        summary = {
            "variable": predictor, "type": types.get(predictor, "unknown"),
            "p_value": np.nan, "estimate": np.nan, "ci_lower": np.nan,
            "ci_upper": np.nan, "n": model["n_complete"],
            "method": model["info"].get("method", "Not estimated"),
            "status": "ok" if model["success"] else "failed",
            "reason": "" if model["success"] else model["error"],
        }
        if model["success"]:
            summary["p_value"] = float(model["variables"].iloc[0]["p_value"])
            if len(model["terms"]) == 1:
                effect = model["terms"].iloc[0]
                for key in ["estimate", "ci_lower", "ci_upper"]:
                    summary[key] = float(effect[key])
            terms = model["terms"].copy()
            terms["n"] = model["n_complete"]
            terms["method"] = model["info"]["method"]
            terms["overall_p_value"] = summary["p_value"]
            effects.append(terms)
        summaries.append(summary)
    return {
        "variables": pd.DataFrame(summaries, columns=BIVARIATE_COLUMNS),
        "terms": pd.concat(effects, ignore_index=True) if effects else pd.DataFrame(
            columns=TERM_COLUMNS + ["n", "method", "overall_p_value"]
        ),
        "models": models,
        "info": {
            "outcome": outcome, "predictors": predictors,
            "family": family or {"continuous": "linear", "binary": "logistic"}.get(types.get(outcome)),
            "event": next((m["info"]["event"] for m in models.values() if m["info"].get("event") is not None), event),
            "references": {
                key: value for model in models.values()
                for key, value in model["info"].get("references", {}).items()
            }, "cov_type": cov_type, "missing": missing,
            "sample_scope": "common complete cases" if common_sample else "model-specific complete cases",
            "n_total": int(len(data)), "common_sample_n": common_n,
            "common_sample_excluded": int(common_excluded),
            "screening": "Optional exploratory screening uses joint whole-variable p-values, not individual dummy levels.",
        },
    }


def screen_variables(result: dict, threshold: float) -> list[str]:
    """Return estimable whole variables whose overall p-value is strictly lower."""
    if not 0 < threshold < 1:
        raise ValueError("The p-value threshold must be between zero and one.")
    table = result["variables"]
    if table.empty:
        return []
    eligible = (table["status"] == "ok") & table["p_value"].notna() & (table["p_value"] < threshold)
    return table.loc[eligible, "variable"].tolist()
