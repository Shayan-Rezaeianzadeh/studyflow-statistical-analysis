"""Transparent complete-case OLS and binary logistic regression.

There is deliberately no formula parsing, automatic term deletion, imputation,
regularisation, or stepwise fitting. Categorical predictors enter as whole
variables using treatment contrasts. Ordinal predictors use the same contrasts;
an equally spaced linear trend is never assumed silently.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import MissingDataError, PerfectSeparationError


TERM_COLUMNS = [
    "variable", "term", "reference", "estimate", "ci_lower", "ci_upper",
    "p_value", "coefficient", "std_error",
]
VARIABLE_COLUMNS = ["variable", "p_value", "df", "method"]
CATEGORICAL_TYPES = {"binary", "nominal", "ordinal"}
# Bound the dense, encoded design before allocating dummy arrays. Fitting and
# diagnostics require additional working memory beyond this matrix itself.
MAX_DESIGN_CELLS = 10_000_000


class ModelError(ValueError):
    """An analysis cannot support the requested model or its inference."""


def category_levels(series: pd.Series, order: list | None = None) -> list[str]:
    """Return observed string labels, respecting an explicitly supplied order."""
    observed = set(series.dropna().astype("string").tolist())
    if order is None:
        return sorted(observed)
    labels = [str(x) for x in order]
    if len(labels) != len(set(labels)):
        raise ModelError(f"Category order for {series.name!r} contains duplicate labels.")
    missing = observed.difference(labels)
    if missing:
        raise ModelError(
            f"Category order for {series.name!r} is missing observed labels: "
            + ", ".join(sorted(missing))
        )
    return [x for x in labels if x in observed]


def _empty_result(n: int, info: dict) -> dict:
    return {
        "success": False,
        "error": None,
        "terms": pd.DataFrame(columns=TERM_COLUMNS),
        "variables": pd.DataFrame(columns=VARIABLE_COLUMNS),
        "n_total": int(n),
        "n_complete": 0,
        "n_dropped": int(n),
        "metrics": {},
        "warnings": [],
        "diagnostics": pd.DataFrame(),
        "vif": pd.DataFrame(columns=["variable", "term", "vif"]),
        "info": info,
    }


def _as_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        raise ModelError(
            f"{series.name!r} contains {int(invalid.sum())} nonnumeric value(s). "
            "Review its type or explicitly approve their conversion during cleaning."
        )
    finite = numeric.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ModelError(
            f"{series.name!r} contains infinite numerical values. "
            "Correct them during cleaning before fitting."
        )
    return numeric.astype(float)


def _scaled_design(x: np.ndarray) -> np.ndarray:
    """Scale only for numerical rank/separation checks, not reported effects."""
    scaled = x.copy()
    if x.shape[1] > 1:
        sd = np.std(x[:, 1:], axis=0)
        if (sd == 0).any():
            raise ModelError("A model column is constant in the analysis sample.")
        scaled[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / sd
    return scaled


def _separation_check(x: np.ndarray, y: np.ndarray) -> tuple[bool, str | None]:
    """Detect complete or quasi separation through a bounded linear programme.

    For Z_i = (2*y_i-1)*X_i, separation exists if Z*b >= 0 for all rows,
    with at least one strict inequality. Maximising sum(Z*b) under these
    constraints and bounded b finds either kind of separation. Scaling makes
    the numerical tolerance less dependent on measurement units. Full rank is
    checked before this function. A solver failure is reported as inconclusive.
    """
    z = (2.0 * y - 1.0)[:, None] * _scaled_design(x)
    lp = optimize.linprog(
        c=-z.mean(axis=0), A_ub=-z, b_ub=np.zeros(len(y)),
        bounds=[(-1.0, 1.0)] * x.shape[1], method="highs",
        options={"time_limit": 10.0},
    )
    if not lp.success:
        return False, "Separation screening was inconclusive: " + str(lp.message)
    return bool(-float(lp.fun) > 1e-7), None


def fit_model(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    types: dict[str, str],
    references: dict[str, Any] | None = None,
    event: Any = None,
    family: str | None = None,
    missing: str = "complete_case",
    cov_type: str = "nonrobust",
    category_orders: dict[str, list] | None = None,
) -> dict:
    """Fit one model and return tables, counts, diagnostics, and explicit issues.

    Parameters use the variable-type labels from ``cleaning.py``. ``missing``
    can be ``complete_case`` or ``raise``; neither imputes values. Linear models
    support classical or HC3 covariance, both with t confidence intervals and
    joint F tests. Logistic models use maximum likelihood, model-based standard
    errors, Wald normal confidence intervals, and joint Wald chi-square tests.

    The effect table excludes the intercept (stored separately in ``info``).
    Logistic ``estimate`` and confidence bounds are odds ratios; ``coefficient``
    and ``std_error`` are always on the fitted linear/log-odds scale. ``row_number``
    in diagnostics is the 1-based row position in ``data``, not its index label.
    Invalid models return ``success=False`` and an explanation without silently
    deleting predictors or displaying their unreliable effect estimates.
    Before dummy encoding, the parameter count must leave residual degrees of
    freedom and ``n_complete * parameters_including_intercept`` must not exceed
    ``MAX_DESIGN_CELLS`` (10,000,000). Oversized models fail explicitly; this is
    an application resource limit, not a statistical sample-size recommendation.
    """
    predictors = list(predictors)
    references = dict(references or {})
    category_orders = category_orders or {}
    info = {
        "outcome": outcome, "predictors": predictors, "family": family,
        "event": None, "non_event": None, "references": {},
        "missing": missing, "cov_type": cov_type, "confidence_level": 0.95,
        "categorical_encoding": "Treatment contrasts; ordinal predictors are categorical",
    }
    result = _empty_result(len(data), info)

    try:
        if data.empty:
            raise ModelError("The dataset contains no observations.")
        if not data.columns.is_unique:
            raise ModelError("Column names must be unique before fitting.")
        if not predictors:
            raise ModelError("Choose at least one predictor before fitting.")
        if len(predictors) != len(set(predictors)):
            raise ModelError("A predictor was selected more than once.")
        if outcome in predictors:
            raise ModelError("The outcome cannot also be a predictor.")
        requested = [outcome] + predictors
        absent = [c for c in requested if c not in data.columns]
        if absent:
            raise ModelError("Variables not found: " + ", ".join(absent))
        if missing not in {"complete_case", "raise"}:
            raise ModelError("Choose complete_case or raise for missing-data handling.")
        if family is None:
            family = {"continuous": "linear", "binary": "logistic"}.get(types.get(outcome))
        if family not in {"linear", "logistic"}:
            raise ModelError(
                "This version supports continuous outcomes (linear regression) and "
                "binary outcomes (logistic regression). Review the outcome's type."
            )
        if family == "linear" and types.get(outcome) != "continuous":
            raise ModelError("Linear regression requires an outcome marked continuous.")
        if family == "logistic" and types.get(outcome) != "binary":
            raise ModelError("Logistic regression requires an outcome marked binary.")
        if cov_type not in {"nonrobust", "HC3"}:
            raise ModelError("Supported covariance choices are nonrobust and HC3.")
        if family == "logistic" and cov_type != "nonrobust":
            raise ModelError("HC3 is available for linear regression only.")
        info["family"] = family
        info["method"] = (
            "OLS linear regression; classical SE; joint F tests"
            if family == "linear" and cov_type == "nonrobust"
            else "OLS linear regression; HC3 SE; joint Wald F tests"
            if family == "linear"
            else "Binary logistic regression; maximum likelihood; joint Wald chi-square tests"
        )
        info["interval_method"] = (
            "Two-sided 95% t confidence intervals"
            if family == "linear"
            else "Two-sided 95% Wald normal confidence intervals, exponentiated to odds ratios"
        )

        work = data.loc[:, requested].copy().reset_index(drop=True)
        initially_complete = work.notna().all(axis=1)
        result.update(
            n_complete=int(initially_complete.sum()),
            n_dropped=int((~initially_complete).sum()),
        )
        full_levels: dict[str, list[str]] = {}
        for variable in requested:
            kind = types.get(variable)
            if kind == "continuous":
                work[variable] = _as_numeric(work[variable])
            elif kind in CATEGORICAL_TYPES:
                work[variable] = work[variable].astype("string")
                full_levels[variable] = category_levels(
                    work[variable], category_orders.get(variable)
                )
                if kind == "binary" and len(full_levels[variable]) != 2:
                    raise ModelError(
                        f"{variable!r} is marked binary but has "
                        f"{len(full_levels[variable])} observed categories; exactly two are required."
                    )
            else:
                raise ModelError(
                    f"{variable!r} has unsupported analysis type {kind!r}. "
                    "Identifiers and dates must be excluded or explicitly transformed first."
                )

        if family == "logistic":
            labels = full_levels[outcome]
            chosen_event = str(event) if event is not None else labels[-1]
            if chosen_event not in labels:
                raise ModelError(f"The selected outcome event {chosen_event!r} is not observed.")
            info["event"] = chosen_event
            info["non_event"] = next(x for x in labels if x != chosen_event)

        for variable in predictors:
            if types[variable] in CATEGORICAL_TYPES:
                labels = full_levels[variable]
                if len(labels) < 2:
                    raise ModelError(f"Predictor {variable!r} has fewer than two observed categories.")
                reference = str(references[variable]) if variable in references else labels[0]
                if reference not in labels:
                    raise ModelError(f"Reference {reference!r} is not observed in {variable!r}.")
                info["references"][variable] = reference

        complete = work.notna().all(axis=1).to_numpy()
        n = int(complete.sum())
        result.update(n_complete=n, n_dropped=int(len(work) - n))
        info["missing_by_variable"] = {c: int(work[c].isna().sum()) for c in requested}
        if missing == "raise" and not complete.all():
            raise ModelError(
                f"{result['n_dropped']} row(s) have missing values in the requested model. "
                "Choose complete-case analysis explicitly or resolve missingness first."
            )
        if n < 3:
            raise ModelError("Fewer than three complete observations remain for this model.")
        used = work.loc[complete].copy()
        row_numbers = np.flatnonzero(complete) + 1
        if family == "linear":
            y = used[outcome].to_numpy(dtype=float)
            if np.ptp(y) == 0:
                raise ModelError("The outcome is constant in the analysis sample.")
        else:
            y = (used[outcome] == info["event"]).to_numpy(dtype=float)
            if len(np.unique(y)) != 2:
                raise ModelError("Only one outcome class remains after complete-case exclusion.")

        # Count treatment-coded columns without allocating their arrays. A
        # nominal identifier with n unique labels would otherwise create an
        # n-by-n matrix before being rejected for having no residual degrees
        # of freedom. Keep every requested level in this count.
        p = 1 + sum(
            1 if types[variable] == "continuous" else len(full_levels[variable]) - 1
            for variable in predictors
        )
        info["parameters_including_intercept"] = int(p)
        info["design_cells"] = int(n * p)
        info["design_cell_limit"] = MAX_DESIGN_CELLS
        if n <= p:
            raise ModelError(
                f"Only {n} complete observations are available for {p} parameters including "
                "the intercept. More observations or a smaller prespecified model are required. "
                "Check high-cardinality categorical predictors; identifiers should be excluded. "
                "No predictors or categories were removed."
            )
        if n * p > MAX_DESIGN_CELLS:
            raise ModelError(
                f"The requested encoded design needs {n * p:,} cells ({n:,} observations "
                f"by {p:,} parameters), exceeding this app's {MAX_DESIGN_CELLS:,}-cell limit. "
                "Choose a smaller prespecified model or review high-cardinality categorical "
                "predictors; identifiers should be excluded. No predictors or categories were removed."
            )

        columns = [np.ones(n)]
        term_metadata = []
        blocks: dict[str, list[int]] = {}
        for variable in predictors:
            blocks[variable] = []
            if types[variable] == "continuous":
                values = used[variable].to_numpy(dtype=float)
                if np.ptp(values) == 0:
                    raise ModelError(f"Predictor {variable!r} is constant in the analysis sample.")
                blocks[variable].append(len(columns))
                columns.append(values)
                term_metadata.append((variable, variable, ""))
            else:
                observed = set(used[variable].tolist())
                reference = info["references"][variable]
                if reference not in observed:
                    raise ModelError(
                        f"Reference category {reference!r} for {variable!r} is absent in "
                        "the complete-case sample. Choose a present reference explicitly."
                    )
                absent_levels = set(full_levels[variable]).difference(observed)
                if absent_levels:
                    raise ModelError(
                        f"Category/categories {', '.join(sorted(absent_levels))} of {variable!r} "
                        "are absent in the complete-case sample; their effects cannot be estimated. "
                        "Review missingness or explicitly revise the analysis variables/categories."
                    )
                for level in full_levels[variable]:
                    if level != reference:
                        blocks[variable].append(len(columns))
                        columns.append((used[variable] == level).to_numpy(dtype=float))
                        term_metadata.append((variable, f"{variable}: {level} vs {reference}", reference))

        x = np.column_stack(columns)
        scaled = _scaled_design(x)
        if np.linalg.matrix_rank(scaled) < p:
            raise ModelError(
                "The design matrix is rank deficient: selected predictors or category columns "
                "are exact linear combinations. Review redundant predictors; none were removed."
            )
        # Scaling for a rank diagnostic does not fix the raw matrix passed to
        # statsmodels. Its pseudoinverse can otherwise silently discard a slope
        # when, for example, a predictor has an enormous constant offset. Refuse
        # such numerical rank loss instead of reporting a spurious coefficient.
        if np.linalg.matrix_rank(x) < p:
            raise ModelError(
                "The raw design matrix loses numerical rank because of predictor units "
                "or large offsets. Center/rescale continuous predictors and fit again. "
                "No predictors were removed or unreliable estimates reported."
            )
        condition = float(np.linalg.cond(scaled))
        if condition > 1e4:
            result["warnings"].append(
                "The scaled design matrix is poorly conditioned; estimates may be numerically "
                "unstable. Review highly correlated or near-redundant predictors."
            )
        if n - p < 10:
            result["warnings"].append(
                "Fewer than 10 residual degrees of freedom remain. Estimates and uncertainty "
                "may be unstable; consider a smaller prespecified model or more observations."
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if family == "linear":
                fitted = sm.OLS(y, x, missing="raise").fit(cov_type=cov_type, use_t=True)
                if fitted.model.rank != p:
                    raise ModelError(
                        "The fitting routine lost numerical rank. Center/rescale continuous "
                        "predictors and fit again; no unreliable estimates were reported."
                    )
                outcome_variance = float(np.var(y))
                if outcome_variance == 0:
                    raise ModelError(
                        "The outcome variance is below numerical precision. Rescale the "
                        "outcome to more suitable measurement units before fitting."
                    )
                # Judge the residual variation relative to outcome variation:
                # an absolute threshold would wrongly reject valid small-unit data.
                if fitted.mse_resid <= np.finfo(float).eps * outcome_variance:
                    raise ModelError(
                        "The linear model has essentially no residual variation. "
                        "Reliable standard errors and significance tests cannot be estimated."
                    )
            else:
                separated, separation_warning = _separation_check(x, y)
                if separated:
                    raise ModelError(
                        "Complete or quasi-complete separation was detected. Ordinary logistic "
                        "maximum-likelihood coefficients are not finite/reliable. Review sparse "
                        "categories or use a specialist penalized/Firth approach; predictors were not removed."
                    )
                if separation_warning:
                    result["warnings"].append(separation_warning)
                fitted = sm.Logit(y, x, missing="raise").fit(disp=False, maxiter=200)
                if not bool(fitted.mle_retvals.get("converged", False)):
                    raise ModelError(
                        "Logistic regression did not converge. Review separation, sparse "
                        "categories, predictor scaling, and model complexity."
                    )
                if any("PerfectSeparation" in warning.category.__name__ for warning in caught):
                    raise ModelError(
                        "The fitting routine detected perfect prediction/separation. "
                        "Ordinary logistic estimates are not reliable."
                    )

            beta = np.asarray(fitted.params, dtype=float)
            se = np.asarray(fitted.bse, dtype=float)
            covariance = np.asarray(fitted.cov_params(), dtype=float)
            intervals = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)
            p_values = np.asarray(fitted.pvalues, dtype=float)
            if (
                not np.isfinite(beta).all() or not np.isfinite(se).all()
                or (se <= 0).any() or not np.isfinite(covariance).all()
                or not np.isfinite(intervals).all() or not np.isfinite(p_values).all()
            ):
                raise ModelError(
                    "Nonfinite coefficients or unreliable standard errors were produced. "
                    "Review separation, sample size, scaling, and collinearity."
                )

            variable_rows = []
            for variable, indices in blocks.items():
                r = np.eye(p)[indices, :]
                subcov = covariance[np.ix_(indices, indices)]
                corr = subcov / np.outer(se[indices], se[indices])
                if np.linalg.matrix_rank(corr) < len(indices):
                    raise ModelError(f"The joint covariance for {variable!r} is singular.")
                if family == "linear":
                    joint = fitted.f_test(r)
                    joint_p = float(np.asarray(joint.pvalue).item())
                    test_name = "Joint F test" if cov_type == "nonrobust" else "HC3 joint Wald F test"
                else:
                    joint = fitted.wald_test(r, use_f=False, scalar=True)
                    joint_p = float(np.asarray(joint.pvalue).item())
                    test_name = "Joint Wald chi-square test"
                if not np.isfinite(joint_p):
                    raise ModelError(f"A reliable overall p-value for {variable!r} could not be computed.")
                variable_rows.append({
                    "variable": variable, "p_value": joint_p,
                    "df": len(indices), "method": test_name,
                })

            predicted = np.asarray(fitted.predict(x), dtype=float)
            residual = y - predicted
            diagnostics = pd.DataFrame({
                "row_number": row_numbers, "observed": y,
                "fitted": predicted, "residual": residual,
            })
            metrics: dict[str, Any] = {
                "parameters_including_intercept": int(p),
                "residual_df": int(n - p), "scaled_condition_number": condition,
                "aic": float(fitted.aic), "bic": float(fitted.bic),
            }
            if family == "linear":
                influence = fitted.get_influence()
                diagnostics["std_residual"] = influence.resid_studentized_internal
                diagnostics["leverage"] = influence.hat_matrix_diag
                diagnostics["cooks_distance"] = influence.cooks_distance[0]
                metrics.update(
                    r_squared=float(fitted.rsquared),
                    adjusted_r_squared=float(fitted.rsquared_adj),
                    rmse=float(np.sqrt(np.mean(residual ** 2))),
                    residual_sd=float(np.sqrt(fitted.mse_resid)),
                )
                n_influential = int((diagnostics["cooks_distance"] > 4.0 / n).sum())
                metrics["cooks_distance_above_4_over_n"] = n_influential
                if n_influential:
                    result["warnings"].append(
                        f"{n_influential} observation(s) exceed the exploratory Cook's distance "
                        "threshold 4/n. Inspect their context; no observations were removed."
                    )
            else:
                clipped = np.clip(predicted, np.finfo(float).eps, 1 - np.finfo(float).eps)
                deviance = -2.0 * (y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
                diagnostics["deviance_residual"] = np.sign(residual) * np.sqrt(deviance)
                events = int(y.sum())
                non_events = n - events
                prop = events / n
                ll_null = events * np.log(prop) + non_events * np.log1p(-prop)
                ranks = stats.rankdata(predicted)
                auc = (ranks[y == 1].sum() - events * (events + 1) / 2) / (events * non_events)
                metrics.update(
                    events=events, non_events=non_events,
                    log_likelihood=float(fitted.llf),
                    mcfadden_pseudo_r_squared=float(1.0 - fitted.llf / ll_null),
                    auc_apparent=float(auc), brier_score_apparent=float(np.mean(residual ** 2)),
                    min_class_per_slope_parameter=float(min(events, non_events) / (p - 1)),
                    converged=True,
                )
                if min(events, non_events) / (p - 1) < 10:
                    result["warnings"].append(
                        "Fewer than 10 observations in the smaller outcome class per slope "
                        "parameter: sparse data/overfitting may make Wald inference unreliable. "
                        "This heuristic is not a sample-size guarantee."
                    )
                if ((predicted < 1e-6) | (predicted > 1 - 1e-6)).any():
                    result["warnings"].append(
                        "Some fitted probabilities are extremely close to 0 or 1. "
                        "Inspect sparse categories and unstable estimates."
                    )

            vif_rows = []
            for i, (variable, term, _) in enumerate(term_metadata, start=1):
                vif_rows.append({"variable": variable, "term": term,
                                 "vif": float(variance_inflation_factor(x, i))})
            vif = pd.DataFrame(vif_rows)
            if (vif["vif"] > 5).any():
                result["warnings"].append(
                    "Some individual design-column VIFs exceed 5. Review collinearity; "
                    "dummy-column VIFs are not generalized VIFs for whole categorical variables."
                )

            rows = []
            for i, (variable, term, reference) in enumerate(term_metadata, start=1):
                estimate = beta[i]
                low, high = intervals[i]
                if family == "logistic":
                    with np.errstate(over="ignore", under="ignore"):
                        estimate, low, high = np.exp([estimate, low, high])
                    if not np.isfinite([estimate, low, high]).all() or min(estimate, low, high) <= 0:
                        raise ModelError(
                            "An odds ratio or confidence bound is outside numerical range. "
                            "The logistic estimates are too unstable to report reliably."
                        )
                rows.append({
                    "variable": variable, "term": term, "reference": reference,
                    "estimate": float(estimate), "ci_lower": float(low), "ci_upper": float(high),
                    "p_value": float(p_values[i]), "coefficient": float(beta[i]),
                    "std_error": float(se[i]),
                })

            info["intercept"] = {
                "coefficient": float(beta[0]), "std_error": float(se[0]),
                "ci_lower": float(intervals[0, 0]), "ci_upper": float(intervals[0, 1]),
                "p_value": float(p_values[0]), "scale": "linear" if family == "linear" else "log odds",
            }
            info["diagnostic_caveat"] = (
                "Diagnostics and performance describe the model's estimation data, not "
                "external validation. Model assumptions require subject-matter assessment."
            )
            result.update(
                success=True, terms=pd.DataFrame(rows, columns=TERM_COLUMNS),
                variables=pd.DataFrame(variable_rows, columns=VARIABLE_COLUMNS),
                metrics=metrics, diagnostics=diagnostics, vif=vif,
            )

        for warning in caught:
            message = str(warning.message)
            if message not in result["warnings"]:
                result["warnings"].append(message)
        return result

    except (ModelError, ValueError, TypeError, KeyError, np.linalg.LinAlgError,
            FloatingPointError, OverflowError, ZeroDivisionError,
            MissingDataError, PerfectSeparationError) as exc:
        result["error"] = str(exc)
        result["success"] = False
        result["terms"] = pd.DataFrame(columns=TERM_COLUMNS)
        result["variables"] = pd.DataFrame(columns=VARIABLE_COLUMNS)
        return result
