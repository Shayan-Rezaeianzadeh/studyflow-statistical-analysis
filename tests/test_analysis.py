"""Numerical reference checks and safeguards for independent-row analyses."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm

from analysis.bivariate import run_bivariate, screen_variables
from analysis.cleaning import clean_data, infer_types, read_dataset
from analysis.comparisons import compare_groups
from analysis.descriptive import descriptive_statistics
from analysis.regression import fit_model


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(814)
    x = rng.normal(size=120)
    z = rng.normal(size=120)
    group = np.tile(["A", "B", "C"], 40)
    y = 2 + 1.5 * x - 0.4 * z + 1.1 * (group == "B") - 1.6 * (group == "C") + rng.normal(size=120)
    return pd.DataFrame({"y": y, "x": x, "z": z, "group": group})


@pytest.fixture
def binary_table():
    rows = []
    for exposure, event, count in [
        ("No", "Yes", 10), ("No", "No", 40),
        ("Yes", "Yes", 30), ("Yes", "No", 20),
    ]:
        rows.extend([{"exposure": exposure, "event": event}] * count)
    return pd.DataFrame(rows)


def test_linear_slope_ci_and_p_match_closed_form(linear_data):
    result = fit_model(linear_data, "y", ["x"], {"y": "continuous", "x": "continuous"})
    assert result["success"], result["error"]
    row = result["terms"].iloc[0]
    x = linear_data.x.to_numpy()
    y = linear_data.y.to_numpy()
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    intercept = y.mean() - slope * x.mean()
    df = len(x) - 2
    se = np.sqrt(np.sum((y - intercept - slope * x) ** 2) / df / np.sum((x - x.mean()) ** 2))
    critical = stats.t.ppf(0.975, df)
    assert row.estimate == pytest.approx(slope, rel=1e-10)
    assert row.ci_lower == pytest.approx(slope - critical * se, rel=1e-10)
    assert row.ci_upper == pytest.approx(slope + critical * se, rel=1e-10)
    assert row.p_value == pytest.approx(2 * stats.t.sf(abs(slope / se), df), rel=1e-8)
    assert result["n_complete"] == 120


def test_hc3_matches_reference_covariance(linear_data):
    result = fit_model(linear_data, "y", ["x", "z"],
                       {"y": "continuous", "x": "continuous", "z": "continuous"}, cov_type="HC3")
    assert result["success"], result["error"]
    reference = sm.OLS(linear_data.y, sm.add_constant(linear_data[["x", "z"]])).fit(cov_type="HC3", use_t=True)
    for _, row in result["terms"].iterrows():
        assert row.std_error == pytest.approx(reference.bse[row.variable], rel=1e-9)
        assert row.p_value == pytest.approx(reference.pvalues[row.variable], rel=1e-8)
        np.testing.assert_allclose([row.ci_lower, row.ci_upper], reference.conf_int().loc[row.variable], rtol=1e-8)


def test_categorical_joint_test_equals_oneway_anova(linear_data):
    result = fit_model(linear_data, "y", ["group"], {"y": "continuous", "group": "nominal"},
                       references={"group": "A"})
    assert result["success"], result["error"]
    reference = stats.f_oneway(*(linear_data.loc[linear_data.group == group, "y"] for group in ["A", "B", "C"]))
    assert len(result["terms"]) == 2
    assert len(result["variables"]) == 1
    assert result["variables"].iloc[0].p_value == pytest.approx(reference.pvalue, rel=1e-9)
    assert result["variables"].iloc[0]["df"] == 2


def test_logistic_odds_ratio_and_ci_match_two_by_two(binary_table):
    result = fit_model(binary_table, "event", ["exposure"],
                       {"event": "binary", "exposure": "binary"},
                       event="Yes", references={"exposure": "No"})
    assert result["success"], result["error"]
    row = result["terms"].iloc[0]
    log_or = np.log((30 / 20) / (10 / 40))
    se = np.sqrt(1 / 30 + 1 / 20 + 1 / 10 + 1 / 40)
    assert row.estimate == pytest.approx(6, rel=1e-8)
    assert row.ci_lower == pytest.approx(np.exp(log_or - stats.norm.ppf(0.975) * se), rel=1e-8)
    assert row.ci_upper == pytest.approx(np.exp(log_or + stats.norm.ppf(0.975) * se), rel=1e-8)
    assert set(np.round(result["diagnostics"].fitted, 8)) == {0.2, 0.6}


@pytest.mark.parametrize("event,reference", [("No", "No"), ("Yes", "Yes")])
def test_logistic_event_or_reference_reversal_reciprocates_or(binary_table, event, reference):
    result = fit_model(binary_table, "event", ["exposure"],
                       {"event": "binary", "exposure": "binary"},
                       event=event, references={"exposure": reference})
    assert result["success"], result["error"]
    assert result["terms"].iloc[0].estimate == pytest.approx(1 / 6, rel=1e-8)


def test_complete_case_counts_are_model_specific_and_original_unchanged(linear_data):
    data = linear_data.copy()
    data.loc[:3, "x"] = np.nan
    data.loc[4:8, "y"] = np.nan
    original = data.copy(deep=True)
    result = fit_model(data, "y", ["x"], {"y": "continuous", "x": "continuous"})
    assert result["success"], result["error"]
    assert (result["n_total"], result["n_complete"], result["n_dropped"]) == (120, 111, 9)
    pd.testing.assert_frame_equal(data, original)
    assert len(result["diagnostics"]) == 111
    assert result["diagnostics"].row_number.min() == 10


def test_require_complete_data_refuses_missingness(linear_data):
    data = linear_data.copy()
    data.loc[0, "x"] = np.nan
    result = fit_model(data, "y", ["x"], {"y": "continuous", "x": "continuous"}, missing="raise")
    assert not result["success"]
    assert "missing" in result["error"].lower()


@pytest.mark.parametrize("bad", [np.inf, "not a number"])
def test_invalid_continuous_data_cannot_be_silently_discarded(linear_data, bad):
    data = linear_data.copy()
    data["x"] = data.x.astype(object)
    data.loc[0, "x"] = bad
    result = fit_model(data, "y", ["x"], {"y": "continuous", "x": "continuous"})
    assert not result["success"]
    assert result["error"]
    assert result["terms"].empty


def test_rank_deficiency_refuses_to_drop_predictors(linear_data):
    data = linear_data.assign(duplicate_x=2 * linear_data.x)
    result = fit_model(data, "y", ["x", "duplicate_x"],
                       {"y": "continuous", "x": "continuous", "duplicate_x": "continuous"})
    assert not result["success"]
    assert any(word in result["error"].lower() for word in ["rank", "collinear", "dependent"])
    assert result["terms"].empty


@pytest.mark.parametrize("x", [np.repeat(1.0, 120), np.repeat("A", 120)])
def test_constant_predictor_is_a_visible_failure(linear_data, x):
    kind = "continuous" if x.dtype.kind == "f" else "nominal"
    result = fit_model(linear_data.assign(x=x), "y", ["x"], {"y": "continuous", "x": kind})
    assert not result["success"]
    assert result["error"]


def test_reference_absent_after_complete_cases_is_not_switched(linear_data):
    data = linear_data.copy()
    data.loc[data.group == "A", "y"] = np.nan
    result = fit_model(data, "y", ["group"], {"y": "continuous", "group": "nominal"},
                       references={"group": "A"})
    assert not result["success"]
    assert "reference" in result["error"].lower()


@pytest.mark.parametrize("x,y", [
    ([-3, -2, -1, 1, 2, 3] * 5, ["No", "No", "No", "Yes", "Yes", "Yes"] * 5),
    ([-2, -1, 0, 0, 1, 2] * 5, ["No", "No", "No", "Yes", "Yes", "Yes"] * 5),
])
def test_complete_and_quasi_separation_have_no_published_estimates(x, y):
    result = fit_model(pd.DataFrame({"x": x, "y": y}), "y", ["x"],
                       {"y": "binary", "x": "continuous"}, event="Yes")
    assert not result["success"]
    assert "separat" in result["error"].lower()
    assert result["terms"].empty


def test_unusual_column_names_are_not_treated_as_formula_code(linear_data):
    data = linear_data.rename(columns={"y": "outcome mg/dL", "x": "age (years)"})
    result = fit_model(data, "outcome mg/dL", ["age (years)"],
                       {"outcome mg/dL": "continuous", "age (years)": "continuous"})
    assert result["success"], result["error"]
    assert result["terms"].iloc[0].variable == "age (years)"


def test_bivariate_joint_rows_and_single_effect_policy(linear_data):
    result = run_bivariate(linear_data, "y", ["x", "group"],
                           {"y": "continuous", "x": "continuous", "group": "nominal"})
    summary = result["variables"].set_index("variable")
    assert len(summary) == 2
    assert np.isfinite(summary.loc["x", "estimate"])
    assert pd.isna(summary.loc["group", "estimate"])
    assert np.isfinite(summary.loc["group", "p_value"])
    assert len(result["terms"].query("variable == 'group'")) == 2


def test_bivariate_common_sample_and_per_model_samples(linear_data):
    data = linear_data.copy()
    data.loc[:3, "x"] = np.nan
    data.loc[4:9, "z"] = np.nan
    types = {"y": "continuous", "x": "continuous", "z": "continuous"}
    separate = run_bivariate(data, "y", ["x", "z"], types)
    common = run_bivariate(data, "y", ["x", "z"], types, common_sample=True)
    assert separate["variables"].set_index("variable").loc["x", "n"] == 116
    assert separate["variables"].set_index("variable").loc["z", "n"] == 114
    assert set(common["variables"]["n"]) == {110}


def test_bivariate_keeps_a_failed_predictor_row(linear_data):
    data = linear_data.assign(constant=1)
    result = run_bivariate(data, "y", ["x", "constant"],
                           {"y": "continuous", "x": "continuous", "constant": "continuous"})
    failed = result["variables"].set_index("variable").loc["constant"]
    assert pd.isna(failed.p_value)
    assert failed.reason
    assert "constant" not in screen_variables(result, 0.25)


def test_screening_uses_joint_p_and_strict_thresholds():
    result = {
        "variables": pd.DataFrame({
            "variable": ["category", "at_threshold", "small", "failed"],
            "p_value": [0.21, 0.05, 0.001, np.nan],
            "status": ["ok", "ok", "ok", "failed"],
        }),
        "terms": pd.DataFrame({"variable": ["category"], "p_value": [0.00001]}),
    }
    assert screen_variables(result, 0.05) == ["small"]
    assert screen_variables(result, 0.25) == ["category", "at_threshold", "small"]


def test_upload_preserves_missing_tokens_and_identifier_text():
    data = read_dataset(BytesIO(b"id,age,label\n001,19,NA\n002,,Yes\n"), filename="example.csv")
    assert data.loc[0, "label"] == "NA"
    assert data.loc[0, "id"] == "001"
    assert data.loc[1, "age"] == ""


def test_excel_upload_round_trip():
    expected = pd.DataFrame({"id": ["001", "002"], "label": ["NA", "Yes"]})
    stream = BytesIO()
    expected.to_excel(stream, index=False)
    stream.seek(0)
    data = read_dataset(stream, filename="example.xlsx")
    assert data.loc[0, "id"] == "001"
    assert data.loc[0, "label"] == "NA"


def test_cleaning_requires_explicit_duplicate_removal_and_preserves_input():
    data = pd.DataFrame({"age": ["20", "NA", "20"], "group": ["A", "B", "A"]})
    original = data.copy(deep=True)
    types = {"age": "continuous", "group": "binary"}
    retained = clean_data(data, types)
    assert not retained["errors"]
    assert len(retained["data"]) == 3
    assert retained["data"].age.isna().sum() == 1
    removed = clean_data(data, types, options={"drop_duplicates": True})
    assert not removed["errors"]
    assert len(removed["data"]) == 2
    assert removed["impact"]["duplicate_rows_removed"] == 1
    assert removed["log"]
    pd.testing.assert_frame_equal(data, original)


def test_invalid_conversion_requires_opt_in():
    data = pd.DataFrame({"age": ["20", "not recorded", "21"]})
    blocked = clean_data(data, {"age": "continuous"})
    assert blocked["errors"]
    allowed = clean_data(data, {"age": "continuous"}, options={"coerce_invalid": True})
    assert not allowed["errors"]
    assert allowed["data"].age.isna().sum() == 1
    assert allowed["impact"]["invalid_values_coerced"] == 1


def test_recode_and_ordinal_order_are_explicit():
    data = pd.DataFrame({"severity": ["mod", "Mild", "Severe", "mod"]})
    result = clean_data(data, {"severity": "ordinal"},
                        options={"recodes": {"severity": {"mod": "Moderate"}}},
                        ordinal_orders={"severity": ["Mild", "Moderate", "Severe"]})
    assert not result["errors"]
    assert list(result["data"].severity.astype(str)) == ["Moderate", "Mild", "Severe", "Moderate"]
    description = descriptive_statistics(result["data"], result["types"],
                                         category_orders={"severity": ["Mild", "Moderate", "Severe"]})
    assert description["categorical"].category.tolist() == ["Mild", "Moderate", "Severe"]


def test_invalid_ordinal_order_is_not_silently_applied():
    result = clean_data(pd.DataFrame({"severity": ["Mild", "Severe"]}), {"severity": "ordinal"},
                        ordinal_orders={"severity": ["Mild"]})
    assert result["errors"]


def test_descriptive_statistics_missingness_and_nonmissing_denominators():
    data = pd.DataFrame({"age": [10.0, 20.0, 30.0, np.nan], "group": ["A", "A", "B", None]})
    result = descriptive_statistics(data, {"age": "continuous", "group": "binary"})
    continuous = result["continuous"].iloc[0]
    assert continuous["n"] == 3
    assert continuous["missing"] == 1
    assert continuous.missing_percent == pytest.approx(25)
    assert continuous["mean"] == pytest.approx(20)
    assert continuous.sd == pytest.approx(10)
    categorical = result["categorical"].set_index("category")
    assert categorical.loc["A", "n"] == 2
    assert categorical.loc["A", "percent"] == pytest.approx(200 / 3)
    assert categorical["percent"].sum() == pytest.approx(100)


def test_welch_comparison_matches_scipy():
    a = [1.0, 2, 3, 8, 10]
    b = [2.0, 3, 4, 5, 6, 6, 7]
    data = pd.DataFrame({"sex": ["A"] * len(a) + ["B"] * len(b), "value": a + b})
    result = compare_groups(data, {"sex": "binary", "value": "continuous"}, "sex", ["value"])
    row = result["tests"].iloc[0]
    assert row.p_value == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue, rel=1e-10)
    assert "welch" in row.method.lower()
    assert row["n"] == len(data)


def test_rank_comparison_matches_scipy():
    a = [1.0, 2, 2, 3, 9]
    b = [2.0, 3, 4, 6, 6, 7]
    data = pd.DataFrame({"group": ["A"] * len(a) + ["B"] * len(b), "value": a + b})
    result = compare_groups(data, {"group": "binary", "value": "continuous"}, "group", ["value"], continuous_test="rank")
    assert result["tests"].iloc[0].p_value == pytest.approx(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue, rel=1e-10)


def test_sparse_two_by_two_uses_fisher():
    data = pd.DataFrame({"sex": ["A"] * 5 + ["B"] * 5,
                         "label": ["Yes"] * 1 + ["No"] * 4 + ["Yes"] * 4 + ["No"] * 1})
    result = compare_groups(data, {"sex": "binary", "label": "binary"}, "sex", ["label"])
    row = result["tests"].iloc[0]
    assert "fisher" in row.method.lower()
    assert row.p_value == pytest.approx(stats.fisher_exact([[1, 4], [4, 1]]).pvalue)


def test_sparse_larger_table_reports_unreliable_test_without_p_value():
    data = pd.DataFrame({"sex": ["A", "A", "B", "B", "C", "C"],
                         "label": ["Yes", "No", "Yes", "No", "Yes", "No"]})
    result = compare_groups(data, {"sex": "nominal", "label": "binary"}, "sex", ["label"])
    row = result["tests"].iloc[0]
    assert pd.isna(row.p_value)
    assert row.note


def test_missing_group_excluded_from_tests_but_overall_retained():
    data = pd.DataFrame({"sex": ["A", "A", "A", "B", "B", "B", None],
                         "value": [1.0, 2, 3, 4, 5, 6, 100]})
    result = compare_groups(data, {"sex": "binary", "value": "continuous"}, "sex", ["value"])
    assert result["tests"].iloc[0]["n"] == 6
    # Overall mean includes the seventh record, while the comparison does not.
    assert any("17.29" in str(value) or "17.3" in str(value) for value in result["table"].Overall)


def test_synthetic_example_has_expected_size_duplicates_and_missingness():
    path = Path(__file__).resolve().parents[1] / "sample_data" / "example_dataset.csv"
    data = pd.read_csv(path)
    assert len(data) == 504
    assert data.duplicated().sum() == 4
    assert data.isna().any().any()
    assert data.participant_id.str.startswith("SYN").all()
    assert set(data.outcome.dropna()) == {"No", "Yes"}
    assert infer_types(data)["participant_id"] == "identifier"
