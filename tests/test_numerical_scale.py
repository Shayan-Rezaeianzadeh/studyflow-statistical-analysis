"""Protect inference against unit changes and numerical rank loss."""

import numpy as np
import pandas as pd
import pytest

from analysis import regression
from analysis.regression import fit_model


def _dataset():
    rng = np.random.default_rng(5)
    x = rng.normal(size=300)
    noise = rng.normal(size=300)
    return pd.DataFrame({"x": x, "y": 1 + 0.3 * x + noise})


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC3"])
def test_small_outcome_units_preserve_estimates_and_inference(cov_type):
    """Changing units must scale slopes/CIs, not spuriously erase residual variance."""
    data = _dataset()
    types = {"x": "continuous", "y": "continuous"}
    baseline = fit_model(data, "y", ["x"], types, cov_type=cov_type)
    small_units = data.copy()
    small_units["y"] *= 1e-10
    scaled = fit_model(small_units, "y", ["x"], types, cov_type=cov_type)

    assert baseline["success"], baseline["error"]
    assert scaled["success"], scaled["error"]
    for column in ("estimate", "ci_lower", "ci_upper", "std_error"):
        np.testing.assert_allclose(
            scaled["terms"][column].to_numpy() / 1e-10,
            baseline["terms"][column].to_numpy(), rtol=1e-10,
        )
    np.testing.assert_allclose(
        scaled["terms"]["p_value"], baseline["terms"]["p_value"], rtol=1e-10,
    )
    assert scaled["metrics"]["r_squared"] == pytest.approx(
        baseline["metrics"]["r_squared"], abs=1e-12,
    )


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC3"])
def test_large_predictor_offset_does_not_report_a_silently_wrong_slope(cov_type):
    """A large-offset raw design must fail explicitly before pseudoinverse truncation."""
    data = _dataset()
    data["x"] += 1e12
    result = fit_model(data, "y", ["x"], {"x": "continuous", "y": "continuous"}, cov_type=cov_type)

    assert result["success"] is False
    assert "numerical rank" in result["error"]
    assert "rescale" in result["error"].lower()
    assert result["terms"].empty
    assert result["variables"].empty
    assert result["n_complete"] == len(data)
    assert result["n_dropped"] == 0


def test_unique_nominal_identifier_is_rejected_before_dummy_allocation(monkeypatch):
    """A mistakenly nominal ID must not trigger an n-by-n dummy allocation."""
    data = pd.DataFrame({
        "y": np.random.default_rng(10).normal(size=1000),
        "record": [f"record_{i}" for i in range(1000)],
    })

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("Dummy-matrix allocation must not be reached")

    monkeypatch.setattr(regression.np, "column_stack", forbidden_allocation)
    result = fit_model(data, "y", ["record"], {"y": "continuous", "record": "nominal"})

    assert result["success"] is False
    assert "1000 parameters" in result["error"]
    assert "high-cardinality" in result["error"]
    assert result["info"]["parameters_including_intercept"] == 1000
    assert result["terms"].empty


def test_design_cell_cap_rejects_before_allocation(monkeypatch):
    """Enforce the independent memory cap even when n exceeds parameter count."""
    data = _dataset()
    assert regression.MAX_DESIGN_CELLS == 10_000_000
    monkeypatch.setattr(regression, "MAX_DESIGN_CELLS", 500)

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("Over-limit design allocation must not be reached")

    monkeypatch.setattr(regression.np, "column_stack", forbidden_allocation)
    result = fit_model(data, "y", ["x"], {"x": "continuous", "y": "continuous"})

    assert result["success"] is False
    assert "600 cells" in result["error"]
    assert "500-cell limit" in result["error"]
    assert result["info"]["design_cells"] == 600
    assert result["info"]["design_cell_limit"] == 500
    assert result["terms"].empty


def test_design_exactly_at_cell_cap_is_allowed(monkeypatch):
    """The cap is inclusive and must not block an ordinary estimable fit."""
    data = _dataset()
    monkeypatch.setattr(regression, "MAX_DESIGN_CELLS", len(data) * 2)
    result = fit_model(data, "y", ["x"], {"x": "continuous", "y": "continuous"})

    assert result["success"], result["error"]
    assert result["info"]["design_cells"] == result["info"]["design_cell_limit"]
