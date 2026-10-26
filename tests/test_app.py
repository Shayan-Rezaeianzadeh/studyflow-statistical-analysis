"""Exercise real Streamlit widgets, result invalidation and export provenance."""

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"
STAGES = ["1. Upload & Clean Data", "2. Descriptive Statistics", "3. Gender Comparison",
          "4. Bivariate Analysis", "5. Variable Selection", "6. Multivariable Regression",
          "7. Results & Export"]


def assert_no_exception(app):
    assert not app.exception, [item.message for item in app.exception]


def goto(app, stage):
    app.radio(key="stage").set_value(STAGES[stage-1]).run()
    assert_no_exception(app)


@pytest.fixture
def example():
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    assert_no_exception(app)
    app.button(key="load_example").click().run()
    assert_no_exception(app)
    app.checkbox(key="w_drop_duplicates").check().run()
    app.button(key="apply_cleaning").click().run()
    assert_no_exception(app)
    assert app.session_state["project"]["cleaned"].shape == (500, 9)
    assert app.session_state["project"]["original"].shape == (504, 9)
    return app


def fit_example(app, linear=False):
    goto(app, 4)
    if linear:
        app.selectbox(key="w_analysis_outcome").set_value("blood_pressure").run()
        app.multiselect(key="w_analysis_predictors").set_value(["age", "bmi", "sex", "smoking", "severity"]).run()
        app.selectbox(key="w_analysis_cov").set_value("HC3").run()
    # The GUI allows explicit references, including an ordinal first category.
    for widget in app.selectbox:
        if widget.label == "Reference for smoking":
            widget.set_value("Never").run()
            break
    app.button(key="run_bivariate").click().run()
    assert_no_exception(app)
    assert app.session_state["project"]["bivariate"]["variables"]["status"].eq("ok").all()
    goto(app, 5)
    app.selectbox(key="w_selection_strategy").set_value("List B · p < 0.25").run()
    if "sex" not in app.multiselect(key="w_final_predictors").value:
        app.multiselect(key="w_final_predictors").select("sex").run()
    goto(app, 6)
    if linear:
        app.selectbox(key="w_model_cov").set_value("HC3").run()
    app.button(key="fit_model").click().run()
    assert_no_exception(app)
    assert app.session_state["project"]["model"]["success"]


def test_logistic_wizard_navigation_group_plot_and_report(example):
    app = example
    goto(app, 2)
    assert app.dataframe
    goto(app, 3)
    app.checkbox(key="w_comparison_tests").check().run()
    app.button(key="run_comparison").click().run()
    assert_no_exception(app)
    assert app.session_state["project"]["comparison"] is not None
    app.selectbox(key="w_comparison_plot_variable").set_value("bmi").run()
    assert app.session_state["project"]["comparison"] is not None
    goto(app, 2)
    goto(app, 3)
    assert app.session_state["project"]["comparison"] is not None
    assert app.selectbox(key="w_comparison_plot_variable").value == "bmi"
    fit_example(app)
    model = app.session_state["project"]["model"]
    assert model["n_complete"] == 429
    assert model["metrics"]["events"] == 125
    selected = app.session_state["project"]["selected_predictors"].copy()
    goto(app, 5)
    assert app.multiselect(key="w_final_predictors").value == selected
    assert app.session_state["project"]["model"] is not None
    goto(app, 7)
    app.button(key="prepare_report").click().run()
    assert_no_exception(app)
    payload = app.session_state["project"]["exports"]
    with ZipFile(BytesIO(payload)) as archive:
        assert "tables/Table3_Multivariable.csv" in archive.namelist()
        assert "figures/adjusted_forest.svg" in archive.namelist()
        summary = json.loads(archive.read("analysis_summary.json"))
        assert summary["multivariable"]["n_complete"] == 429
        assert summary["settings"]["selected_predictors"] == selected
        assert summary["settings"]["analysis"]["event"] == "Yes"
        assert summary["settings"]["original_variable_types"]["severity"] == "ordinal"
        assert "cleaned_data.csv" not in archive.namelist()


def test_linear_hc3_flow_and_outcome_change_invalidates_results(example):
    app = example
    fit_example(app, linear=True)
    model = app.session_state["project"]["model"]
    assert model["n_complete"] == 434
    assert model["info"]["cov_type"] == "HC3"
    assert model["metrics"]["r_squared"] == pytest.approx(0.364711, abs=1e-6)
    goto(app, 4)
    assert app.session_state["project"]["model"] is not None
    app.selectbox(key="w_analysis_outcome").set_value("outcome").run()
    assert_no_exception(app)
    state = app.session_state["project"]
    assert state["bivariate"] is None
    assert state["model"] is None
    assert state["selected_predictors"] is None
    assert any("Rerun" in item.value for item in app.info)
    goto(app, 7)
    assert any("partial analysis" in item.value for item in app.info)


def test_missing_policy_refuses_exclusions_with_visible_error(example):
    app = example
    fit_example(app)
    app.selectbox(key="w_model_missing").set_value("raise").run()
    assert app.session_state["project"]["model"] is None
    app.button(key="fit_model").click().run()
    assert_no_exception(app)
    model = app.session_state["project"]["model"]
    assert not model["success"]
    assert model["n_dropped"] == 71
    assert any("missing" in item.value.lower() for item in app.error)
    goto(app, 7)
    assert_no_exception(app)


def test_cleaning_drafts_do_not_modify_accepted_data_and_apply_invalidates(example):
    app = example
    fit_example(app)
    goto(app, 1)
    # An unapplied duplicate-removal change changes only the preview.
    app.checkbox(key="w_drop_duplicates").uncheck().run()
    state = app.session_state["project"]
    assert len(state["cleaned"]) == 500
    assert state["cleaning_options"]["drop_duplicates"] is True
    assert state["model"]["success"]
    goto(app, 7)
    summaries = [item.value for item in app.markdown if "# StudyFlow analysis summary" in item.value]
    assert len(summaries) == 1
    assert '"drop_duplicates": true' in summaries[0]
    goto(app, 1)
    assert not app.checkbox(key="w_drop_duplicates").value
    app.button(key="apply_cleaning").click().run()
    assert_no_exception(app)
    state = app.session_state["project"]
    assert len(state["cleaned"]) == 504
    assert state["bivariate"] is None and state["model"] is None
    assert state["revision"] == 2
