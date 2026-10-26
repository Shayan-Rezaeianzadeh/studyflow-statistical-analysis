"""StudyFlow: a seven-stage, local statistical analysis wizard.

Run with: streamlit run app.py
All patient/user data stays in this server process; no external service is used.
"""

from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis.cleaning import TYPE_LABELS, clean_data, infer_types, profile_data, read_dataset
from analysis.descriptive import descriptive_statistics
from analysis.comparisons import compare_groups
from analysis.bivariate import run_bivariate, screen_variables
from analysis.regression import fit_model
from analysis.visualization import distribution_plot, forest_plot, diagnostic_plot, figure_bytes
from analysis.export import (csv_bytes, excel_bytes, json_bytes, analysis_summary,
                             summary_markdown, bundle_bytes, interpretations)

BASE = Path(__file__).resolve().parent
STAGES = ["1. Upload & Clean Data", "2. Descriptive Statistics", "3. Gender Comparison",
          "4. Bivariate Analysis", "5. Variable Selection", "6. Multivariable Regression",
          "7. Results & Export"]
ANALYSIS_TYPES = {"continuous", "binary", "nominal", "ordinal"}
CAT_TYPES = {"binary", "nominal", "ordinal"}

st.set_page_config(page_title="StudyFlow · Statistical analysis", page_icon="📊", layout="wide")
st.markdown("""<style>
 .block-container {max-width:1250px; padding-top:2.5rem; padding-bottom:3rem;}
 h1 {letter-spacing:-.035em;} h2 {letter-spacing:-.02em;}
 [data-testid="stMetric"] {background:#f4f8fb; padding:1rem 1.2rem; border-radius:10px;}
 [data-testid="stSidebar"] {background:#f4f7fa;}
 .small-label {color:#087f8c; font-size:.78rem; font-weight:700; letter-spacing:.13em;}
 </style>""", unsafe_allow_html=True)


def fingerprint(value):
    return sha256(json_bytes(value)).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def record(state, action, detail, **extra):
    state.setdefault("log", []).append({"timestamp_utc": timestamp(), "action": action,
                                        "detail": detail, **extra})


def set_stage(index):
    st.session_state["stage"] = STAGES[index]


def next_stage(index, label=None, disabled=False):
    st.divider()
    st.button(label or f"Continue to {STAGES[index].split('. ', 1)[1]}", type="primary",
              key=f"next_{index}", on_click=set_stage, args=(index,), disabled=disabled)


def choose(label, options, cfg, field, *, default=None, key=None, **kwargs):
    options = list(options)
    if not options:
        return None
    widget = "w_" + (key or field)
    selected = cfg.get(field, default)
    if selected not in options:
        selected = default if default in options else options[0]
    if widget not in st.session_state or st.session_state[widget] not in options:
        st.session_state[widget] = selected
    value = st.selectbox(label, options, key=widget, **kwargs)
    cfg[field] = value
    return value


def multi(label, options, cfg, field, *, default=None, key=None, **kwargs):
    options = list(options)
    widget = "w_" + (key or field)
    saved = cfg.get(field, default or [])
    saved = [x for x in saved if x in options]
    if widget not in st.session_state:
        st.session_state[widget] = saved
    elif any(x not in options for x in st.session_state[widget]):
        st.session_state[widget] = [x for x in st.session_state[widget] if x in options]
    value = st.multiselect(label, options, key=widget, **kwargs)
    cfg[field] = value
    return value


def check(label, cfg, field, *, default=False, key=None, **kwargs):
    widget = "w_" + (key or field)
    if widget not in st.session_state:
        st.session_state[widget] = bool(cfg.get(field, default))
    value = st.checkbox(label, key=widget, **kwargs)
    cfg[field] = value
    return value


def text_input(label, cfg, field, *, default="", key=None, multiline=False, **kwargs):
    widget = "w_" + (key or field)
    if widget not in st.session_state:
        st.session_state[widget] = cfg.get(field, default)
    method = st.text_area if multiline else st.text_input
    value = method(label, key=widget, **kwargs)
    cfg[field] = value
    return value


def load_data(frame, raw, name, sheet=0, example=False):
    """Replace one working session only when the user loads another data source."""
    for key in list(st.session_state):
        if key.startswith("w_"):
            del st.session_state[key]
    guessed = infer_types(frame)
    orders = {}
    if example:
        guessed["participant_id"] = "identifier"
        guessed["severity"] = "ordinal"
        orders["severity"] = "Mild\nModerate\nSevere"
    state = {
        "original": frame.copy(deep=True),
        "source": {"name": name, "sha256": sha256(raw).hexdigest(), "rows": len(frame),
                   "columns": len(frame.columns), "sheet": sheet, "synthetic_example": example},
        "draft": {"types": guessed, "include_columns": list(frame.columns), "order_text": orders},
        "schema_version": 0, "revision": 0, "cleaned": None,
        "types": {}, "orders": {}, "cleaning": {}, "cleaning_options": {},
        "analysis": {}, "bivariate": None, "bivariate_signature": None,
        "selection": {}, "selected_predictors": None, "model": None,
        "model_settings": {}, "group_settings": {}, "comparison": None,
        "log": [], "exports": None,
    }
    record(state, "source_loaded", f"Loaded {name}", rows=len(frame), columns=len(frame.columns))
    st.session_state["project"] = state


def invalidate(state, *, bivariate=False, model=False, comparison=False, reason="Settings changed"):
    removed = []
    if bivariate:
        if state.get("bivariate") is not None:
            removed.append("bivariate results and variable selection")
        state["bivariate"] = None
        state["bivariate_signature"] = None
        state["selected_predictors"] = None
        state["selection"] = {}
        state.pop("selection_origin", None)
        # Discard widget state tied to obsolete lists before it is rendered again.
        for key in ("w_final_predictors", "w_selection_strategy"):
            st.session_state.pop(key, None)
        model = True
    if model:
        if state.get("model") is not None:
            removed.append("multivariable results")
        state["model"] = None
        state.pop("model_signature", None)
    if comparison:
        if state.get("comparison") is not None:
            removed.append("group comparisons")
        state["comparison"] = None
        state.pop("comparison_signature", None)
    state["exports"] = None
    if removed:
        message = reason + ". Rerun: " + ", ".join(removed) + "."
        st.info(message)
        record(state, "results_invalidated", message)


def need_cleaned(state):
    if not state or state.get("cleaned") is None:
        st.info("Load a dataset, review the cleaning preview, and click Apply cleaning in Stage 1.")
        st.button("Go to Upload & Clean Data", on_click=set_stage, args=(0,))
        return False
    return True


def available_variables(state):
    return [c for c in state["cleaned"] if state["types"].get(c) in ANALYSIS_TYPES]


def show_table(frame, name, *, downloads=True):
    configs = {}
    for col in frame.columns:
        if "p_value" in str(col) or col == "P-value":
            configs[col] = st.column_config.NumberColumn(str(col).replace("_", " "), format="%.4g")
        elif pd.api.types.is_float_dtype(frame[col]):
            configs[col] = st.column_config.NumberColumn(str(col).replace("_", " "), format="%.4g")
    st.dataframe(frame, hide_index=True, width="stretch", column_config=configs)
    if downloads:
        left, right, _ = st.columns([1,1,4])
        left.download_button("CSV", csv_bytes(frame), file_name=f"{name}.csv", mime="text/csv", key=f"csv_{name}")
        right.download_button("Excel", excel_bytes({name: frame}), file_name=f"{name}.xlsx",
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xlsx_{name}")


def display_figure(fig, name):
    st.pyplot(fig, width="stretch")
    cols = st.columns([1,1,4])
    cols[0].download_button("PNG · 300 dpi", figure_bytes(fig), file_name=f"{name}.png", mime="image/png", key=f"png_{name}")
    cols[1].download_button("SVG · vector", figure_bytes(fig, "svg"), file_name=f"{name}.svg", mime="image/svg+xml", key=f"svg_{name}")
    plt.close(fig)


def page_upload(state):
    st.header("Upload & clean data")
    st.write("Start with CSV or Excel. Review the preview before applying changes; the uploaded dataset is preserved separately.")
    left, right = st.columns([3,1], gap="large")
    with left:
        upload = st.file_uploader("Choose your dataset", type=["csv", "xlsx"], key="upload_source",
                                  help="A row represents an independent observation; the first row contains column names.")
        sheet = 0
        raw = None
        if upload is not None:
            raw = upload.getvalue()
            if upload.name.lower().endswith(".xlsx"):
                try:
                    sheets = pd.ExcelFile(BytesIO(raw), engine="openpyxl").sheet_names
                    sheet = st.selectbox("Excel worksheet", sheets, key="upload_sheet")
                except Exception as exc:
                    st.error(f"Cannot open this workbook: {exc}")
                    raw = None
            if st.button("Load uploaded dataset", disabled=raw is None, type="primary"):
                try:
                    frame = read_dataset(BytesIO(raw), filename=upload.name, sheet_name=sheet)
                    load_data(frame, raw, upload.name, sheet)
                    st.rerun()
                except (ValueError, TypeError, OSError, UnicodeError) as exc:
                    st.error(f"Could not load the dataset: {exc}")
    with right:
        st.markdown("#### Try the example")
        st.caption("Synthetic study data with continuous, binary and categorical variables, missing values and duplicate records.")
        if st.button("Load example dataset", key="load_example", width="stretch"):
            raw = (BASE / "sample_data" / "example_dataset.csv").read_bytes()
            frame = read_dataset(BytesIO(raw), filename="example_dataset.csv")
            load_data(frame, raw, "example_dataset.csv", example=True)
            st.rerun()
    if not state:
        st.info("Your data is processed by the local Python application. No account or database is needed.")
        return
    original = state["original"]
    draft = state["draft"]
    st.divider()
    st.subheader(state["source"]["name"])
    a,b,c = st.columns(3)
    a.metric("Uploaded rows", f"{len(original):,}")
    b.metric("Variables", len(original.columns))
    c.metric("Exact duplicate rows", int(original.duplicated().sum()))
    with st.expander("View original data", expanded=False):
        st.dataframe(original.head(100), width="stretch", hide_index=True)
        st.caption("Preview shows the first 100 rows. The original is never edited.")
    st.subheader("1 · Review variable types and inclusion")
    st.caption("Numeric category codes may look continuous. Verify types, especially binary outcomes and identifiers. Date/time and identifier fields are excluded from statistical models.")
    schema = pd.DataFrame({
        "Variable": original.columns,
        "Include": [c in draft["include_columns"] for c in original],
        "Type": [TYPE_LABELS[draft["types"][c]] for c in original],
    })
    edited = st.data_editor(schema, disabled=["Variable"], hide_index=True, width="stretch",
                            key=f"w_schema_{state['schema_version']}",
                            column_config={"Type": st.column_config.SelectboxColumn("Type", options=list(TYPE_LABELS.values()), required=True),
                                           "Include": st.column_config.CheckboxColumn("Include", required=True)})
    reverse_types = {v:k for k,v in TYPE_LABELS.items()}
    draft["types"] = dict(zip(edited["Variable"], edited["Type"].map(reverse_types)))
    draft["include_columns"] = edited.loc[edited["Include"], "Variable"].tolist()
    st.subheader("2 · Choose cleaning operations")
    col1,col2 = st.columns(2)
    with col1:
        blanks = check("Treat blank text as missing", draft, "blank_missing", default=True)
        missing_text = text_input("Other missing-value tokens (one per line)", draft, "missing_text",
                                  default="NA\nN/A\nNaN\nnull\nNone\n?", multiline=True, height=110)
        st.caption("Tokens are exact, case-sensitive matches. Remove a token if it is a real category in your study.")
        strip = check("Trim leading and trailing spaces", draft, "strip_whitespace")
    with col2:
        duplicates = check("Remove exact duplicate rows", draft, "drop_duplicates",
                           help="Duplicates are assessed after cleaning, using all retained columns. Keep an identifier included if it distinguishes separate records; its Identifier type still excludes it from analyses.")
        coerce = check("Allow invalid numeric/date values to become missing", draft, "coerce_invalid",
                       help="The preview reports conversion losses. This never imputes a value or removes an outlier.")
        st.caption("Conversions follow your selected variable types. Invalid values block Apply unless you explicitly allow conversion to missing.")
    with st.expander("Recode category labels (optional)"):
        st.caption("Add one row per replacement. Both labels must be nonempty. These edits remain a preview until you apply cleaning.")
        recode_base = pd.DataFrame(draft.get("recode_rows", []), columns=["Variable", "From", "To"])
        recode_frame = st.data_editor(recode_base, num_rows="dynamic", hide_index=True, width="stretch", key="w_recodes",
                                      column_config={"Variable": st.column_config.SelectboxColumn("Variable", options=list(original.columns), required=True)})
        draft["recode_rows"] = recode_frame.to_dict("records")
    recodes = {}
    local_errors = []
    for row in draft.get("recode_rows", []):
        vals = [row.get(k) for k in ("Variable", "From", "To")]
        if all(pd.isna(v) or str(v) == "" for v in vals):
            continue
        if any(pd.isna(v) or str(v) == "" for v in vals):
            local_errors.append("Complete all three fields in each recoding row, or delete the row.")
            continue
        col,old,new = [str(v) for v in vals]
        if old in recodes.setdefault(col, {}):
            local_errors.append(f"Repeated recoding rule for {col}: {old}.")
        recodes[col][old] = new
    options = {"missing_tokens": ([""] if blanks else []) + [s for s in missing_text.splitlines() if s],
               "strip_whitespace": strip, "drop_duplicates": duplicates,
               "include_columns": draft["include_columns"], "recodes": recodes, "coerce_invalid": coerce}
    orders = {}
    ordinal = [c for c in draft["include_columns"] if draft["types"][c] == "ordinal"]
    if ordinal:
        st.markdown("#### Ordinal category order")
        st.caption("Enter every observed category in the intended order, one per line. Regression uses category contrasts rather than an assumed linear score.")
        preliminary = clean_data(original, draft["types"], options)
        order_cfg = draft.setdefault("order_text", {})
        for col in ordinal:
            source = preliminary["data"][col] if col in preliminary["data"] else original[col]
            defaults = "\n".join(sorted(source.dropna().astype(str).unique().tolist()))
            order_text = text_input(f"Order for {col}", order_cfg, col, default=defaults,
                                    key="order_" + fingerprint(col)[:12], multiline=True, height=100)
            orders[col] = [x for x in order_text.splitlines() if x]
    preview = clean_data(original, draft["types"], options, ordinal_orders=orders)
    st.subheader("3 · Inspect the cleaning preview")
    imp = preview["impact"]
    a,b,c,d = st.columns(4)
    a.metric("Rows after cleaning", f"{imp.get('rows_after', len(preview['data'])):,}")
    b.metric("Rows affected", imp.get("rows_affected", 0))
    c.metric("Duplicates removed", imp.get("duplicate_rows_removed", 0))
    d.metric("Invalid values → missing", imp.get("invalid_values_coerced", 0))
    if imp.get("excluded_columns"):
        st.info("Variables marked for exclusion: " + ", ".join(imp["excluded_columns"]))
    for warning in preview["warnings"]:
        st.warning(warning)
    for error in local_errors + preview["errors"]:
        st.error(error)
    profile = profile_data(preview["data"], preview["types"])
    st.dataframe(profile, hide_index=True, width="stretch")
    if not profile.empty:
        with st.expander("Exclude variables above a missingness threshold"):
            threshold = st.slider("Missing values (%)", 0, 100, 50, key="w_missing_threshold")
            high = profile.loc[profile["missing_percent"] > threshold, "variable"].tolist()
            st.write("Above threshold: " + (", ".join(high) if high else "None"))
            if st.button("Mark these variables for exclusion", disabled=not high):
                draft["include_columns"] = [c for c in draft["include_columns"] if c not in high]
                state["schema_version"] += 1
                st.rerun()
    with st.expander("Preview cleaned rows and operation counts"):
        st.dataframe(preview["data"].head(100), width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame(preview["log"]), width="stretch", hide_index=True)
    clean_signature = fingerprint({"types": draft["types"], "options": options, "orders": orders})
    pending = clean_signature != state.get("cleaning_signature")
    if pending and state.get("cleaned") is not None:
        st.warning("This preview has unapplied changes. Later stages still use the last accepted cleaned dataset.")
    st.caption("Apply accepts the operations and type conversions shown above. No values are imputed and no outliers are removed.")
    if st.button("Apply cleaning", type="primary", key="apply_cleaning", disabled=bool(local_errors or preview["errors"])):
        if pending:
            invalidate(state, bivariate=True, comparison=True, reason="The cleaned dataset or variable types changed")
            state["cleaned"] = preview["data"].copy(deep=True)
            state["types"] = preview["types"].copy()
            state["orders"] = deepcopy(orders)
            state["cleaning"] = preview
            state["cleaning_options"] = deepcopy(options)
            state["cleaning_types"] = draft["types"].copy()
            state["cleaning_signature"] = clean_signature
            state["revision"] += 1
            record(state, "cleaning_applied", "Accepted the cleaning preview", **imp)
        st.success("Cleaned dataset accepted. Continue to descriptive statistics.")
    if state.get("cleaned") is not None:
        st.caption(f"Accepted dataset: {len(state['cleaned']):,} rows · {len(state['cleaned'].columns)} variables · revision {state['revision']}.")
        st.download_button("Download accepted cleaned dataset (CSV)", csv_bytes(state["cleaned"]),
                           file_name="cleaned_data.csv", mime="text/csv", key="cleaned_download")
    next_stage(1, disabled=state.get("cleaned") is None)


def page_descriptive(state):
    st.header("Descriptive statistics")
    if not need_cleaned(state):
        return
    st.write("Summarize the accepted dataset before fitting models. Missing values are reported explicitly.")
    desc = descriptive_statistics(state["cleaned"], state["types"], category_orders=state["orders"])
    st.caption("Category percentages use nonmissing values for that variable. Missing percentages use all observations. Identifier and date/time fields are excluded.")
    tabs = st.tabs(["Table 1 · Characteristics", "Continuous variables", "Category frequencies"])
    with tabs[0]:
        show_table(desc["table1"], "table1_descriptive")
    with tabs[1]:
        show_table(desc["continuous"], "continuous_statistics")
    with tabs[2]:
        show_table(desc["categorical"], "category_frequencies")
    st.subheader("Explore a distribution")
    cfg = state.setdefault("descriptive_settings", {})
    options = available_variables(state)
    variable = choose("Variable", options, cfg, "variable", key="desc_variable")
    if variable:
        kinds = ["histogram", "boxplot"] if state["types"][variable] == "continuous" else ["bar chart"]
        kind = choose("Figure", kinds, cfg, "kind", key="desc_kind")
        try:
            fig = distribution_plot(state["cleaned"], variable, state["types"], kind=kind, category_orders=state["orders"])
            display_figure(fig, "descriptive_distribution")
        except ValueError as exc:
            st.info(str(exc))
    else:
        st.warning("No analysis variables are included. Check their types in Stage 1.")
    next_stage(2)


def page_comparison(state):
    st.header("Gender / sex comparison")
    if not need_cleaned(state):
        return
    st.write("This optional step compares descriptive characteristics across the groups recorded in your dataset. The same controls can be used for another categorical grouping variable.")
    cfg = state["group_settings"]
    groups = [c for c in available_variables(state) if state["types"][c] in CAT_TYPES]
    if not groups:
        st.info("No categorical grouping variable is available. You can continue to bivariate regression.")
        next_stage(3)
        return
    default = next((c for c in groups if c.lower() in {"sex", "gender"}), groups[0])
    group = choose("Gender, sex or other grouping variable", groups, cfg, "group", default=default, key="comparison_group")
    variables = multi("Variables to compare", [c for c in available_variables(state) if c != group], cfg,
                       "variables", default=[c for c in available_variables(state) if c != group], key="comparison_variables")
    tests = check("Include hypothesis tests", cfg, "include_tests", default=False, key="comparison_tests")
    test_labels = {"welch": "Compare means · Welch’s t-test / Welch ANOVA", "rank": "Compare distributions · Mann–Whitney / Kruskal–Wallis", "pooled": "Compare means · equal-variance t-test / ANOVA"}
    test_kind = choose("Continuous-variable comparison", list(test_labels), cfg, "continuous_test", default="welch", key="comparison_method", format_func=lambda x:test_labels[x])
    if tests:
        st.caption("Choose a method based on your question, design and data. Rank tests generally compare distributions; they are not automatically tests of medians. No normality-test rule selects the method.")
    else:
        st.caption("Descriptive summaries will be shown without p-values.")
    signature = fingerprint({"revision": state["revision"], "settings": {
        k:cfg[k] for k in ("group", "variables", "include_tests", "continuous_test")}})
    if state.get("comparison") is not None and signature != state.get("comparison_signature"):
        invalidate(state, comparison=True, reason="The group comparison settings changed")
    n_groups = int(state["cleaned"][group].nunique(dropna=True))
    if n_groups < 2 or n_groups > 12:
        st.warning("Choose a grouping variable with 2–12 observed groups.")
    if st.button("Run group comparison", type="primary", disabled=not variables or n_groups < 2 or n_groups > 12, key="run_comparison"):
        state["comparison"] = compare_groups(state["cleaned"], state["types"], group, variables,
                                                include_tests=tests, continuous_test=test_kind, category_orders=state["orders"])
        state["comparison_signature"] = signature
        state["exports"] = None
        record(state, "group_comparison", "Generated stratified descriptive table", group=group, variables=variables, include_tests=tests, method=test_kind)
    result = state.get("comparison")
    if result:
        for warning in result.get("warnings", []):
            st.warning(warning)
        info = result.get("info", {})
        st.caption(f"Overall summaries use all {info.get('overall_n', len(state['cleaned'])):,} rows. Group comparisons exclude {info.get('missing_group_n', 0):,} rows with missing group labels; each test also excludes missing values in that variable.")
        show_table(result["table"], "table1_stratified")
        if tests:
            with st.expander("Test methods, sample sizes and limitations"):
                show_table(result["tests"], "group_comparison_tests")
        previous_plot = cfg.get("plot_variable")
        plot_var = choose("Compare distributions visually", variables, cfg, "plot_variable", key="comparison_plot_variable")
        if plot_var != previous_plot:
            state["exports"] = None
        if plot_var:
            try:
                fig = distribution_plot(state["cleaned"], plot_var, state["types"],
                                         kind="boxplot" if state["types"][plot_var] == "continuous" else "bar chart",
                                         group=group, category_orders=state["orders"])
                display_figure(fig, "group_distribution")
            except ValueError as exc:
                st.info(str(exc))
    next_stage(3, "Continue to bivariate analysis · comparison is optional")


def analysis_choices(state):
    """Render and return the authoritative specification for the bivariate stage."""
    cfg = state["analysis"]
    types = state["types"]
    data = state["cleaned"]
    outcomes = [c for c in data if types[c] in {"continuous", "binary"}]
    if not outcomes:
        st.warning("This MVP supports continuous or binary outcomes. Correct the variable type in Stage 1 if needed.")
        return None
    default = "outcome" if "outcome" in outcomes else outcomes[0]
    outcome = choose("Outcome (dependent variable)", outcomes, cfg, "outcome", default=default, key="analysis_outcome")
    family = "linear" if types[outcome] == "continuous" else "logistic"
    cfg["family"] = family
    st.info("Recommended model: " + ("linear regression for a continuous outcome." if family == "linear" else "binary logistic regression."))
    if family == "logistic":
        levels = sorted(data[outcome].dropna().astype(str).unique().tolist())
        event = choose("Event category (modeled as 1)", levels, cfg, "event", default=levels[-1] if levels else None, key="analysis_event")
        if len(levels) != 2:
            st.error("A binary outcome must have exactly two observed categories.")
        st.caption(f"Odds ratios refer to the event '{event}'. The other category is the non-event.")
    else:
        cfg["event"] = None
    choices = [c for c in available_variables(state) if c != outcome]
    default_predictors = [c for c in choices if c != "blood_pressure"] if state["source"].get("synthetic_example") and family == "logistic" else choices
    predictors = multi("Candidate predictors (independent variables)", choices, cfg, "predictors",
                       default=default_predictors, key="analysis_predictors")
    refs = cfg.setdefault("references", {})
    refs = {c:v for c,v in refs.items() if c in predictors and types[c] in CAT_TYPES}
    with st.expander("Reference categories", expanded=bool([c for c in predictors if types[c] in CAT_TYPES])):
        for col in predictors:
            if types[col] in CAT_TYPES:
                levels = sorted(data[col].dropna().astype(str).unique().tolist())
                if col in state["orders"]:
                    levels = [c for c in state["orders"][col] if c in levels]
                choose(f"Reference for {col}", levels, refs, col, key="reference_" + fingerprint(col)[:12])
        st.caption("A reference that disappears from a model’s complete-case sample causes a visible failure; it is never silently replaced.")
    cfg["references"] = refs
    left,right = st.columns(2)
    with left:
        common = check("Use one complete-case sample across all candidate predictors", cfg, "common_sample", default=False, key="bivariate_common_sample")
        st.caption("Default: each unadjusted model uses its own complete cases. Check this to make their samples identical.")
    with right:
        if family == "linear":
            cov_labels = {"nonrobust": "Conventional standard errors", "HC3": "HC3 heteroskedasticity-robust standard errors"}
            choose("Linear-model inference", list(cov_labels), cfg, "cov_type", default="nonrobust", key="analysis_cov", format_func=lambda x:cov_labels[x])
        else:
            cfg["cov_type"] = "nonrobust"
            st.caption("Logistic inference uses maximum likelihood estimates and Wald confidence intervals / joint tests.")
    cfg["missing"] = "complete_case"
    cfg["common_sample"] = common
    return cfg


def page_bivariate(state):
    st.header("Bivariate regression")
    if not need_cleaned(state):
        return
    st.write("Fit one unadjusted model per predictor. This is a regression analysis against the selected outcome; gender comparisons remain a separate descriptive step.")
    cfg = analysis_choices(state)
    if cfg is None:
        return
    signature = fingerprint({"revision":state["revision"], "analysis":cfg})
    if state.get("bivariate") is not None and signature != state.get("bivariate_signature"):
        invalidate(state, bivariate=True, reason="The outcome, predictors, references or analysis settings changed")
    st.caption("Ordinal and nominal predictors use indicator variables. Screening uses an overall joint test of all coefficients for each predictor.")
    if st.button("Run bivariate analysis", type="primary", key="run_bivariate", disabled=not cfg["predictors"]):
        with st.spinner("Fitting unadjusted models…"):
            result = run_bivariate(state["cleaned"], cfg["outcome"], cfg["predictors"], state["types"],
                                    references=cfg["references"], event=cfg["event"], family=cfg["family"],
                                    missing="complete_case", cov_type=cfg["cov_type"], category_orders=state["orders"], common_sample=cfg["common_sample"])
        invalidate(state, model=True, reason="Bivariate models were rerun")
        state["bivariate"] = result
        state["bivariate_signature"] = signature
        state["selected_predictors"] = None
        state["selection"] = {}
        state.pop("selection_origin", None)
        st.session_state.pop("w_final_predictors", None)
        record(state, "bivariate_models", "Fitted one unadjusted model per predictor", **cfg)
    result = state.get("bivariate")
    if result:
        st.subheader("Overall predictor associations")
        st.caption("This table supplies the p-values used in Stage 5. Missing p-values mean an association could not be estimated reliably.")
        show_table(result["variables"], "table2_bivariate_overall")
        st.subheader("Level-specific effect estimates")
        st.caption("Estimate = odds ratio for logistic regression; coefficient in outcome units for linear regression. Continuous estimates are per one-unit increase. All intervals are 95% CIs.")
        show_table(result["terms"], "table2_bivariate_effects")
        with st.expander("Model warnings and failures"):
            notes = []
            for variable, model in result["models"].items():
                if model.get("error"):
                    notes.append({"Variable":variable, "Note":model["error"]})
                notes.extend({"Variable":variable, "Note":w} for w in model.get("warnings", []))
            if notes:
                st.dataframe(pd.DataFrame(notes), hide_index=True, width="stretch")
            else:
                st.write("No estimation warnings were raised. This does not establish that all modeling assumptions hold.")
    next_stage(4, disabled=result is None)


def page_selection(state):
    st.header("Variable selection")
    if not need_cleaned(state):
        return
    result = state.get("bivariate")
    if result is None:
        st.info("Run the bivariate models in Stage 4 first. You can then use a screening list or choose predictors manually.")
        st.button("Go to bivariate analysis", on_click=set_stage, args=(3,))
        return
    st.warning("P-value screening is an optional exploratory strategy. Retain prespecified predictors and scientifically important confounders regardless of their bivariate p-values. The final confidence intervals do not account for data-driven selection.")
    left,right = st.columns(2)
    lists = {}
    for col, threshold, label in [(left,0.05,"List A · p < 0.05"),(right,0.25,"List B · p < 0.25")]:
        names = screen_variables(result, threshold)
        lists[threshold] = names
        with col:
            st.subheader(label)
            table = result["variables"].loc[result["variables"]["variable"].isin(names)]
            columns = [c for c in ["variable","type","p_value","estimate","ci_lower","ci_upper","n"] if c in table]
            st.dataframe(table[columns], hide_index=True, width="stretch")
            if not names:
                st.caption("No variables meet this threshold. Use manual selection below.")
    cfg = state["selection"]
    options = ["Manual / prespecified", "List A · p < 0.05", "List B · p < 0.25"]
    strategy = choose("Starting set", options, cfg, "strategy", default=options[0], key="selection_strategy")
    seed_id = fingerprint({"strategy":strategy, "bivariate":state["bivariate_signature"]})
    if seed_id != state.get("selection_origin"):
        seed = lists[0.05] if strategy == options[1] else lists[0.25] if strategy == options[2] else []
        state["selected_predictors"] = seed
        state["selection_origin"] = seed_id
        cfg["predictors"] = seed
        st.session_state.pop("w_final_predictors", None)
        invalidate(state, model=True, reason="The predictor selection strategy changed")
    candidates = state["analysis"]["predictors"]
    predictors = multi("Final predictors · add or remove whole variables", candidates, cfg, "predictors", default=[], key="final_predictors")
    if predictors != state.get("selected_predictors"):
        invalidate(state, model=True, reason="The selected multivariable predictors changed")
    state["selected_predictors"] = predictors.copy()
    st.caption("All candidate predictors from Stage 4 are available here, including variables whose unadjusted model failed or had a large p-value. Add a new candidate in Stage 4 if needed.")
    if predictors:
        data = state["cleaned"]
        complete = data[[state["analysis"]["outcome"]] + predictors].notna().all(axis=1)
        st.info(f"Current final set: {len(predictors)} predictors. Complete cases for this set: {int(complete.sum()):,} / {len(data):,} observations.")
    else:
        st.info("Select at least one predictor to continue. An intercept-only model is outside this workflow.")
    next_stage(5, disabled=not predictors)


def model_view(model, prefix="model", *, diagnostics=True):
    if not model.get("success"):
        st.error(model.get("error") or "The model could not be estimated.")
        for warning in model.get("warnings", []):
            st.warning(warning)
        st.info("Review category counts, missingness, redundant predictors and outcome coding. Change the model deliberately; predictors are never removed automatically.")
        return
    left,middle,right = st.columns(3)
    left.metric("Observations included", f"{model['n_complete']:,}")
    middle.metric("Rows excluded", f"{model['n_dropped']:,}")
    logistic = model["info"]["family"] == "logistic"
    adjusted = len(model["terms"]["variable"].unique()) > 1
    metrics = model["metrics"]
    right.metric("Events" if logistic else "Adjusted R²", metrics.get("events", "—") if logistic else f"{metrics.get('adjusted_r_squared', float('nan')):.3f}")
    st.caption(("Adjusted " if adjusted else "Unadjusted · one predictor · ") +
               ("odds ratios" if logistic else "coefficients in outcome units"))
    show_table(model["terms"], f"{prefix}_adjusted_effects")
    with st.expander("Overall adjusted predictor tests and fit statistics"):
        show_table(model["variables"], f"{prefix}_overall_tests")
        st.dataframe(pd.DataFrame([{"Statistic":k.replace("_"," "),"Value":v} for k,v in metrics.items()]), hide_index=True, width="stretch")
        if model["info"].get("intercept"):
            st.write("Intercept (on the coefficient / log-odds scale):", model["info"]["intercept"])
        st.caption("Fit statistics describe the fitting sample. AIC/BIC comparisons require the same outcome and observations; pseudo-R² is not the fraction of variance explained.")
    for warning in model.get("warnings", []):
        st.warning(warning)
    try:
        display_figure(forest_plot(model), prefix + "_forest")
    except ValueError as exc:
        st.warning(str(exc))
    with st.expander("Interpret the estimates", expanded=False):
        for line in interpretations(model):
            st.write(line)
        st.caption("Associations are not causal effects. Clinical importance requires the scale, uncertainty, study design and context; a p-value is not a measure of clinical importance.")
    if diagnostics:
        st.subheader("Diagnostics and multicollinearity")
        st.caption("These are prompts for review. They cannot certify independence, a correct functional form or the absence of bias.")
        try:
            display_figure(diagnostic_plot(model), prefix + "_diagnostics")
        except ValueError as exc:
            st.info(str(exc))
        with st.expander("Variance inflation factors (VIF) and observation diagnostics"):
            show_table(model["vif"], prefix + "_vif")
            st.caption("VIF is reported for each encoded design column, not as a generalized VIF for an entire categorical predictor. Review high values; no predictor is deleted automatically.")
            st.download_button("Download observation diagnostics (CSV)", csv_bytes(model["diagnostics"]),
                               file_name="observation_diagnostics.csv", mime="text/csv", key=prefix + "_diagnostics_csv")


def page_multivariable(state):
    st.header("Multivariable regression")
    if not need_cleaned(state):
        return
    predictors = state.get("selected_predictors")
    if not predictors or state.get("bivariate") is None:
        st.info("Choose your final predictor set in Stage 5 before fitting the model.")
        st.button("Go to variable selection", on_click=set_stage, args=(4,))
        return
    analysis = state["analysis"]
    st.write(f"Outcome: **{analysis['outcome']}** · Model: **{'Binary logistic' if analysis['family'] == 'logistic' else 'Linear'} regression**")
    st.write("Predictors: " + ", ".join(predictors))
    st.caption("Reference categories and event coding are inherited from Stage 4. Return there to change them; affected models will need to be rerun.")
    cfg = state["model_settings"]
    missing_labels = {"complete_case": "Complete-case analysis · exclude rows missing any model variable", "raise": "Require complete data · stop if any model value is missing"}
    missing = choose("Missing-data handling", list(missing_labels), cfg, "missing", default="complete_case", key="model_missing", format_func=lambda x:missing_labels[x])
    complete = state["cleaned"][[analysis["outcome"]] + predictors].notna().all(axis=1)
    n = int(complete.sum())
    st.info(f"This model would include {n:,} of {len(complete):,} observations and exclude {len(complete)-n:,} with missing outcome or predictor data. No imputation is performed.")
    st.caption("The final model uses its selected predictors to define complete cases. Its sample can differ from the bivariate samples, including a common candidate-variable sample.")
    cov = analysis["cov_type"]
    if analysis["family"] == "linear":
        labels = {"nonrobust": "Conventional standard errors", "HC3": "HC3 heteroskedasticity-robust standard errors"}
        cov = choose("Linear-model inference", list(labels), cfg, "cov_type", default=cov, key="model_cov", format_func=lambda x:labels[x])
    else:
        cfg["cov_type"] = "nonrobust"
    signature = fingerprint({"revision":state["revision"], "analysis":analysis, "predictors":predictors, "settings":cfg})
    if state.get("model") is not None and signature != state.get("model_signature"):
        invalidate(state, model=True, reason="The final model settings changed")
    if st.button("Fit multivariable model", type="primary", key="fit_model"):
        with st.spinner("Fitting the adjusted model and checking estimability…"):
            model = fit_model(state["cleaned"], analysis["outcome"], predictors, state["types"],
                                references=analysis["references"], event=analysis["event"], family=analysis["family"],
                                missing=missing, cov_type=cov, category_orders=state["orders"])
        state["model"] = model
        state["model_signature"] = signature
        state["exports"] = None
        record(state, "multivariable_fit", "Fitted final model" if model["success"] else "Model estimation failed",
               outcome=analysis["outcome"], predictors=predictors, selection=state["selection"].get("strategy"),
               missing=missing, cov_type=cov, n_complete=model.get("n_complete"), n_dropped=model.get("n_dropped"), error=model.get("error"))
    model = state.get("model")
    if model:
        model_view(model)
    next_stage(6)


def collect_results(state):
    desc = descriptive_statistics(state["cleaned"], state["types"], category_orders=state["orders"])
    tables = {"Table1_Overall":desc["table1"], "Continuous_statistics":desc["continuous"],
              "Category_frequencies":desc["categorical"], "Missingness":profile_data(state["cleaned"], state["types"])}
    if state.get("comparison"):
        tables["Table1_Stratified"] = state["comparison"]["table"]
        tables["Group_tests"] = state["comparison"]["tests"]
    if state.get("bivariate"):
        tables["Table2_Bivariate"] = state["bivariate"]["terms"]
        tables["Bivariate_overall_tests"] = state["bivariate"]["variables"]
    if state.get("model") and state["model"].get("success"):
        model = state["model"]
        tables["Table3_Multivariable"] = model["terms"]
        tables["Adjusted_overall_tests"] = model["variables"]
        tables["Fit_statistics"] = pd.DataFrame([{"statistic":k,"value":v} for k,v in model["metrics"].items()])
        tables["VIF"] = model["vif"]
    log = state.get("log", []) + [{"timestamp_utc":"", **item} for item in state["cleaning"].get("log", [])]
    tables["Analysis_log"] = pd.DataFrame(log)
    settings = {"analysis":state["analysis"], "selection":state["selection"],
                "selected_predictors":state.get("selected_predictors"), "model":state["model_settings"],
                "cleaning":state["cleaning_options"], "cleaning_impact":state["cleaning"].get("impact"),
                "original_variable_types":state.get("cleaning_types", state["types"]),
                "category_orders":state["orders"], "group_comparison":state["group_settings"],
                "dataset_revision":state["revision"]}
    summary = analysis_summary(source=state["source"], variable_types=state["types"],
                                cleaning_log=state["cleaning"].get("log", []), settings=settings,
                                bivariate=state.get("bivariate"), model=state.get("model"), comparison=state.get("comparison"))
    return tables, summary, log


def page_results(state):
    st.header("Results & export")
    if not need_cleaned(state):
        return
    st.write("Review the available tables and save the results with their settings, sample sizes and analysis log.")
    tables, summary, log = collect_results(state)
    model = state.get("model")
    if not model or not model.get("success"):
        st.info("This export is a partial analysis: no successful multivariable model is available. All completed tables can still be downloaded.")
    tabs = st.tabs(["Table 1 · Characteristics", "Table 2 · Bivariate", "Table 3 · Multivariable", "Methods & log"])
    with tabs[0]:
        if state.get("comparison"):
            show_table(tables["Table1_Stratified"], "results_table1_stratified")
        else:
            show_table(tables["Table1_Overall"], "results_table1")
    with tabs[1]:
        if state.get("bivariate"):
            show_table(tables["Table2_Bivariate"], "results_table2")
            show_table(tables["Bivariate_overall_tests"], "results_bivariate_overall")
        else:
            st.info("Run Stage 4 to add bivariate results.")
    with tabs[2]:
        if model:
            model_view(model, prefix="results", diagnostics=False)
        else:
            st.info("Run Stage 6 to add adjusted results.")
    with tabs[3]:
        st.markdown(summary_markdown(summary))
        st.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)
    st.subheader("Download your analysis")
    st.caption("The report ZIP contains tables, a workbook, figures, the analysis summary and the log. Raw and cleaned datasets are excluded; the cleaned dataset can be downloaded separately in Stage 1.")
    left,right = st.columns(2)
    left.download_button("Combined Excel workbook", excel_bytes(tables), file_name="studyflow_results.xlsx",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    right.download_button("Reproducible summary (JSON)", json_bytes(summary), file_name="analysis_summary.json", mime="application/json", width="stretch")
    left.download_button("Readable analysis summary", summary_markdown(summary), file_name="analysis_summary.md", mime="text/markdown", width="stretch")
    right.download_button("Cleaning & analysis log", csv_bytes(pd.DataFrame(log)), file_name="analysis_log.csv", mime="text/csv", width="stretch")
    if st.button("Prepare complete report ZIP", type="primary", key="prepare_report"):
        figures = {}
        with st.spinner("Preparing the result tables and figures…"):
            if model and model.get("success"):
                for name,plotter in [("adjusted_forest",forest_plot),("regression_diagnostics",diagnostic_plot)]:
                    try:
                        fig = plotter(model)
                        for fmt in ("png","svg"):
                            figures[f"{name}.{fmt}"] = figure_bytes(fig, fmt)
                        plt.close(fig)
                    except ValueError as exc:
                        st.warning(f"{name}: {exc}")
            # Include a clear overview figure for every analysis variable (up to 20).
            vars_to_plot = available_variables(state)[:20]
            for i,var in enumerate(vars_to_plot):
                kinds = ["histogram","boxplot"] if state["types"][var] == "continuous" else ["bar chart"]
                for kind in kinds:
                    try:
                        fig = distribution_plot(state["cleaned"], var, state["types"], kind=kind, category_orders=state["orders"])
                        for fmt in ("png","svg"):
                            figures[f"distribution_{i+1}_{kind.replace(' ', '_')}.{fmt}"] = figure_bytes(fig, fmt)
                        plt.close(fig)
                    except ValueError:
                        continue
            if state.get("comparison"):
                cfg = state["group_settings"]
                plot_var = cfg.get("plot_variable") or next(iter(cfg.get("variables", [])), None)
                if plot_var:
                    try:
                        fig = distribution_plot(state["cleaned"], plot_var, state["types"], kind="boxplot", group=cfg["group"], category_orders=state["orders"])
                        for fmt in ("png","svg"):
                            figures[f"group_comparison.{fmt}"] = figure_bytes(fig, fmt)
                        plt.close(fig)
                    except ValueError:
                        pass
            state["exports"] = bundle_bytes(tables, summary, log, figures)
        st.success("Report ready. Save the original source file alongside the report for reproducibility.")
    if state.get("exports") is not None:
        st.download_button("Download complete report ZIP", state["exports"], file_name="studyflow_analysis_report.zip",
                           mime="application/zip", type="primary", width="stretch", key="report_download")
    st.caption("For more than 20 analysis variables, the ZIP contains overview figures for the first 20. Individual plots remain available in Stage 2.")


if "stage" not in st.session_state:
    st.session_state["stage"] = STAGES[0]
state = st.session_state.get("project")
with st.sidebar:
    st.markdown("<div class='small-label'>LOCAL RESEARCH WORKSPACE</div>", unsafe_allow_html=True)
    st.title("StudyFlow")
    st.caption("From dataset to interpretable results")
    stage = st.radio("Analysis workflow", STAGES, key="stage", label_visibility="collapsed")
    stage_index = STAGES.index(stage)
    st.progress((stage_index+1)/7, text=f"Stage {stage_index+1} of 7")
    if state:
        st.divider()
        st.caption("CURRENT DATASET")
        st.write(state["source"]["name"])
        data = state.get("cleaned")
        if data is not None:
            st.caption(f"{len(data):,} rows · {len(data.columns)} variables")
        else:
            st.caption("Cleaning has not been applied")
    st.divider()
    st.caption("Navigation keeps your session. Browser reloads and server restarts clear it; download results before leaving.")
    st.caption("Version 0.1.0 · Linear & binary logistic regression")

if state and state.get("notice"):
    st.info(state.pop("notice"))
pages = [page_upload,page_descriptive,page_comparison,page_bivariate,page_selection,page_multivariable,page_results]
pages[stage_index](state)
