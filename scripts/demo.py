"""Exercise all analytical stages with synthetic data and write reviewable results.

Run from the repository root: python scripts/demo.py --output demo_output
"""

import argparse
import hashlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.bivariate import run_bivariate, screen_variables  # noqa: E402
from analysis.cleaning import clean_data, read_dataset  # noqa: E402
from analysis.comparisons import compare_groups  # noqa: E402
from analysis.descriptive import descriptive_statistics  # noqa: E402
from analysis.export import (  # noqa: E402
    analysis_summary, bundle_bytes, csv_bytes, excel_bytes, json_bytes, summary_markdown,
)
from analysis.regression import fit_model  # noqa: E402
from analysis.visualization import (  # noqa: E402
    diagnostic_plot, distribution_plot, figure_bytes, forest_plot,
)

TYPES = {
    "participant_id": "identifier", "age": "continuous", "bmi": "continuous",
    "sex": "nominal", "smoking": "nominal", "severity": "ordinal",
    "blood_pressure": "continuous", "outcome": "binary", "visit_date": "datetime",
}
PREDICTORS = ["age", "bmi", "sex", "smoking", "severity"]
REFERENCES = {"sex": "Female", "smoking": "Never", "severity": "Mild"}
ORDERS = {"severity": ["Mild", "Moderate", "Severe"]}


def save_figure(fig, stem, output, figures):
    for extension in ["png", "svg"]:
        name = f"{stem}.{extension}"
        content = figure_bytes(fig, format=extension)
        (output / name).write_bytes(content)
        figures[name] = content
    plt.close(fig)


def run_demo(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    path = PROJECT_ROOT / "sample_data" / "example_dataset.csv"
    original = read_dataset(path, filename=path.name)
    cleaning_options = {"drop_duplicates": True}
    cleaned = clean_data(original, TYPES, options=cleaning_options, ordinal_orders=ORDERS)
    if cleaned["errors"]:
        raise RuntimeError("Example cleaning failed: " + "; ".join(cleaned["errors"]))
    data = cleaned["data"]
    descriptives = descriptive_statistics(data, TYPES, category_orders=ORDERS)
    comparison = compare_groups(data, TYPES, "sex", PREDICTORS[:2] + ["smoking", "severity"],
                                continuous_test="welch", category_orders=ORDERS)
    tables = {
        "Table1_Overall": descriptives["table1"],
        "Table1_BySex": comparison["table"],
        "Continuous_Summary": descriptives["continuous"],
        "Categorical_Summary": descriptives["categorical"],
        "Group_Tests": comparison["tests"],
    }
    figures = {}
    save_figure(distribution_plot(data, "age", TYPES), "age_histogram", output, figures)
    save_figure(distribution_plot(data, "blood_pressure", TYPES, kind="boxplot", group="sex"),
                "blood_pressure_by_sex", output, figures)
    save_figure(distribution_plot(data, "smoking", TYPES, kind="bar", category_orders=ORDERS),
                "smoking_frequencies", output, figures)
    shared_tables = dict(tables)
    shared_figures = dict(figures)
    demo_results = {
        "source_rows": len(original), "cleaned_rows": len(data),
        "duplicates_removed": cleaned["impact"]["duplicate_rows_removed"],
        "models": {},
    }
    for family, outcome in [("linear", "blood_pressure"), ("logistic", "outcome")]:
        cov_type = "HC3" if family == "linear" else "nonrobust"
        unadjusted = run_bivariate(data, outcome, PREDICTORS, TYPES, references=REFERENCES,
                                   event="Yes" if family == "logistic" else None,
                                   family=family, cov_type=cov_type, category_orders=ORDERS)
        list_a = screen_variables(unadjusted, 0.05)
        list_b = screen_variables(unadjusted, 0.25)
        # Illustrate manual retention of prespecified variables after screening.
        selected = list(dict.fromkeys(list_b + ["age", "sex"]))
        model = fit_model(data, outcome, selected, TYPES, references=REFERENCES,
                          event="Yes" if family == "logistic" else None,
                          family=family, cov_type=cov_type, category_orders=ORDERS)
        if not model["success"]:
            raise RuntimeError(f"Example {family} model failed: {model['error']}")
        suffix = "Linear" if family == "linear" else "Logistic"
        tables[f"Table2_{suffix}"] = unadjusted["terms"]
        tables[f"Screening_{suffix}"] = unadjusted["variables"]
        tables[f"Table3_{suffix}"] = model["terms"]
        tables[f"JointTests_{suffix}"] = model["variables"]
        tables[f"VIF_{suffix}"] = model["vif"]
        settings = {
            "outcome": outcome, "family": family, "candidate_predictors": PREDICTORS,
            "selected_predictors": selected, "references": REFERENCES,
            "event": "Yes" if family == "logistic" else None,
            "missing": "complete_case", "common_bivariate_sample": False,
            "cov_type": cov_type, "category_orders": ORDERS,
            "screening_threshold": 0.25, "p_lt_0_05": list_a, "p_lt_0_25": list_b,
            "manual_additions": [v for v in ["age", "sex"] if v not in list_b],
            "cleaning_options": cleaning_options,
        }
        analysis_log = cleaned["log"] + [
            {"action": "Bivariate regression", "detail": f"{family}; candidates: {', '.join(PREDICTORS)}"},
            {"action": "Predictor selection", "detail": f"p < 0.25, then retain age and sex; selected: {', '.join(selected)}"},
            {"action": "Multivariable regression", "detail": f"{family}; n={model['n_complete']}; excluded={model['n_dropped']}"},
        ]
        summary = analysis_summary(
            source={"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": len(original), "columns": len(original.columns)},
            variable_types=TYPES, cleaning_log=cleaned["log"], settings=settings,
            bivariate=unadjusted, model=model, comparison=comparison,
        )
        (output / f"{family}_analysis_summary.json").write_bytes(json_bytes(summary))
        (output / f"{family}_analysis_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
        save_figure(forest_plot(model), f"{family}_forest", output, figures)
        save_figure(diagnostic_plot(model), f"{family}_diagnostics", output, figures)
        model_tables = {**shared_tables, **{name: table for name, table in tables.items() if name.endswith(suffix)}}
        model_figures = {**shared_figures, **{name: content for name, content in figures.items() if name.startswith(family)}}
        (output / f"{family}_results_bundle.zip").write_bytes(
            bundle_bytes(model_tables, summary, analysis_log, figures=model_figures)
        )
        demo_results["models"][family] = {
            "outcome": outcome, "list_a": list_a, "list_b": list_b,
            "selected": selected, "n": model["n_complete"], "excluded": model["n_dropped"],
            "metrics": model["metrics"], "warnings": model["warnings"],
            "estimates": model["terms"].to_dict(orient="records"),
        }
    for name, table in tables.items():
        (output / f"{name}.csv").write_bytes(csv_bytes(table))
    (output / "example_analysis.xlsx").write_bytes(excel_bytes(tables))
    (output / "cleaning_log.json").write_bytes(json_bytes(cleaned["log"]))
    (output / "demo_results.json").write_bytes(json_bytes(demo_results))
    print(f"Read {len(original)} synthetic records; removed {len(original) - len(data)} approved duplicates.")
    for family, result in demo_results["models"].items():
        print(f"{family.title()}: n={result['n']}; excluded={result['excluded']}; predictors={', '.join(result['selected'])}")
    print(f"Wrote tables, workbook, summaries, logs, figures, and ZIP bundles to {output.resolve()}")
    return demo_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo_output"), help="Directory for generated example outputs")
    args = parser.parse_args()
    run_demo(args.output)
