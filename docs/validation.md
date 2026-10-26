# Validation record

Validated on 25 October 2026 (America/Vancouver) on macOS using Python 3.12.14.
The direct package versions are listed in `demo_results.md` and recorded in the
generated analysis summaries. These are executed checks, not proposed tests.

## Automated suite

Command, from the repository root with the development dependencies installed:

```bash
python -m pytest -q
```

Final result:

```text
53 passed in 39.74s
```

| Test file | Tests | Main checks |
|---|---:|---|
| `tests/test_analysis.py` | 34 | Analytic OLS and odds-ratio benchmarks, HC3, joint categorical tests, event/reference reversal, separation, rank failures, transparent missing samples, cleaning, descriptives, group tests |
| `tests/test_exports.py` | 6 | CSV/XLSX formula protection, numeric preservation, valid worksheet names, standard JSON, provenance and ZIP contents |
| `tests/test_numerical_scale.py` | 7 | Numerical rank loss, outcome unit changes, high-cardinality and design-size allocation guards |
| `tests/test_app.py` | 4 | Both regression workflows through Streamlit widgets, group-plot navigation, result invalidation, missing-data refusal, accepted settings versus draft settings, report ZIP generation |
| `tests/test_visualization.py` | 2 | Correct one-predictor interpretation and finite-measurement plot counts |

Streamlit tests use its own `AppTest` framework and exercise the real application
script. They are headless widget tests, not a claim of manual testing in every
browser or operating system.

## Server startup

Executed the documented launch command with a temporary test port:

```bash
python -m streamlit run app.py --server.headless=true --server.port=8518
```

The local `/_stcore/health` endpoint returned HTTP **200** with body **ok**.
The test server was stopped afterwards.

## Full synthetic demonstration

Executed:

```bash
python scripts/demo.py --output demo_output
```

The script read 504 synthetic rows, explicitly removed four duplicate rows,
generated the descriptive and group-comparison tables, fitted both model
families, screened whole variables, manually retained the specified variable,
and wrote the tables, workbook, summaries, logs, figures and report bundles.

The linear model used 434 complete observations. The logistic model used 429
observations, including 125 events. Detailed effects and diagnostics are
documented in `demo_results.md`.

The generated report ZIPs passed CRC integrity checks. Their JSON summaries
were parsed, and the workbooks were opened with openpyxl: each per-model
workbook contained 10 sheets, while the combined example workbook contained 15.
The generated logistic forest plot and linear diagnostic plot were visually
reviewed for readable labels, correct axes, and clipping. The included sample CSV
also matched the deterministic synthetic-data generator exactly.

## Verification limits

- No real patient data were used.
- The local test environment was Python 3.12 on macOS. Windows and Linux
  launch instructions were not executed in this review.
- Dependency ranges permit versions beyond the one tested combination.
- The supplied GitHub Actions workflow has not been run in a remote repository;
  it is configured for Python 3.10 and 3.12 when published.
- Passing tests checks software behavior and selected numerical references.
  It does not establish that a particular study meets model assumptions or that
  exploratory selection yields valid confirmatory inference.
