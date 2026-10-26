# StudyFlow — Statistical Analysis App

A Streamlit app for cleaning study data, building descriptive tables, comparing
groups, and fitting linear or binary logistic regression. Its seven-stage
workflow records the cleaning steps, reference categories, model choices and
sample sizes alongside the results.

**Run it with `streamlit run app.py`.** No account, database, API key, or external
analysis service is required. The included data are entirely synthetic.

## Quick start

Use Python **3.10 or later**. The project has been checked with Python 3.12.
Download and extract this repository, then open a terminal inside the
`studyflow-statistical-analysis` directory. Alternatively, clone it:

```bash
git clone https://github.com/Shayan-Rezaeianzadeh/studyflow-statistical-analysis.git
cd studyflow-statistical-analysis
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

These commands do not require activating the environment or changing PowerShell's
execution policy. If `py` is unavailable, use `python` for the first command.

### macOS or Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Streamlit prints the local address, usually `http://localhost:8501`, and opens
the interface in a browser. Keep the terminal open while working. Press
`Ctrl+C` in that terminal when finished.

If you prefer environment activation, use `source .venv/bin/activate` on
macOS/Linux or `.venv\Scripts\Activate.ps1` on Windows, then run:

```bash
streamlit run app.py
```

## Try the example first

Click **Load example dataset**, or upload
`sample_data/example_dataset.csv` in Stage 1. The file contains **504 rows**:
500 simulated people and 4 deliberate duplicate records. It includes missing
values so you can inspect the cleaning and sample-size reporting.

| Column | Suggested type | Meaning in this example |
|---|---|---|
| `participant_id` | Identifier | Synthetic record identifier; exclude from analysis |
| `age` | Continuous | Age in years |
| `bmi` | Continuous | Body mass index |
| `sex` | Nominal | Three simulated groups: Female, Male, Nonbinary |
| `smoking` | Nominal | Never, Former, Current |
| `severity` | Ordinal | Explicit order: Mild, Moderate, Severe |
| `blood_pressure` | Continuous | Simulated continuous outcome |
| `outcome` | Binary | Simulated event, with Yes as the event of interest |
| `visit_date` | Date/time | Synthetic date; exclude from regression |

The variable name `sex` and its categories are an illustrative dataset convention;
the app allows the researcher to select the variable and does not infer whether
a column measures sex or gender.

For a first linear analysis, choose `blood_pressure` as the outcome and `age`,
`bmi`, `sex`, `smoking`, and `severity` as candidate predictors. For a logistic
analysis, select `outcome` and explicitly confirm that `Yes` is the event.
Useful reference categories are Female, Never, and Mild. Review type guesses,
set the severity order, preview cleaning, and explicitly apply it. Removing the
4 duplicates is an optional exercise and is used in the scripted demonstration.

To reproduce the logistic demonstration through the interface:

1. Click **Load example dataset**. Review the types and severity order, check
   **Remove exact duplicate rows**, inspect the impact preview, and click
   **Apply cleaning**.
2. Review Descriptive Statistics. For the optional Gender Comparison, choose
   `sex`, select the desired variables, and click **Run group comparison**.
3. In Bivariate Analysis, set **Outcome (dependent variable)** to `outcome`,
   **Event category (modeled as 1)** to `Yes`, and **Candidate predictors
   (independent variables)** to `age`, `bmi`, `sex`, `smoking`, and `severity`.
   Set references to Female, Never, and Mild. Leave the common-sample option
   unchecked and click **Run bivariate analysis**.
4. In Variable Selection, set **Starting set** to **List B · p < 0.25** and add
   `sex` to **Final predictors · add or remove whole variables**. This is an
   example of deliberately retaining a prespecified variable.
5. Keep complete-case analysis in Multivariable Regression and click
   **Fit multivariable model**. The example includes 429 observations and 125
   events when those settings are used.
6. In Results & Export, inspect the tables and click **Prepare complete report
   ZIP** to produce the download. Save the original dataset with your report.

For the linear example, return to Stage 4, choose `blood_pressure`, keep the same
five candidate predictors (exclude the separate `outcome` variable), and set
**Linear-model inference** to **HC3 heteroskedasticity-robust standard errors**.
Rerun bivariate analysis, start from List B, and fit the adjusted model. Select
HC3 for the adjusted model as well. Those settings reproduce the 434-observation
linear model in the scripted demonstration. The app's initial linear-inference
default is conventional standard errors, so choosing HC3 is deliberate.

## The seven stages

### 1. Upload & Clean Data

Upload a CSV or `.xlsx` workbook and review its preview, size, missingness, unique
values, and potential coding issues. Review the suggested types before applying
them. Supported types are continuous, binary, nominal, ordinal, date/time, and
identifier. Date/time and identifier variables are excluded from regression.

Choose missing-value tokens, optional whitespace trimming, category recodes,
included columns, and optional duplicate removal. Define the complete category
order for each ordinal variable. Numeric or date values that cannot be converted
are reported; converting such values to missing requires explicit opt-in.

Cleaning is previewed before you apply it. The original parsed upload is retained
separately from the cleaned dataset. No outliers are removed and no values are
imputed. Excluding a high-missingness column is a deliberate variable-selection
decision, not an automatic deletion rule.
Duplicate removal is evaluated after cleaning across all retained columns. Keep
identifiers included when they distinguish separate records; marking a column
as Identifier still excludes it from summaries and models without deleting it.

### 2. Descriptive Statistics

Review Table 1, detailed numeric and categorical summaries, and optional plots.
Continuous summaries include observed and missing counts, mean, sample standard
deviation, median, quartiles, and range. Category percentages use the nonmissing
observations for that variable as their denominator; missingness is reported
separately. Ordinal frequencies follow your specified order.

### 3. Gender Comparison (optional)

Select the sex/gender column and variables to compare. The app supports two or
more groups and can display descriptive tables without hypothesis tests.
Choose a comparison of means or a rank-based comparison according to your
question and assumptions. It does not choose tests from a normality-test p-value.

Rows with a missing grouping value remain in the overall column but are omitted
from between-group comparisons. Each test uses records observed for that variable
and the group, and its sample size is displayed. Groups are not recoded, merged,
or excluded automatically.

### 4. Bivariate Analysis

Choose one outcome and the candidate predictors. A continuous outcome uses
linear regression; a binary outcome uses logistic regression. Confirm the
event and reference categories and inference settings.
Every candidate is fitted in its own unadjusted model.

The effect table contains coefficients or odds ratios, 95% confidence intervals,
p-values, and sample sizes. The screening table has one joint p-value per original
variable. For a predictor with more than two categories, its joint p-value tests
all its indicator coefficients together; a single effect estimate is deliberately
left blank because several contrasts are involved. Failed models retain a row
with an explanation.

### 5. Variable Selection

Inspect the two strict screening lists: **p < 0.05** and **p < 0.25**. Select a
starting list, then manually add or remove whole variables. You can retain a
prespecified confounder regardless of its unadjusted p-value, and you can select
predictors manually when no variables meet a threshold.

P-value screening is an optional exploratory operation. An unadjusted association
does not identify a confounder, and screening can omit important variables, select
chance associations, and make subsequent estimates and uncertainty misleading.
Choose the model using the study question and subject knowledge. The reported
intervals and p-values do **not** account for data-driven variable selection or
multiple comparisons.

### 6. Multivariable Regression

Fit the selected predictors together. Review adjusted effects and intervals,
sample size, exclusions, model fit, joint variable tests, VIF, and diagnostic plots.
Changing upstream analysis inputs invalidates dependent results so that old and
new settings are not silently combined.

No predictor is silently dropped to make a model fit. Constant predictors,
rank-deficient design matrices, missing reference categories, separation,
convergence failures, or unusable estimates are reported. If a model fails,
review coding, sparse categories, redundant predictors, sample size, and the study
design. A different model or specialist method may be needed; this app does not
automatically regularize or combine categories.

### 7. Results & Export

Review Table 1, the bivariate table, and the adjusted model together with the
methods and cautious interpretation. Download individual tables, a combined
Excel workbook, PNG/SVG figures, a machine-readable analysis summary, and the
cleaning/analysis log. The summary records settings, variable types, references,
the event definition, sample sizes, exclusions, and software versions.

Retain the original input and the summary together. A summary records the analysis
but is not a substitute for the input data. Downloads are explicit user actions;
the application does not write uploaded research data into this repository.

## Statistical methods and limits

| Component | Implementation |
|---|---|
| Continuous descriptive SD | Sample SD (`ddof=1`); undefined when fewer than 2 observed values |
| Categorical denominators | Nonmissing observations within each variable/group; separate missingness |
| Two-group mean comparison | Welch's independent-samples t-test by default |
| Three or more group means | Welch ANOVA with unequal-variance correction |
| Rank comparisons | Two-sided Mann–Whitney U (2 groups) or Kruskal–Wallis (3+ groups) |
| Categorical comparisons | Pearson chi-square when expected counts are adequate; Fisher exact for sparse 2×2 tables |
| Sparse larger contingency table | No unreliable asymptotic p-value; limitation reported for review |
| Linear regression | Ordinary least squares with an intercept; classical or HC3 covariance; t-based 95% CIs |
| Overall linear predictor p-value | Joint F test of all coefficients belonging to the original predictor |
| Binary logistic regression | Maximum likelihood with an intercept; log-odds Wald inference, exponentiated to ORs and CIs |
| Overall logistic predictor p-value | Joint Wald chi-square test of that predictor's coefficients |
| Categorical predictor encoding | Treatment/indicator coding, with an explicit reference category |
| Ordinal predictors | Indicator coding by default; ordering does not impose equal spacing or a linear trend |
| Missing-data default | Model-specific complete-case analysis; included/excluded counts shown |
| Other missing-data choice | Refuse estimation when required values are missing; optional common complete-case sample for bivariate comparisons |
| Collinearity | Rank check and per-design-column VIF; dummy-column VIF is not a whole-factor GVIF |

Rank tests compare distributions/ranks; they are not generally tests of a median
difference unless additional distributional assumptions hold. Chi-square and
Fisher tests assess association in counts. Neither these tests nor the regression
coefficients demonstrate causality.

Complete-case results can be biased when the mechanism that creates missingness
is incompatible with the analysis. They can also lose precision. This MVP does
not implement multiple imputation. Separate unadjusted models can have different
samples; use the common-sample option if that comparison is important, recognizing
the additional exclusions. The adjusted model uses complete cases for its own
selected predictors.

Interpret a continuous linear coefficient per one unit of the predictor and a
continuous-predictor odds ratio per one unit of the predictor. For a categorical
predictor, the contrast is against the displayed reference. An odds ratio is not
a risk ratio. Confidence intervals that cross the null value (0 for coefficients,
1 for odds ratios) do not prove that an association is absent; statistical
significance alone does not determine clinical importance.

Linear diagnostic plots include residual behavior, normal quantiles, and influence
where available. Logistic diagnostics concern the fitted sample and include
grouped apparent calibration and fitted probabilities by outcome. Observation
diagnostics also provide deviance residuals as a CSV download. Apparent performance is not external
validation. Diagnostics can reveal problems but cannot establish that all
assumptions hold. HC3 can address heteroskedastic covariance under appropriate
conditions; it does not fix confounding, an incorrect functional form, dependence,
or informative missingness.

**Scope:** independent observations; additive main effects; continuous or binary
outcomes. Multinomial/ordinal outcomes, survival/time-to-event models, mixed or
repeated-measures models, survey weights, cluster-robust inference, interactions,
splines, automated causal modeling, multiple imputation, penalized/Firth logistic
regression, and externally validated prediction are outside this MVP. Dates must
be transformed into a scientifically meaningful numeric/categorical variable
before they can enter regression. The app is research software, not a clinical
decision system.

To keep the local MVP responsive, uploads are limited to 50 MB, the group-comparison
interface accepts 2–12 groups, and a regression design is limited to 10,000,000
numeric cells (complete observations × encoded parameters including the intercept).
These are application resource limits, not statistical adequacy criteria. A model
at or below the limit must still pass rank, sample-size, and inference checks.
High-cardinality variables are never silently reduced. Frequency plots display
at most 40 categories; their full frequency tables remain available. The complete
report ZIP includes distribution figures for the first 20 analysis variables;
individual figures for other variables remain available in Stage 2.

## Reproduce the demonstration without the interface

With the environment active:

```bash
python scripts/demo.py --output demo_output
```

Or use `.venv/bin/python` / `.venv\Scripts\python` as above. The script explicitly
removes the four synthetic duplicates, creates descriptive and sex-stratified
tables, fits all five unadjusted predictors, constructs both screening lists, and
fits a linear and a logistic model. It starts each adjusted model from p < 0.25
and manually retains `age` and `sex` to demonstrate prespecified retention.

It writes CSV tables, `example_analysis.xlsx`, both analysis summaries, cleaning
logs, forest/diagnostic/distribution figures in PNG and SVG, ZIP bundles, and
`demo_results.json`. These generated files go into a git-ignored directory.
Regenerate the input file with:

```bash
python sample_data/generate_example.py
```

The generator fixes its NumPy random seed at `20261026`. These data illustrate
software behavior; their associations are not research findings about people.
See [the executed example results](docs/demo_results.md) for sample sizes, fitted
effects, screening choices, and the software versions used in the current
validation run.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests compare an OLS slope/CI with closed-form calculations, HC3 inference with
a directly specified reference model, and a categorical joint test with ANOVA.
A 2×2 example checks the logistic odds ratio and confidence interval against
the corresponding analytic formulas. Additional tests cover event/reference
reversal, complete/quasi separation, rank deficiency, missing samples, invalid
values, full-variable screening, explicit cleaning, category denominators,
ordinal ordering, group tests, exports, and Streamlit navigation/invalidation.
See [the validation record](docs/validation.md) for the actual executed
test count, server startup check, and limits of the verification.

The GitHub Actions workflow runs the tests and scripted demonstration on Python
3.10 and 3.12 when pushed. This workflow is supplied for your repository; a local
test run does not claim that a remote Actions run has already occurred.

## Architecture

`app.py` owns the Streamlit interface and session state. `analysis/` contains
plain Python functions that can be tested or reused without starting Streamlit:

| File | Responsibility |
|---|---|
| `cleaning.py` | Read files, suggest/review types, preview explicit transformations |
| `descriptive.py` | Detailed summaries and overall Table 1 |
| `comparisons.py` | Optional stratified descriptions and between-group tests |
| `regression.py` | Design matrices, model fitting, joint tests, diagnostics, failures |
| `bivariate.py` | One-predictor fits and whole-variable screening lists |
| `visualization.py` | Distribution, forest, and diagnostic figures |
| `export.py` | CSV/Excel exports, summaries, and result bundles |

There is no background job service, server-side database, or account layer.
Direct runtime dependencies are Streamlit, Pandas, NumPy, SciPy, statsmodels,
Matplotlib, and openpyxl; pytest is a development-only dependency. Streamlit and
these libraries install their own necessary transitive dependencies.

## Privacy and session behavior

Data are analyzed in the local Python process when you run this app locally.
Avoid uploading private data to a remotely hosted instance without appropriate
authorization and safeguards. The app does not call an external analysis or AI
service, and Streamlit usage-statistics collection is disabled in the provided
configuration. A local app is not a substitute for your institution's data
governance or device security.

Selections and results persist while you navigate within the active Streamlit
session. Refreshing or closing the browser, restarting the server, or resetting
the session can discard them. Export your work before ending a session. Session
state is not persistent storage. Exported spreadsheets/CSVs protect text cells
that could otherwise be interpreted as formulas; numeric results remain numeric.

## License

The project is distributed under the [MIT license](LICENSE).

## Troubleshooting

- **`streamlit` is not recognized:** use the environment's Python executable with
  `-m streamlit`, as in the quick-start commands.
- **A module cannot be imported:** install requirements with the same Python
  executable used to launch the app. Confirm that the terminal is in this project.
- **A CSV has the wrong shape or encoding:** save it as UTF-8 comma-separated CSV,
  or use `.xlsx`. Use one header row and one observation per row.
- **Numeric data are identified as categories:** inspect unexpected tokens or
  punctuation, explicitly select Continuous, and review the conversion preview.
- **A binary outcome has more than two values:** fix inconsistent labels or choose
  another outcome. Changing only the type does not combine categories.
- **A model is not estimable:** read its error; do not interpret empty estimates
  as a null finding. Revisit coding, model size, missingness, and design.
- **Results disappeared after an edit:** upstream changes intentionally invalidate
  dependent analyses. Rerun the indicated stage with the new settings.
- **Port 8501 is occupied:** run `python -m streamlit run app.py --server.port 8502`.

## Implementation references

- [Streamlit session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)
- [Streamlit application testing](https://docs.streamlit.io/develop/api-reference/app-testing)
- [statsmodels OLS results and inference](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.html)
- [SciPy chi-square contingency tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
- [SciPy Fisher exact test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)
