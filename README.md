# Personalised Air Quality Prediction & Health Advisory

## Project description

**Personalised Air Quality Prediction & Health Advisory** is a
Python-based machine-learning project implemented in a Google Colab
notebook. It analyses an uploaded air-quality dataset, trains and
compares regression models to estimate a selected numerical air-quality
target (such as AQI or PM2.5 concentration), and demonstrates a separate
rule-based advisory layer.

The project combines data inspection, cleaning, exploratory data
analysis, preprocessing, regression modelling, model evaluation,
visualisation, and model export in one reusable workflow. Its advisory
demonstration uses an air-quality category together with an activity
choice and an optional sensitivity preference to generate general
environmental guidance.

> **Important scope note:** The notebook is a configurable starter
> implementation. It does not include or identify a final dataset,
> publisher, geographic coverage, or achieved model scores. These
> details must be supplied after selecting and running a documented CSV
> dataset. With measurements from the same observation time, the model
> performs **same-time estimation**, not future forecasting.

## Main objectives

-   Load and inspect an air-quality CSV dataset.
-   Identify a target variable and suitable predictor features.
-   Handle duplicate records, missing target values, and missing
    predictor values.
-   Explore target distribution, numeric correlations, and time trends
    (when a timestamp is available).
-   Preprocess numerical and categorical features.
-   Train and compare three regression algorithms.
-   Evaluate model performance using MAE, MSE, RMSE, and R².
-   Visualise model errors and compare predictions with actual values.
-   Demonstrate activity-aware, rule-based air-quality guidance.
-   Export the best-performing model and evaluation table.

## Algorithms used

1.  **Linear Regression** --- provides a baseline by modelling a linear
    relationship between input features and the numerical target.
2.  **Decision Tree Regressor** --- models non-linear relationships
    using a sequence of feature-based splits.
3.  **Random Forest Regressor** --- combines multiple decision trees to
    produce an ensemble prediction.

The notebook selects the model with the lowest test-set RMSE as its
initial comparison rule. Review all evaluation metrics before deciding
which model is most appropriate.

## Workflow

1.  Install and import the required Python libraries.
2.  Upload a CSV file in Google Colab.
3.  Configure `TARGET_COL`, `DATETIME_COL`, and `DROP_COLS` to match the
    uploaded dataset.
4.  Inspect data types, missing values, duplicates, and target
    statistics.
5.  Clean exact duplicate rows and remove records with a missing target.
6.  Perform exploratory data analysis.
7.  Create date/time features when a timestamp column is configured.
8.  Split the observations in their existing chronological/row order:
    first 80% for training and final 20% for testing.
9.  Impute missing numerical values using the training-set median;
    impute missing categorical values using the most frequent training
    value; scale numerical features and one-hot encode categorical
    features.
10. Train the three regression models and calculate evaluation metrics.
11. Compare model performance and inspect actual-versus-predicted and
    residual plots.
12. Test the advisory function and export the selected model and
    results.

## Technology stack

-   Python
-   Google Colab
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   Joblib

## How to run

1.  Open the project notebook in Google Colab.
2.  Run the first setup/import cell.
3.  Upload your air-quality dataset in CSV format when prompted.
4.  Check the displayed column names.
5.  Edit the configuration cell:
    -   `TARGET_COL`: the exact name of the numerical target column,
        such as `AQI` (example only) or `PM2.5`.
    -   `DATETIME_COL`: the exact timestamp column name, or `None` if
        the dataset has no timestamp.
    -   `DROP_COLS`: columns that should not be used as predictors, such
        as identifiers, text descriptions, or target-derived labels.
6.  Run the remaining cells in order.
7.  Review the printed metrics and plots. Record the actual dataset
    source, licence, units, coverage, cleaning counts, and measured
    results in your project report.

## Evaluation metrics

-   **MAE (Mean Absolute Error):** average absolute difference between
    actual and predicted values. Lower is better.
-   **MSE (Mean Squared Error):** average squared prediction error;
    larger errors receive greater penalty. Lower is better.
-   **RMSE (Root Mean Squared Error):** square root of MSE, expressed in
    the target's unit. Lower is better.
-   **R² (Coefficient of Determination):** indicates how much of the
    target variation is explained by the model on the test data. Higher
    is generally better; interpret it alongside the error metrics.

No performance values are stated here because they depend on the dataset
uploaded and the actual notebook run.

## Personalised advisory demonstration

The notebook includes
`make_advisory(category, activity, sensitivity="General")`. It produces
general guidance based on: - an air-quality category supplied to the
function, - an activity such as walking, jogging, running, or cycling, -
an optional sensitivity preference.

The category is currently entered manually for demonstration. Before
connecting it to a predicted AQI value, select the applicable official
AQI standard and implement its exact category thresholds. The notebook
intentionally does not invent threshold values. The advisory is general
environmental information, not medical advice.

## Important limitations and responsible use

-   The user must provide a documented CSV dataset; the notebook does
    not download a dataset automatically.
-   Dataset source, publisher, licence, location, date range, sampling
    frequency, and measurement units must be verified and reported.
-   The example column names in the configuration cell may need to be
    changed to match the uploaded CSV exactly.
-   The 80/20 split preserves row order. If a timestamp is available,
    records are sorted by it; otherwise, verify that the CSV is already
    in meaningful chronological order.
-   Same-time pollutant/weather measurements can cause target leakage if
    they are used to estimate an AQI that is calculated from those same
    pollutant measurements. Choose predictors that would genuinely be
    available at prediction time.
-   The current implementation does not create lagged features or define
    a future forecast horizon. Do not claim that it predicts future air
    quality.
-   The advisory category is manually supplied in the demonstration and
    is not automatically derived from the model output.
-   The advisory is not a medical diagnosis or a substitute for official
    local air-quality information.

## Output files

After a successful run, the notebook saves:

-   `best_air_quality_model.joblib` --- the selected trained regression
    pipeline.
-   `model_evaluation_results.csv` --- MAE, MSE, RMSE, and R² results
    for the three models.

The notebook prompts Google Colab to download these files.

## Suggested repository structure

``` text
Personalised-Air-Quality-Prediction/
├── Personalised_Air_Quality_Colab_Starter.ipynb
├── README.md
├── best_air_quality_model.joblib       # generated after running
└── model_evaluation_results.csv        # generated after running
```

## Information to add before final submission

-   Dataset name and publisher
-   Dataset URL, access date, and licence
-   Geographic coverage and observation period
-   Sampling frequency and target measurement units
-   Number of records before and after cleaning
-   Final predictor columns and target column
-   Actual MAE, MSE, RMSE, and R² for each model
-   Selected official AQI category standard and thresholds, if the
    advisory is linked to a predicted AQI
-   A clear statement of whether the task is same-time estimation or
    future forecasting

------------------------------------------------------------------------

**Project title:** Personalised Air Quality Prediction & Health
Advisory\
**Implementation:** Python / Google Colab\
**Project type:** Machine Learning --- Regression and Rule-Based
Advisory Demonstration
