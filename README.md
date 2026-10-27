# COVID-19 Vaccination Data Analysis and Forecasting

## Overview

This project focuses on analyzing global COVID-19 vaccination data and developing a machine learning model to predict next-day vaccination activity in the United States.

Using Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn, the project combines exploratory data analysis (EDA), data preprocessing, statistical visualization, time-series feature engineering, and supervised machine learning.

A Random Forest Regressor is trained to forecast daily vaccination counts using historical vaccination records and temporal features. Its performance is evaluated against a persistence baseline to assess whether the model provides an improvement over a simple forecasting approach.

## Objectives

- Analyze global COVID-19 vaccination trends across countries.
- Explore cumulative and daily vaccination statistics over time.
- Investigate missing values, duplicate records, and potential reporting anomalies.
- Compare vaccination progress across countries and vaccine manufacturers.
- Engineer time-series features for next-day forecasting.
- Train and evaluate a Random Forest regression model.
- Compare model performance against a persistence baseline.
- Save and reload the trained model for future use.

## Dataset

The project uses two datasets:

### 1. Country Vaccination Data
`country_vaccinations.csv`

Contains country-level vaccination records, including:

- Country name and ISO code
- Date of observation
- Total vaccinations
- People vaccinated and fully vaccinated
- Daily vaccination counts
- Vaccination rates per hundred people
- Daily vaccinations per million people
- Vaccine types and data sources

### 2. Vaccination Data by Manufacturer
`country_vaccinations_by_manufacturer.csv`

Contains vaccination records categorized by vaccine manufacturer, including:

- Country or location
- Date
- Vaccine manufacturer
- Total vaccinations attributed to each manufacturer

The datasets cover the period from December 2020 to July 2021 in the version used by this project.

**Note:** The datasets must be available in the working directory before running the notebook.

## Technologies Used

- **Python** — Core programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **Scikit-learn** — Machine learning, preprocessing, and evaluation
- **Joblib** — Model serialization and persistence
- **Jupyter Notebook** — Interactive development and experimentation

## Project Workflow

### 1. Data Loading and Preparation

Both datasets are loaded into Pandas DataFrames. The preparation process includes:

- Converting date columns to datetime objects
- Converting numerical fields to appropriate numeric types
- Checking missing values and data dimensions
- Sorting records chronologically
- Checking duplicate country-date and location-date-manufacturer combinations

### 2. Exploratory Data Analysis (EDA)

The exploratory analysis examines the structure, quality, and distribution of vaccination data.

Key analyses include:

- Dataset dimensions and descriptive statistics
- Missing-value counts and percentages
- Country-level data coverage
- Cumulative vaccination trends
- Global daily vaccination activity
- Correlations between numerical variables
- Identification of changes in cumulative vaccination counts

### 3. Data Preprocessing

Several preprocessing techniques are applied to prepare the data for analysis and modeling:

- Linear interpolation of missing daily vaccination values within each country
- Percentile-based winsorization using the 1st and 99th percentiles to reduce the influence of extreme values
- Weekly aggregation of vaccination statistics
- Extraction of calendar-based features, including year, month, and day of the week

The United States is selected as the primary country for the forecasting task.

### 4. Data Visualization and Storytelling

The project generates visualizations to communicate important patterns in vaccination activity, including:

- Global daily vaccination trends
- Daily and cumulative vaccination progress in the United States
- Monthly average daily vaccination counts
- Countries with the highest reported cumulative vaccination totals
- Vaccination shares by manufacturer
- Manufacturer-specific vaccination trajectories
- Correlation heatmaps
- Distributions before and after winsorization

These visualizations help explore historical trends and compare vaccination activity across countries and manufacturers.

### 5. Feature Engineering

The forecasting dataset is constructed using historical vaccination counts and temporal information.

The model uses the following eight features:

| Feature | Description |
|---|---|
| `daily_vaccinations` | Vaccination count recorded on the current date |
| `daily_lag_1` | Vaccination count from the previous observation |
| `daily_lag_7` | Vaccination count from seven observations earlier |
| `rolling_mean_7` | Seven-observation rolling mean |
| `rolling_std_7` | Seven-observation rolling standard deviation |
| `day_of_week` | Day of the week |
| `day_of_year` | Day of the year |
| `month` | Calendar month |

**Target variable:** `target_next_day`, representing the next day's vaccination count.

Only records with consecutive calendar dates and a valid next-day target are retained.

### 6. Machine Learning Model

A **Random Forest Regressor** is used to predict next-day vaccination counts.

The model is implemented through a Scikit-learn Pipeline containing:

- `SimpleImputer(strategy="median")` for handling missing feature values
- `RandomForestRegressor` with 300 trees, a minimum leaf size of 2, and a fixed random seed of 42

The available observations are split chronologically:

- **Training set:** 80%
- **Test set:** 20%

The data is not shuffled, preserving the chronological order required for time-series evaluation.

### 7. Baseline Comparison

The model is compared against a persistence baseline, which assumes that the next day's vaccination count will equal the current day's count.

This comparison provides a simple reference point for determining whether the machine learning model improves forecasting accuracy.

### 8. Model Evaluation

The following regression metrics are calculated:

- **Mean Absolute Error (MAE):** Average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Square root of the average squared prediction error, giving greater weight to larger errors.
- **R² Score:** Measures predictive performance relative to a mean-based reference.

Additional visualizations compare actual and predicted values, examine residual distributions, and assess prediction accuracy over time.

## Results

The model was evaluated on 41 test observations covering June 4 to July 14, 2021.

The following results were obtained from the executed notebook:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest Regressor | 116,469.62 | 146,819.59 | 0.6351 |
| Persistence Baseline | 59,230.68 | 73,374.60 | 0.9089 |

The Random Forest model achieved an R² score of approximately 0.635, but the persistence baseline performed better on all three evaluation metrics.

This indicates that, on this test period, the Random Forest model did not outperform the simple assumption that tomorrow's vaccination count would equal today's count.

The feature importance analysis also showed that the current day's vaccination count was the dominant predictor, accounting for approximately 85.8% of the model's reported feature importance.

These results highlight the importance of evaluating machine learning models against meaningful baselines rather than relying on model complexity alone.

## Model Persistence

The trained pipeline and its associated metadata are saved using Joblib.

**Output file:** `covid_vaccination_model.joblib`

The saved bundle contains:

- The trained preprocessing and prediction pipeline
- The selected country
- The feature column names
- The target variable name
- A description of the forecasting task

The notebook also reloads the saved bundle and verifies that it can generate predictions.

## How to Run

### Prerequisites

Install Python and the required libraries.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

### Setup

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Place the two CSV datasets in the working directory:

```text
country_vaccinations.csv
country_vaccinations_by_manufacturer.csv
```

### Run the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and execute the cells sequentially.

The notebook will perform the analysis, train and evaluate the model, generate visualizations, and save the model bundle as `covid_vaccination_model.joblib`.

## Limitations

- The datasets cover a historical period ending in July 2021 and do not represent current vaccination activity.
- The forecasting model focuses exclusively on the United States.
- The available historical data is relatively limited for machine learning.
- Missing-value interpolation and winsorization may affect the underlying data distribution.
- The persistence baseline outperformed the Random Forest model on the selected test period.
- The model's performance on this test set does not guarantee similar performance on other periods or countries.
- Reported vaccination counts and manufacturer totals depend on the completeness and consistency of the original data sources.

## Future Improvements

Potential extensions include:

- Testing additional forecasting models, such as linear regression, gradient boosting, and time-series-specific approaches.
- Performing time-series cross-validation and hyperparameter tuning.
- Comparing different feature engineering strategies and forecast horizons.
- Evaluating model performance across multiple countries.
- Investigating alternative approaches to missing-value handling and outlier treatment.
- Incorporating additional explanatory variables where reliable historical data is available.
- Improving the forecasting pipeline and benchmarking models against stronger baselines.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for historical COVID-19 vaccination analysis, from data preparation and exploratory visualization to feature engineering, regression modeling, evaluation, and model persistence.

Beyond building a predictive model, the project emphasizes the importance of chronological validation, transparent performance metrics, and baseline comparisons when working with time-series data.
