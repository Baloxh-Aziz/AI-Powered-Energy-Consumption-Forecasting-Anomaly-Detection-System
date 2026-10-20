# ⚡ AI-Powered Energy Consumption Forecasting & Anomaly Detection System

Forecasting household electricity usage and detecting abnormal consumption from smart-meter data collected in
**18 houses across 6 Pakistani cities**: Karachi, Lahore, Peshawar, Multan, Skardu and Islamabad.

The project covers the full pipeline: data cleaning, feature engineering, baselines, machine-learning and
deep-learning forecasters, anomaly detection, a multi-city comparison, and an interactive Streamlit dashboard.

---

## Table of contents
1. [Dataset](#dataset)
2. [Project pipeline](#project-pipeline)
3. [Methodology](#methodology)
4. [Key findings](#key-findings)
5. [Dashboard](#dashboard)
6. [Repository structure](#repository-structure)
7. [How to run](#how-to-run)
8. [Limitations](#limitations)
9. [Future work](#future-work)

---

## Dataset

**REWDP: Residential Energy and Weather Data Pakistan.** Minute-level smart-meter readings (total usage plus
appliance-level channels such as AC, fridge, kitchen and washing machine) with matching weather data
(temperature, humidity, wind speed, solar radiation).

- 18 houses, 3 per city, 6 cities
- About one year of data per house (for House 38: Sept 2023 to Aug 2024)
- The raw dataset is **not included** in this repository because of its size. Source: [LUMS Energy Institute, Residential Energy and Weather Data Pakistan](https://lei.lums.edu.pk/datasets/residential-energy-and-weather-data-pakistan.html)

## Project pipeline

| # | Notebook | What it does |
|---|----------|--------------|
| 00 | `00_raw_exploration` | First inspection of the raw files and their problems |
| 01 | `01_data_cleaning_and_eda` | Fixes timestamps and sensor downtime, resamples to hourly, EDA |
| 02 | `02_feature_engineering_and_baseline_model` | Features, chronological split, baselines, Random Forest |
| 03 | `03_lstm_deep_learning_models` | Univariate and multivariate LSTM, compared on identical test hours |
| 04 | `04_anomaly_detection` | Residual-based detection, sensitivity analysis, Isolation Forest cross-check |
| 05 | `05_multi_city_comparative_analysis` | The same pipeline on all 18 houses |
| 06 | `06_model_export_for_deployment` | Saves models and a deployment config; verifies inference from raw history |
| 07 | `07_weather_enhanced_forecasting` | Tests whether weather features improve the forecast |

Notebooks 02, 03, 04 and 06 read files saved by earlier notebooks. Each has `IN_DIR` and `WORK_DIR` variables at
the top (on Kaggle, attach the previous notebook's output as an input and set `IN_DIR` to it).

## Methodology

### 1. Data cleaning
- **Duplicate timestamps:** the meter buffered readings during connection drops and dumped them under one
  timestamp. Each run of identical timestamps is re-stamped one second apart, so nothing is dropped.
- **Fake zero streaks:** runs of 10 or more zero-minutes are sensor downtime, not real zero usage, and are set to
  missing. Keeping them would create false "drop to zero" anomalies.
- **Hourly resampling:** hourly mean; hours with less than 30 valid minutes become missing. Gaps of up to 3 hours
  are interpolated, longer gaps stay missing.
- **`observed` flag:** marks real measured hours. Only observed hours are used as forecasting targets, so models
  are never trained or scored on filled values.

### 2. Feature engineering (no target leakage)
- Calendar: hour, day of week, weekend flag (month is deliberately excluded, since with one year of data the test
  months never appear in training).
- Lags: 1, 2, 24, 48 and 168 hours.
- Rolling mean and standard deviation of the last 24 hours, rolling mean of the last 168 hours, all computed on
  past values only (`shift(1)`).

### 3. Evaluation protocol
- **Chronological split:** train 70%, validation 10%, test 20%. No random splitting.
- **Baselines:** persistence (previous hour), seasonal naive 24 h, seasonal naive 168 h.
- **Metrics:** MAE, RMSE, sMAPE, normalised MAE (MAE divided by mean usage), and a **skill score**
  `1 - MAE_model / MAE_baseline` (0 = no better than the baseline, negative = worse).

### 4. Forecasting models
- **Random Forest (delta target):** predicts the *change* from a base lag (`y - lag_1` for 1 h ahead,
  `y - lag_24` for day-ahead) and adds the lag back. The day-ahead model only uses information known 24 hours
  earlier.
- **LSTM (univariate and multivariate):** 24-hour input windows; scaler fitted on the training period only;
  early stopping on a separate validation period; compared with the Random Forest on the same test hours.

### 5. Anomaly detection
- **Main method:** relative residual `(actual - predicted) / (predicted + eps)` converted to a robust z-score
  (median and MAD). Flags `|z| > K` (default `K = 5`); positive z is a spike, negative z a drop.
- **Threshold calibrated on the validation set**, not the test set, so the number of alerts is a real result and
  not a fixed percentage by construction.
- **Isolation Forest** is used as an unsupervised cross-check.

### 6. Weather experiment
Weather features (temperature, humidity, wind speed, solar radiation) were added and tested in notebook 07. They
did not help reliably, so they are not part of the final model.

## Key findings

- **Persistence is a very strong baseline at 1 hour ahead.** The last hour already explains most of the next one,
  so a model must be judged against it.
- **The Random Forest with the delta target beat the LSTM variants** and was selected as the final forecaster.
  Engineered lag features carry most of the signal on this small tabular dataset.
- **Gains are modest.** Day-ahead improvement over "same hour yesterday" is small on average and negative in some
  houses, so results should not be oversold.
- **Cities differ.** Houses with very spiky load (for example in Skardu) have higher normalised error and a higher
  anomaly rate. The anomaly rate is not an accuracy score.
- **Anomalies often fall on unusual days** (such as public holidays), which suggests a calendar feature could
  reduce false alarms.
- Weather features did not give a reliable improvement.

> Add your own numbers here after running the notebooks, for example a small table of the final MAE, nMAE and skill
> scores for House 38 and the best and worst cities from notebook 05.

## Dashboard

A Streamlit app (`app.py`) to explore results per house:

- Choose city, house, anomaly threshold `k` and date range
- Actual vs day-ahead forecast with anomalies highlighted
- Sortable anomaly list with CSV download
- Comparison table across all houses
- Average usage by hour and by weekday

It reads precomputed files from `artifacts/` (one CSV per house with columns
`datetime, actual, predicted, z, forecast_24h`, plus `meta.json`), so no model loading is needed and it runs on any host.

## Repository structure

```
.
├── app.py                        # Streamlit dashboard
├── artifacts/                    # per-house CSVs + meta.json (used by the dashboard)
├── models/                       # exported .pkl models + deployment_config.json
├── notebooks/                    # 00 to 07, with outputs
├── requirements.txt              # dashboard dependencies
├── requirements-notebooks.txt    # notebook dependencies (incl. TensorFlow)
└── README.md
```

## How to run

**Dashboard**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Notebooks** (developed on Kaggle)
1. `pip install -r requirements-notebooks.txt` for local use, or use Kaggle directly.
2. Set `DATA_ROOT` (and `WEATHER_ROOT` for notebook 07) to your dataset location.
3. Run notebooks in order 00 to 07, setting `IN_DIR` and `WORK_DIR` as described in each notebook's header.

**Using a saved model**
```python
import joblib
bundle = joblib.load("models/energy_forecast_24h.pkl")
model, features, base = bundle["model"], bundle["features"], bundle["base"]
# prediction = X[base] + model.predict(X[features])
```
Each saved bundle contains the model together with its exact feature list, so the wrong columns cannot be fed in.

## Limitations

- About one year of data per house, so the model never sees the same season twice.
- Test windows are short and mostly cover summer; results may not transfer to other seasons.
- Anomalies are statistical outliers, not confirmed faults.
- The detector targets point anomalies; sustained level shifts are not flagged.
- Some houses have low data coverage (`pct_observed`) and their numbers are less reliable.

## Future work

- Add a public-holiday feature to reduce false alarms
- Change-point detection (for example CUSUM) for sustained level shifts
- Multi-year data to capture full seasonality
- A single model trained across houses instead of one model per house

## Group Member

`Azizullah Asad & Muhammad Yahya`
