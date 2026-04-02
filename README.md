# Weather Station Risk Predictor

A machine learning system that classifies Tamil Nadu weather stations into operational risk tiers — **Low**, **Moderate**, or **High** — based on four years of hourly rainfall telemetry.

The system is designed to support **parametric insurance for gig workers**, where payouts are triggered automatically by weather thresholds rather than manual claims.

---

# Dataset Description

**Source:** Tamil Nadu State Water Resources Department — Surface Water and Groundwater telemetry network  
**Coverage:** January 2022 to September 2025 (with sparse records from August 2021)  
**Size:** 161,593 rows across **145 unique weather stations** in **36 districts**

## Key Columns

| Column | Description |
|------|------|
| `Station` | Name of the weather station |
| `District` | Administrative district |
| `Latitude` / `Longitude` | Station coordinates |
| `Data Acquisition Time` | Timestamp of the reading (DD-MM-YYYY HH:MM) |
| `Telemetry Hourly Rainfall (mm)` | Rainfall recorded in that hour |

---

# Machine Learning Approach

The prediction task is a **three-class classification problem**.

Given a station's complete rainfall history summarized into **15 engineered features**, the model predicts one of three risk labels:

- Low  
- Moderate  
- High  

---

## Why Station-Level Aggregation

The raw dataset is a **time series of hourly rainfall events**.

Training directly on hourly rows would cause two problems:

1. The model would learn **hour-level noise** rather than station-level characteristics.
2. The same historical time series would be required during inference.

Instead, the hourly rainfall data is **aggregated into a single feature vector per station** that captures rainfall behavior across multiple years.

---

# Feature Groups

| Group | Features | What It Captures |
|------|------|------|
| Magnitude | avg_rain, std_rain, max_rain, median_rain | Typical and extreme rainfall intensity |
| Rolling peaks | max_24h, avg_24h, p95_24h, max_72h | Flood-relevant rainfall accumulation |
| Frequency | rain_freq, heavy_rate, extreme_rate | Frequency of rainfall events |
| Seasonal | nemr_rain_share, sw_rain_share | Exposure to monsoon seasons |
| Spatial | latitude, longitude | Geographic location |

---

# Classification Models

Two ensemble models are trained and evaluated using cross-validation.

## Gradient Boosting Classifier

Sequential ensemble model using decision trees.

Configuration:
max_depth = 3
learning_rate = 0.05
n_estimators = 300
subsample = 0.8

---

## Random Forest Classifier

Parallel ensemble model using bagged decision trees.

Configuration:
max_depth = 6
min_samples_leaf = 3
class_weight = balanced
n_estimators = 500

Both models are wrapped with:
CalibratedClassifierCV(method="isotonic")

This improves **probability calibration**, producing reliable confidence scores.

---

# Risk Score Formula

Each station is assigned a **composite rainfall risk score** derived from weighted percentile ranking across rainfall features.

## Composite Risk Score
R_i = w1 * rank_norm(A_i) + w2 * rank_norm(H_i) + w3 * rank_norm(E_i) + w4 * rank_norm(F_i)

Where:

| Symbol | Variable | Weight | Description |
|------|------|------|------|
| A_i | avg_rain | 0.30 | Mean rainfall intensity |
| H_i | heavy_rate | 0.30 | Fraction of hours rainfall > 10 mm |
| E_i | extreme_rate | 0.25 | Fraction of hours rainfall > 30 mm |
| F_i | rain_freq | 0.15 | Fraction of hours rainfall > 1 mm |

### Rank Normalization
rank_norm(x_i) = rank(x_i) / N

Where:
N = 145 stations

---

# Risk Label Assignment

Stations are divided into three classes based on percentile thresholds.
risk_label_i =
Low if R_i < P33
Moderate if P33 <= R_i < P67
High if R_i >= P67

This produces an approximately balanced dataset:

- 48 Low-risk stations  
- 49 Moderate-risk stations  
- 48 High-risk stations  

---

# Time-Series Feature Engineering

Rolling rainfall statistics are calculated per station in chronological order.

## Rolling Rainfall Accumulation
rain_6h(s,t) = sum( rainfall(s,t-5) ... rainfall(s,t) )
rain_24h(s,t) = sum( rainfall(s,t-23) ... rainfall(s,t) )
rain_72h(s,t) = sum( rainfall(s,t-71) ... rainfall(s,t) )

The **maximum 24-hour rainfall (`max_24h`)** captures extreme rainfall events.

---

## Rainfall Intensity Classification
is_rainy(t) = 1 if rainfall(t) > 1 mm
is_heavy(t) = 1 if rainfall(t) > 10 mm
is_extreme(t) = 1 if rainfall(t) > 30 mm

These indicators are aggregated to compute rainfall frequency statistics.

---

## Seasonal Indicators

Tamil Nadu experiences two major monsoon systems.
is_nemr_monsoon(t) = 1 if month(t) in {10,11,12}
is_sw_monsoon(t) = 1 if month(t) in {6,7,8,9}

---

# Model Training

The dataset is split using stratified sampling.
train_test_split(
X,
y,
test_size = 0.20,
stratify = y,
random_state = 42
)

Probability calibration is applied using isotonic regression.
CalibratedClassifierCV(base_model, cv=5, method="isotonic")

Model selection is based on **5-fold cross-validation accuracy**.

---

# Model Evaluation

Evaluation metrics include:

- Accuracy
- Confusion Matrix
- Classification Report
- Feature Importance

Typical result:
Accuracy : 1.0000

Top contributing features often include:

- avg_rain
- heavy_rate
- avg_24h
- extreme_rate
- p95_24h

---

# Prediction Functions

## predict_risk()

Displays a formatted terminal report for a given station.

Example output:
Weather Station Risk Report
Station : Thiruvadanai_1
District : Ramanathapuram
Predicted Risk : HIGH
Confidence : 91%

---

## predict_station_risk()

Returns API-compatible JSON output.

Example:

```json
{
  "station": "Anaikidangu",
  "district": "Kanyakumari",
  "risk_level": "Low",
  "confidence": 0.76,
  "probabilities": {
    "Low": 0.76,
    "Moderate": 0.18,
    "High": 0.06
  }
}
Output Files
File	Description
weather_risk_model.joblib	Trained machine learning model
station_features.csv	Aggregated station feature dataset
station_risk_summary.csv	Station statistics and predicted risk
risk_probabilities.csv	Probability scores for each station
How to Run
Step 1 — Clone the repository
git clone https://github.com/ss120406/weather-risk-predictor.git
cd weather-risk-predictor
Step 2 — Create a virtual environment
python3 -m venv venv
source venv/bin/activate
Windows:
venv\Scripts\activate
Step 3 — Install dependencies
pip install pandas numpy scipy scikit-learn joblib
Step 4 — Place the dataset
Copy the rainfall dataset into:
data/rainfall_tel_hr_tamil_nadu_sw_gw_tn_2021_2025.csv
Step 5 — Run the pipeline
python weather_risk_predictor.py
Step 6 — Interactive prediction
Example:
Enter station name: Anaikidangu
Output:
Station    : Anaikidangu
District   : Kanyakumari
Risk Level : Low
Confidence : 0.76
Type quit to exit.
