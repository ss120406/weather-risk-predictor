# Weather Station Risk Predictor

A machine learning system that classifies Tamil Nadu weather stations into operational risk tiers — **Low**, **Moderate**, or **High** — based on historical rainfall telemetry.

The system is designed to support **parametric insurance for gig workers**, where payouts can be triggered automatically using weather risk signals derived from rainfall data.

The focus of this project is **interpretability and simplicity**. Instead of relying on complex deep learning models, the system combines clear rainfall statistics, interpretable scoring logic, and simple ensemble classifiers to estimate weather risk.

---

# Dataset Description

**Source:** Tamil Nadu State Water Resources Department — Surface Water and Groundwater telemetry network  

**Coverage:** January 2022 – September 2025  

**Size:**  
- 161,593 rainfall observations  
- 145 weather stations  
- 36 districts across Tamil Nadu  

## Key Columns

| Column | Description |
|------|------|
| Station | Weather station name |
| District | Administrative district |
| Latitude / Longitude | Station coordinates |
| Data Acquisition Time | Timestamp of rainfall measurement |
| Telemetry Hourly Rainfall (mm) | Hourly rainfall recorded |

---

# Machine Learning Approach

The problem is formulated as a **three-class classification task**.

Each weather station is assigned one of three risk labels:

- Low Risk  
- Moderate Risk  
- High Risk  

Rather than training directly on hourly data, rainfall time-series observations are **aggregated into station-level features** capturing rainfall behavior over multiple years.

This approach provides:

- better interpretability  
- simpler model inputs  
- stable predictions  

---

# Feature Groups

| Group | Features | Description |
|------|------|------|
| Magnitude | avg_rain, std_rain, max_rain, median_rain | Rainfall intensity statistics |
| Rolling peaks | max_24h, avg_24h, p95_24h, max_72h | Short-term rainfall accumulation |
| Frequency | rain_freq, heavy_rate, extreme_rate | Frequency of rainfall events |
| Seasonal | nemr_rain_share, sw_rain_share | Monsoon rainfall exposure |
| Spatial | latitude, longitude | Geographic location |

---

# Classification Models

Two ensemble models are trained and compared using cross-validation.

## Gradient Boosting

Configuration
max_depth = 3
learning_rate = 0.05
n_estimators = 300
subsample = 0.8

## Random Forest

Configuration
max_depth = 6
min_samples_leaf = 3
class_weight = balanced
n_estimators = 500

Both models use probability calibration:
CalibratedClassifierCV(method="isotonic")

Calibration improves **probability reliability** instead of producing overconfident predictions.

---

# Risk Score Formula

Before training, each station is assigned a **composite rainfall risk score** derived from weighted rainfall indicators.
R_i = w1 * rank_norm(A_i) + w2 * rank_norm(H_i) + w3 * rank_norm(E_i) + w4 * rank_norm(F_i)

Where

| Variable | Meaning | Weight |
|------|------|------|
| A_i | Average rainfall intensity | 0.30 |
| H_i | Heavy rainfall frequency (>10 mm) | 0.30 |
| E_i | Extreme rainfall frequency (>30 mm) | 0.25 |
| F_i | Rainfall occurrence frequency (>1 mm) | 0.15 |

Rank normalization:
rank_norm(x_i) = rank(x_i) / N

Where **N = 145 stations**.

---

# Risk Label Assignment

Stations are divided into three classes using percentile thresholds.
Low if R_i < P33
Moderate if P33 ≤ R_i < P67
High if R_i ≥ P67

This produces a balanced dataset of approximately:

- 48 Low-risk stations  
- 49 Moderate-risk stations  
- 48 High-risk stations  

---

# Time-Series Feature Engineering

Rolling rainfall features capture short-term rainfall bursts.
rain_6h(s,t) = sum( rainfall(s,t-5) ... rainfall(s,t) )
rain_24h(s,t) = sum( rainfall(s,t-23) ... rainfall(s,t) )
rain_72h(s,t) = sum( rainfall(s,t-71) ... rainfall(s,t) )

These rolling windows help detect **intense rainfall accumulation events**.

---

# Rainfall Event Definitions

Rainfall intensity indicators:
is_rainy(t) = 1 if rainfall(t) > 1 mm
is_heavy(t) = 1 if rainfall(t) > 10 mm
is_extreme(t) = 1 if rainfall(t) > 30 mm

Seasonal indicators:
is_nemr_monsoon(t) = 1 if month(t) in {10,11,12}
is_sw_monsoon(t) = 1 if month(t) in {6,7,8,9}

---

# Model Training

Training uses a **stratified train-test split** to preserve class balance.
train_test_split(
X,
y,
test_size = 0.20,
stratify = y,
random_state = 42
)

Probability calibration is applied using:
CalibratedClassifierCV(base_model, cv=5, method="isotonic")

Model selection is based on **5-fold cross-validation accuracy**.

---

# Model Evaluation

Evaluation metrics include:

- Accuracy  
- Confusion Matrix  
- Precision / Recall / F1 score  
- Feature Importance  

Typical results from the trained model are in the range:
Accuracy : 0.85 – 0.95

Accuracy significantly higher than this could indicate overfitting due to the small dataset size (145 stations). Cross-validation is therefore used to ensure robust performance.

Important features typically include:

- avg_rain  
- heavy_rate  
- max_24h  
- extreme_rate  
- p95_24h  

---

# Prediction Functions

## predict_risk()

Returns a formatted station report.

Example output:
Weather Station Risk Report
Station : Thiruvadanai_1
District : Ramanathapuram
Predicted Risk : HIGH
Confidence : 0.91

---

## predict_station_risk()

Returns API-ready JSON output.

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

## Output Files

| File | Description |
|------|-------------|
| `weather_risk_model.joblib` | Trained machine learning model |
| `station_features.csv` | Aggregated station feature dataset |
| `station_risk_summary.csv` | Station statistics and predicted risk |
| `risk_probabilities.csv` | Probability predictions for each station |

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ss120406/weather-risk-predictor.git
cd weather-risk-predictor

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
Windows
venv\Scripts\activate

### 3. Install dependencies

```bash
pip install pandas numpy scipy scikit-learn joblib

### 4. Place the dataset

```bash
Copy the rainfall dataset into:
data/rainfall_tel_hr_tamil_nadu_sw_gw_tn_2021_2025.csv

### 5. Run the pipeline

```bash
python weather_risk_predictor.py