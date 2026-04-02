"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        Weather Station Risk Predictor — Tamil Nadu Gig Worker Insurance     ║
║        ─────────────────────────────────────────────────────────────────    ║
║  Input  : Station name                                                       ║
║  Output : Risk level → Low | Moderate | High  (+confidence scores)          ║
║  Data   : Hourly rainfall telemetry  2022–2025  (161 593 rows, 145 stations)║
╚══════════════════════════════════════════════════════════════════════════════╝

Design notes
────────────
  • This dataset is a SENSOR-TRIGGER log: only non-zero rainfall hours are
    recorded (minimum value = 0.5 mm). All 145 stations appear "rainy"
    because they only appear when it rains. Thresholds must therefore be
    RELATIVE (percentile-based) rather than absolute.

  • Risk labels are derived from a WEIGHTED COMPOSITE SCORE that ranks
    each station across four rainfall dimensions:
        avg_rain (30%), heavy_rate (30%), extreme_rate (25%), rain_freq (15%)
    Tertile split => Low (bottom 33%) | Moderate (mid 33%) | High (top 33%)
    This yields a balanced 48/49/48 class distribution for robust training.

  • Rolling features use per-station chronological order to prevent
    data leakage across stations.

Pipeline modules
────────────────
  1. load_data()
  2. preprocess_data()
  3. feature_engineering()
  4. aggregate_station_features()
  5. label_stations()          <- composite percentile scoring
  6. train_model()             <- GradientBoosting + RandomForest ensemble
  7. evaluate_model()
  8. predict_risk()            <- human-readable
  9. predict_station_risk()    <- JSON / API output
 10. save_model() / load_model()

Bonus: supabase_integration_demo()
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import joblib
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH     = "data/rainfall_tel_hr_tamil_nadu_sw_gw_tn_2021_2025.csv"
MODEL_PATH    = "models/weather_risk_model.joblib"
FEATURES_PATH = "models/station_features.csv"
SUMMARY_PATH  = "outputs/station_risk_summary.csv"

SENSOR_CAP_MM = 200.0

# Weighted composite score dimensions
RISK_WEIGHTS = {
    "avg_rain":     0.30,
    "heavy_rate":   0.30,
    "extreme_rate": 0.25,
    "rain_freq":    0.15,
}

LABEL_MAP   = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_NAMES = ["Low", "Moderate", "High"]
RISK_ICONS  = {"Low": "GREEN", "Moderate": "YELLOW", "High": "RED"}

FEATURE_COLS = [
    "avg_rain", "std_rain", "max_rain", "median_rain",
    "max_24h", "avg_24h", "p95_24h", "max_72h",
    "rain_freq", "heavy_rate", "extreme_rate",
    "nemr_rain_share", "sw_rain_share",
    "latitude", "longitude",
]


# ==============================================================================
# MODULE 1 — load_data
# ==============================================================================

def load_data(path="data/rainfall_tel_hr_tamil_nadu_sw_gw_tn_2021_2025.csv"):
    """
    Read the raw CSV. Only the six modelling columns are kept.

    Returns
    -------
    pd.DataFrame  raw dataframe
    """
    keep = [
        "Station", "District", "Latitude", "Longitude",
        "Data Acquisition Time", "Telemetry Hourly Rainfall (mm)",
    ]
    df = pd.read_csv(path, usecols=keep, low_memory=False)

    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    null_rain = df["Telemetry Hourly Rainfall (mm)"].isnull().sum()
    if null_rain:
        print(f"[load_data]  WARNING: {null_rain:,} null rainfall rows dropped.")
        df = df.dropna(subset=["Telemetry Hourly Rainfall (mm)"])

    print(f"[load_data]  OK  {len(df):,} rows | "
          f"{df['Station'].nunique()} stations | "
          f"{df['District'].nunique()} districts")
    return df


# ==============================================================================
# MODULE 2 — preprocess_data
# ==============================================================================

def preprocess_data(df):
    """
    1. Parse timestamps (DD-MM-YYYY HH:MM  ->  datetime64)
    2. Cap sensor outliers at SENSOR_CAP_MM
    3. Sort by Station then datetime  (mandatory for correct rolling windows)
    4. Rename rainfall column to 'rainfall'

    Returns
    -------
    pd.DataFrame  clean, sorted
    """
    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["Data Acquisition Time"], dayfirst=True, errors="coerce"
    )
    n_bad = df["datetime"].isnull().sum()
    if n_bad:
        print(f"[preprocess]  WARNING: {n_bad:,} unparseable timestamps dropped.")
        df = df.dropna(subset=["datetime"])

    df = df.rename(columns={"Telemetry Hourly Rainfall (mm)": "rainfall"})

    n_capped = (df["rainfall"] > SENSOR_CAP_MM).sum()
    if n_capped:
        print(f"[preprocess]  WARNING: {n_capped:,} readings > {SENSOR_CAP_MM} mm "
              f"capped (sensor artefacts).")
    df["rainfall"] = df["rainfall"].clip(upper=SENSOR_CAP_MM)

    df = df.sort_values(["Station", "datetime"]).reset_index(drop=True)

    print(f"[preprocess]  OK  Date range: "
          f"{df['datetime'].min().date()} -> {df['datetime'].max().date()}")
    return df


# ==============================================================================
# MODULE 3 — feature_engineering
# ==============================================================================

def feature_engineering(df):
    """
    Enrich the sorted hourly time-series with temporally-aware features.

    Rolling accumulations (per-station, chronological):
        rain_6h   -- 6-hour rolling sum   (short burst intensity)
        rain_24h  -- 24-hour rolling sum  (primary daily flood risk driver)
        rain_72h  -- 72-hour rolling sum  (sustained multi-day event)

    Intensity flags (binary per row):
        is_rainy   : rainfall > 1 mm
        is_heavy   : rainfall > 10 mm  (IMD heavy rain threshold)
        is_extreme : rainfall > 30 mm  (IMD very heavy rain)

    Temporal context:
        month, hour
        is_nemr_monsoon : Oct-Dec (North-East Monsoon -- peak TN flood season)
        is_sw_monsoon   : Jun-Sep (South-West Monsoon)

    Returns
    -------
    pd.DataFrame with new columns appended (same row count)
    """
    df = df.copy()
    grp = df.groupby("Station")["rainfall"]

    df["rain_6h"]  = grp.rolling( 6, min_periods=1).sum().reset_index(level=0, drop=True)
    df["rain_24h"] = grp.rolling(24, min_periods=1).sum().reset_index(level=0, drop=True)
    df["rain_72h"] = grp.rolling(72, min_periods=1).sum().reset_index(level=0, drop=True)

    df["is_rainy"]   = (df["rainfall"] > 1.0).astype(int)
    df["is_heavy"]   = (df["rainfall"] > 10.0).astype(int)
    df["is_extreme"] = (df["rainfall"] > 30.0).astype(int)

    df["month"]           = df["datetime"].dt.month
    df["hour"]            = df["datetime"].dt.hour
    df["is_nemr_monsoon"] = df["month"].isin([10, 11, 12]).astype(int)
    df["is_sw_monsoon"]   = df["month"].isin([6, 7, 8, 9]).astype(int)

    print(f"[feature_eng]  OK  {df.shape[1]} columns after feature engineering")
    return df


# ==============================================================================
# MODULE 4 — aggregate_station_features
# ==============================================================================

def aggregate_station_features(df):
    """
    Collapse the hourly time-series into ONE row per station.

    Features produced
    -----------------
    Magnitude  : avg_rain, std_rain, max_rain, median_rain, total_rain
    Rolling    : max_24h, avg_24h, p95_24h, max_72h
    Frequency  : rain_freq, heavy_rate, extreme_rate
    Seasonal   : nemr_rain_share, sw_rain_share
    Spatial    : latitude, longitude
    Metadata   : district (display only)

    Returns
    -------
    pd.DataFrame  shape (145, ~24 columns)
    """
    def p95(s):
        return s.quantile(0.95)

    frames = []

    frames.append(df.groupby("Station")["rainfall"].agg(
        avg_rain="mean", std_rain="std", max_rain="max",
        median_rain="median", total_rain="sum", total_hours="count",
    ))
    frames.append(df.groupby("Station")["rain_24h"].agg(
        max_24h="max", avg_24h="mean", p95_24h=p95,
    ))
    frames.append(df.groupby("Station")["rain_72h"].agg(max_72h="max"))
    frames.append(df.groupby("Station")["is_rainy"].agg(rainy_hours="sum"))
    frames.append(df.groupby("Station")["is_heavy"].agg(heavy_events="sum"))
    frames.append(df.groupby("Station")["is_extreme"].agg(extreme_events="sum"))
    frames.append(df.groupby("Station")["is_nemr_monsoon"].agg(nemr_hours="sum"))
    frames.append(df.groupby("Station")["is_sw_monsoon"].agg(sw_hours="sum"))
    frames.append(df.groupby("Station")[["Latitude","Longitude"]].first()
                    .rename(columns={"Latitude":"latitude","Longitude":"longitude"}))
    frames.append(df.groupby("Station")["District"].first().rename("district"))

    sdf = pd.concat(frames, axis=1).reset_index()
    sdf["std_rain"] = sdf["std_rain"].fillna(0)

    h = sdf["total_hours"]
    sdf["rain_freq"]       = sdf["rainy_hours"]    / h
    sdf["heavy_rate"]      = sdf["heavy_events"]   / h
    sdf["extreme_rate"]    = sdf["extreme_events"] / h
    sdf["nemr_rain_share"] = sdf["nemr_hours"]     / h
    sdf["sw_rain_share"]   = sdf["sw_hours"]       / h

    print(f"[aggregate]  OK  Station matrix: {sdf.shape}  "
          f"({len(FEATURE_COLS)} model features)")
    return sdf


# ==============================================================================
# MODULE 5 — label_stations
# ==============================================================================

def label_stations(station_df):
    """
    Assign a three-class risk label via a weighted composite percentile score.

    Why relative ranking, not hard thresholds?
    ------------------------------------------
    The dataset only logs non-zero rainfall hours (sensor triggers at 0.5 mm).
    Every station therefore looks "rainy" in absolute terms.  Ranking correctly
    surfaces which stations are genuinely riskier within the TN context.

    Steps
    -----
    1. For each feature in RISK_WEIGHTS, rank all 145 stations.
    2. Normalise rank to [0, 1].
    3. Weighted sum -> composite_score in [0, 1].
    4. Tertile split: Low (<P33) | Moderate (P33-P67) | High (>=P67).
       Produces a balanced ~48 / 49 / 48 distribution.

    Returns
    -------
    pd.DataFrame with new columns: risk_score (float), risk_label (int 0/1/2)
    """
    df = station_df.copy()

    score = np.zeros(len(df))
    for feat, w in RISK_WEIGHTS.items():
        ranks = rankdata(df[feat].values, method="average")
        score += w * (ranks / len(df))
    df["risk_score"] = score

    t33 = np.percentile(score, 33.3)
    t67 = np.percentile(score, 66.7)

    df["risk_label"] = 1
    df.loc[df["risk_score"] < t33,  "risk_label"] = 0
    df.loc[df["risk_score"] >= t67, "risk_label"] = 2

    df.attrs["risk_t33"] = round(t33, 6)
    df.attrs["risk_t67"] = round(t67, 6)

    dist = df["risk_label"].value_counts().sort_index()
    print("[label]  OK  Risk label distribution (tertile composite score):")
    icons = {0: "[LOW]", 1: "[MOD]", 2: "[HI] "}
    for code, name in LABEL_MAP.items():
        n   = dist.get(code, 0)
        pct = n / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"         {icons[code]} {name:>10} ({code}): "
              f"{n:3d} stations  {pct:4.1f}%  {bar}")
    return df


# ==============================================================================
# MODULE 6 — train_model
# ==============================================================================

def train_model(station_df, model_type="both", test_size=0.20):
    """
    Train one or both classifiers on the station-level feature matrix.

    Split strategy
    --------------
    Stratified 80/20 split preserves class proportions.
    No time-leakage risk: features are station-level aggregates,
    not raw hourly rows.

    Parameters
    ----------
    model_type : "gradient_boosting" | "random_forest" | "both"
    test_size  : float (default 0.20)

    Returns
    -------
    model        -- best fitted estimator (by 5-fold CV accuracy)
    X_test       -- held-out feature array
    y_test       -- held-out labels
    feature_imp  -- pd.Series (feature importances, descending)
    """
    X = station_df[FEATURE_COLS].values
    y = station_df["risk_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"[train]  Train: {len(X_train)} | Test: {len(X_test)} stations")

    candidates = {}

    if model_type in ("gradient_boosting", "both"):
        gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,        # reduced from 4 to reduce overfitting
        min_samples_leaf=4, # increased from 2 to reduce overfitting
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gb_calibrated = CalibratedClassifierCV(gb, cv=5, method="isotonic")
    gb_calibrated.fit(X_train, y_train)
    cv_score = cross_val_score(gb_calibrated, X, y, cv=cv, scoring="accuracy").mean()
    candidates["GradientBoosting"] = (gb_calibrated, cv_score)

    if model_type in ("random_forest", "both"):
        rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,            # limit depth to reduce overfitting
        min_samples_leaf=3,     # increased to reduce overfitting
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    rf_calibrated = CalibratedClassifierCV(rf, cv=5, method="isotonic")
    rf_calibrated.fit(X_train, y_train)
    cv_score = cross_val_score(rf_calibrated, X, y, cv=cv, scoring="accuracy").mean()
    candidates["RandomForest"] = (rf_calibrated, cv_score)

    best_name, (best_model, best_cv) = max(
        candidates.items(), key=lambda kv: kv[1][1]
    )
    print(f"[train]  OK  Selected: {best_name}  (CV={best_cv:.4f})")

    # Extract the base estimator from the calibrated wrapper
    if hasattr(best_model, "calibrated_classifiers_"):
        base_estimator = best_model.calibrated_classifiers_[0].estimator
        feature_imp = pd.Series(
            base_estimator.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
    else:
        feature_imp = pd.Series(
            best_model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)


    return best_model, X_test, y_test, feature_imp


# ==============================================================================
# MODULE 7 — evaluate_model
# ==============================================================================

def evaluate_model(model, X_test, y_test, feature_imp):
    """
    Print and return a comprehensive evaluation report.

    Returns
    -------
    dict  {accuracy, confusion_matrix, report, feature_importances}
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=LABEL_NAMES, zero_division=0
    )

    print("\n" + "=" * 62)
    print("  MODEL EVALUATION")
    print("=" * 62)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f} %)")

    print("\n  Confusion Matrix  (rows = Actual, cols = Predicted)")
    print("  " + "-" * 44)
    cm_df = pd.DataFrame(
        cm,
        index   =[f"  Actual {n:<9}" for n in LABEL_NAMES],
        columns =[f"Pred {n}" for n in LABEL_NAMES],
    )
    print(cm_df.to_string())

    print("\n  Classification Report:")
    for line in report.strip().split("\n"):
        print(f"    {line}")

    print("\n  Feature Importances (top 10):")
    for feat, imp in feature_imp.head(10).items():
        bar = "#" * max(1, int(imp * 120))
        print(f"    {feat:<22}  {imp:.4f}  {bar}")
    print("=" * 62 + "\n")

    return {
        "accuracy":            acc,
        "confusion_matrix":    cm.tolist(),
        "report":              report,
        "feature_importances": feature_imp.to_dict(),
    }


# ==============================================================================
# MODULE 8 — predict_risk  (human-readable station card)
# ==============================================================================

def predict_risk(station_name, model, station_df, verbose=True):
    """
    Predict and display a formatted risk card for one station.

    Lookup is case-insensitive; falls back to partial substring match.

    Parameters
    ----------
    station_name : str   station name (exact or partial, case-insensitive)
    model        :       fitted sklearn estimator
    station_df   :       output of aggregate_station_features()
    verbose      : bool  print the card (default True)

    Returns
    -------
    str  "Low" | "Moderate" | "High"
    """
    row = _lookup_station(station_name, station_df)

    X_row = row[FEATURE_COLS].values.reshape(1, -1)
    code  = int(model.predict(X_row)[0])
    proba = model.predict_proba(X_row)[0]
    risk  = LABEL_MAP[code]

    if verbose:
        print("\n" + "-" * 56)
        print(f"  Weather Station Risk Report")
        print("-" * 56)
        print(f"  Station     : {row['Station']}")
        print(f"  District    : {row['district']}")
        print(f"  Location    : {row['latitude']:.4f}N  {row['longitude']:.4f}E")
        print(f"  Data points : {int(row['total_hours']):,} recorded rain-hours")
        print()
        print(f"  Avg rainfall     : {row['avg_rain']:.2f} mm/hr")
        print(f"  Max 24-h sum     : {row['max_24h']:.0f} mm")
        print(f"  Max 72-h sum     : {row['max_72h']:.0f} mm")
        print(f"  Heavy events     : {int(row['heavy_events']):,}  "
              f"({row['heavy_rate']*100:.1f}% of observed hours)")
        print(f"  Extreme events   : {int(row['extreme_events']):,}  "
              f"({row['extreme_rate']*100:.1f}% of observed hours)")
        print(f"  Rainy frequency  : {row['rain_freq']*100:.1f}%")
        print(f"  NE monsoon share : {row['nemr_rain_share']*100:.1f}%  "
              f"  SW monsoon: {row['sw_rain_share']*100:.1f}%")
        print()
        print(f"  +------------------------------------------+")
        print(f"  |  Predicted Risk : [{risk.upper()}]              |")
        print(f"  |  Confidence     : {max(proba)*100:5.1f}%                  |")
        prob_str = "   ".join(
            f"{LABEL_NAMES[i]}: {proba[i]*100:.0f}%" for i in range(3)
        )
        print(f"  |  Scores  ->  {prob_str}    |")
        print(f"  +------------------------------------------+")
        print()

    return risk


# ==============================================================================
# MODULE 9 — predict_station_risk  (JSON / API output)
# ==============================================================================

def predict_station_risk(station, model, station_df):
    """
    API-compatible prediction function with proper probability outputs.
    """
    try:
        row = _lookup_station(station, station_df)
    except ValueError as e:
        return {"error": str(e)}

    X_row = row[FEATURE_COLS].values.reshape(1, -1)

    # Predictions
    code  = int(model.predict(X_row)[0])
    proba = model.predict_proba(X_row)[0]   # ✅ FIXED

    risk  = LABEL_MAP[code]

    return {
        "station":    row["Station"],
        "district":   row["district"],
        "latitude":   round(float(row["latitude"]),  4),
        "longitude":  round(float(row["longitude"]), 4),

        "risk_level": risk,
        "risk_code":  code,

        # Keep interpretable score
        "risk_score": round(float(row["risk_score"]), 4),

        # Model confidence
        "confidence": round(float(max(proba)), 4),

        # Clean probability output (single source of truth)
        "probabilities": {
            "Low":      round(float(proba[0]), 4),
            "Moderate": round(float(proba[1]), 4),
            "High":     round(float(proba[2]), 4),
        },

        # Flat columns (useful for frontend + CSV)
        "prob_low":      round(float(proba[0]), 4),
        "prob_moderate": round(float(proba[1]), 4),
        "prob_high":     round(float(proba[2]), 4),

        "key_features": {
            "avg_rain_mmhr": round(float(row["avg_rain"]), 2),
            "max_24h_mm":    round(float(row["max_24h"]), 1),
            "heavy_pct":     f"{row['heavy_rate']*100:.1f}%",
            "extreme_pct":   f"{row['extreme_rate']*100:.1f}%",
            "rainy_pct":     f"{row['rain_freq']*100:.1f}%",
        },
    }

def export_probabilities_csv(model, station_df, output_path="outputs/risk_probabilities.csv"):
    rows = []
    for _, row in station_df.iterrows():
        X_row  = row[FEATURE_COLS].values.reshape(1, -1)
        code   = int(model.predict(X_row)[0])
        proba  = model.predict_proba(X_row)[0]

        rows.append({
            "station":       row["Station"],
            "district":      row["district"],
            "latitude":      round(float(row["latitude"]),  4),
            "longitude":     round(float(row["longitude"]), 4),
            "predicted_label": LABEL_MAP[code],
            "risk_score":    round(float(row["risk_score"]), 4),
            "prob_low":      round(float(proba[0]), 4),
            "prob_moderate": round(float(proba[1]), 4),
            "prob_high":     round(float(proba[2]), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[export]  OK  Probability scores -> {output_path}")
    return df

# -- internal lookup helper ---------------------------------------------------

def _lookup_station(name, station_df):
    """Case-insensitive exact -> substring station lookup."""
    nl   = name.strip().lower()
    mask = station_df["Station"].str.lower() == nl
    if not mask.any():
        mask = station_df["Station"].str.lower().str.contains(nl, na=False)
    if not mask.any():
        sample = station_df["Station"].tolist()[:12]
        raise ValueError(
            f"Station '{name}' not found.\n"
            f"Sample available stations: {sample} ..."
        )
    return station_df[mask].iloc[0]


# ==============================================================================
# MODEL PERSISTENCE
# ==============================================================================

def save_model(model, station_df, model_path=MODEL_PATH, features_path=FEATURES_PATH):
    """Save fitted model + feature table to disk."""
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS}, model_path)
    station_df.to_csv(features_path, index=False)
    print(f"[save]  OK  Model    -> {model_path}")
    print(f"[save]  OK  Features -> {features_path}")


def load_model(model_path="models/weather_risk_model.joblib", features_path="models/station_features.csv"):
    """Reload model and station features from disk."""
    bundle     = joblib.load(model_path)
    model      = bundle["model"]
    station_df = pd.read_csv(features_path)
    print("[load]  OK  Model and features loaded from disk.")
    return model, station_df


# ==============================================================================
# BONUS -- Supabase Integration
# ==============================================================================

SUPABASE_SQL_SCHEMA = """\
-- ===========================================================================
--  Supabase SQL Schema  (paste into the Supabase SQL editor)
-- ===========================================================================

CREATE EXTENSION IF NOT EXISTS postgis;   -- for geo proximity queries

CREATE TABLE weather_stations (
    station_id   SERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    district     TEXT,
    latitude     FLOAT8,
    longitude    FLOAT8,
    risk_level   TEXT CHECK (risk_level IN ('Low','Moderate','High')),
    risk_code    SMALLINT,
    confidence   FLOAT4,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE station_features (
    station_id    INT PRIMARY KEY REFERENCES weather_stations(station_id),
    avg_rain      FLOAT4,
    max_24h       FLOAT4,
    heavy_rate    FLOAT4,
    extreme_rate  FLOAT4,
    rain_freq     FLOAT4,
    risk_score    FLOAT4
);

-- Fast lookups
CREATE INDEX idx_risk_level ON weather_stations(risk_level);
CREATE INDEX idx_station_geo ON weather_stations
    USING GIST (ST_MakePoint(longitude, latitude));

-- ============================================================
-- Geo query: High-risk stations within 50 km of a rider
-- ============================================================
SELECT
    name, district, risk_level, confidence,
    ST_Distance(
        ST_MakePoint(longitude, latitude)::geography,
        ST_MakePoint(80.27, 13.08)::geography   -- rider GPS coords
    ) / 1000 AS distance_km
FROM weather_stations
WHERE risk_level = 'High'
  AND ST_DWithin(
      ST_MakePoint(longitude, latitude)::geography,
      ST_MakePoint(80.27, 13.08)::geography,
      50000    -- 50 km radius
  )
ORDER BY distance_km;
"""

SUPABASE_PYTHON_CLIENT = """\
# ===========================================================================
#  Supabase Python Client  (pip install supabase)
# ===========================================================================
from supabase import create_client
from weather_risk_predictor import load_model, predict_station_risk

SUPABASE_URL = "https://<your-project>.supabase.co"
SUPABASE_KEY = "<your-service-role-key>"

supabase   = create_client(SUPABASE_URL, SUPABASE_KEY)
model, sdf = load_model()


def get_all_stations():
    \"\"\"Fetch station list from Supabase.\"\"\"
    return supabase.table("weather_stations").select("*").execute().data


def predict_and_store(station_name: str) -> dict:
    \"\"\"Predict risk for one station and upsert into Supabase.\"\"\"
    result = predict_station_risk(station_name, model, sdf)
    if "error" in result:
        return result
    supabase.table("weather_stations").upsert({
        "name":         result["station"],
        "district":     result["district"],
        "latitude":     result["latitude"],
        "longitude":    result["longitude"],
        "risk_level":   result["risk_level"],
        "risk_code":    result["risk_code"],
        "confidence":   result["confidence"],
        "last_updated": "now()",
    }, on_conflict="name").execute()
    return result


def refresh_all_risks():
    \"\"\"Batch-predict and store risk for every station.\"\"\"
    for name in sdf["Station"].tolist():
        predict_and_store(name)
    print(f"Refreshed {len(sdf)} stations in Supabase.")


def get_risk_near_rider(lat: float, lon: float, radius_km: float = 30) -> list:
    \"\"\"Find risk levels of stations near a delivery rider (uses PostGIS RPC).\"\"\"
    return supabase.rpc("stations_within_radius", {
        "rider_lat": lat,
        "rider_lon": lon,
        "radius_m":  radius_km * 1000,
    }).execute().data
"""

SUPABASE_EDGE_FUNCTION = """\
// ===========================================================================
//  Supabase Edge Function  (Deno / TypeScript)
//  Deploy: supabase functions deploy predict-risk
// ===========================================================================
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const ML_API = Deno.env.get("ML_API_URL")!          // your FastAPI endpoint
const SB_URL = Deno.env.get("SUPABASE_URL")!
const SB_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!

serve(async (req: Request) => {
  const { station } = await req.json()

  // 1. Call ML prediction API
  const mlRes  = await fetch(`${ML_API}/predict`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ station }),
  })
  const result = await mlRes.json()

  // 2. Upsert prediction into weather_stations table
  await fetch(`${SB_URL}/rest/v1/weather_stations`, {
    method:  "POST",
    headers: {
      "apikey":        SB_KEY,
      "Authorization": `Bearer ${SB_KEY}`,
      "Content-Type":  "application/json",
      "Prefer":        "resolution=merge-duplicates",
    },
    body: JSON.stringify({
      name:         result.station,
      district:     result.district,
      latitude:     result.latitude,
      longitude:    result.longitude,
      risk_level:   result.risk_level,
      risk_code:    result.risk_code,
      confidence:   result.confidence,
      last_updated: new Date().toISOString(),
    }),
  })

  return new Response(JSON.stringify(result), {
    headers: { "Content-Type": "application/json" },
  })
})
"""


def supabase_integration_demo(station_df, model):
    """
    Simulate the full Supabase integration flow (no live connection required):
    fetch stations -> predict all -> build upsert payload -> show summary.
    """
    print("\n" + "=" * 62)
    print("  SUPABASE INTEGRATION DEMO  (simulated)")
    print("=" * 62)

    stations = station_df["Station"].tolist()
    print(f"  Simulating fetch of {len(stations)} stations from Supabase...\n")

    rows = [predict_station_risk(s, model, station_df) for s in stations]
    upsert_df = pd.DataFrame(rows)
    dist = upsert_df["risk_level"].value_counts()

    print("  Risk distribution across all stations:")
    for lvl in ["High", "Moderate", "Low"]:
        tag = {"High": "[HIGH]", "Moderate": "[MOD] ", "Low": "[LOW] "}[lvl]
        print(f"    {tag}  {lvl:<10}: {dist.get(lvl,0):3d} stations")

    print(f"\n  Sample upsert rows (3 of {len(rows)}):")
    cols = ["station", "district", "risk_level", "risk_code", "confidence"]
    print(upsert_df[cols].head(3).to_string(index=False))

    print("\n  In production code:")
    print("    supabase.table('weather_stations')")
    print("           .upsert(payload, on_conflict='name').execute()")

    print("\n\n--- SQL SCHEMA ---\n")
    print(SUPABASE_SQL_SCHEMA)
    print("\n--- PYTHON CLIENT CODE ---\n")
    print(SUPABASE_PYTHON_CLIENT)
    print("\n--- EDGE FUNCTION (TypeScript/Deno) ---\n")
    print(SUPABASE_EDGE_FUNCTION)


# ==============================================================================
# FULL PIPELINE ORCHESTRATOR
# ==============================================================================

def run_pipeline(data_path=DATA_PATH):
    """
    End-to-end orchestration:
        load -> preprocess -> features -> aggregate -> label -> train -> evaluate

    Returns
    -------
    model, station_df, eval_results
    """
    print("\n" + "#" * 62)
    print("  WEATHER STATION RISK PREDICTOR  --  FULL PIPELINE")
    print("#" * 62 + "\n")

    df         = load_data(data_path)
    df         = preprocess_data(df)
    df         = feature_engineering(df)
    station_df = aggregate_station_features(df)
    station_df = label_stations(station_df)

    model, X_test, y_test, feature_imp = train_model(
        station_df, model_type="both"
    )
    eval_results = evaluate_model(model, X_test, y_test, feature_imp)
    save_model(model, station_df)

    return model, station_df, eval_results


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    # 1. Full pipeline
    model, station_df, eval_results = run_pipeline()

    # 2. Human-readable predictions
    '''demo_stations = [
        "Anaikidangu",                    # Kanyakumari coastal
        "Thiruvadanai_1",                 # highest avg rainfall
        "Pattukottai_1",                  # lowest avg rainfall
        "Sholaiyur",                      # highest data volume
        "Chennai Mylapore (DGPOffice)",   # urban metro
        "Valparai",                       # Nilgiris high-altitude
        "Kodaikanal",                     # partial match test
    ]

    print("\n" + "#" * 62)
    print("  SAMPLE PREDICTIONS -- predict_risk()")
    print("#" * 62)
    for s in demo_stations:
        try:
            predict_risk(s, model, station_df)
        except ValueError as e:
            print(f"  WARNING: {e}\n")'''

    # 3. JSON / API output
    print("\n" + "#" * 62)
    print("  API JSON OUTPUT -- predict_station_risk()")
    print("#" * 62)
    for s in ["Anaikidangu", "Thiruvadanai_1", "Pattukottai_1"]:
        result = predict_station_risk(s, model, station_df)
        print(f"\n  >>> predict_station_risk('{s}')")
        print(json.dumps(result, indent=4))

    # 4. Export risk summary CSV
    summary = station_df[[
        "Station", "district", "latitude", "longitude",
        "avg_rain", "max_24h", "heavy_rate", "extreme_rate",
        "rain_freq", "risk_score", "risk_label",
    ]].copy()
    summary["risk_level"] = summary["risk_label"].map(LABEL_MAP)
    summary = summary.sort_values("risk_score", ascending=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\n[export]  OK  Station risk summary -> {SUMMARY_PATH}")

    # Export probability scores for all stations
    prob_df = export_probabilities_csv(model, station_df)
    print(prob_df[["station", "predicted_label", "prob_low", "prob_moderate", "prob_high"]].head(10).to_string(index=False))

    # 5. Supabase integration demo
    # supabase_integration_demo(station_df, model)

    # 6. Interactive prediction
    print("\n" + "#" * 62)
    print("  INTERACTIVE PREDICTION")
    print("#" * 62)
    print("  Type a station name to get its risk level.")
    print("  Type 'quit' to exit.\n")

    while True:
        station_input = input("  Enter station name: ").strip()
        if station_input.lower() == "quit":
            print("  Exiting. Bye!")
            break
        result = predict_station_risk(station_input, model, station_df)
        if "error" in result:
            print(f"  {result['error']}\n")
        else:
            print(f"\n  Station    : {result['station']}")
            print(f"  District   : {result['district']}")
            print(f"  Risk Level : {result['risk_level']}")

            print(f"\n  --- Probabilities ---")
            print(f"  Low      : {result['prob_low']:.4f}")
            print(f"  Moderate : {result['prob_moderate']:.4f}")
            print(f"  High     : {result['prob_high']:.4f}")

            print(f"\n  Confidence (max prob): {result['confidence']:.4f}")

            # Optional (VERY good for demo)
            print(f"  Risk Score (interpretable): {result['risk_score']}\n")
