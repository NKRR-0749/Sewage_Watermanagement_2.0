"""
ml.py - The AI / machine-learning parts of the plant.

Two models, both trained ONLY on a separate month of normal sewage
(influent.training_dataset). They never see the demo month or its events.

1. Anomaly detector (Isolation Forest, unsupervised)
   Learns what normal raw sewage looks like and flags hours that look
   unusual. The MonitoringAgent uses it; when it raises an alarm the
   EqualizationAgent can divert the shock to the holding tank.

2. Load forecaster (Ridge regression, supervised)
   Predicts next hour's inflow, BOD and ammonia from the recent past and
   the time of day. The AerationAgent uses it to set the blowers BEFORE the
   load arrives (feed-forward control) instead of reacting late.

A plain rule-based detector is also here so the two detectors can be compared.
"""

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

from .config import DESIGN_FLOW
from .influent import DIURNAL_FLOW, training_dataset

ANOMALY_FEATURES = ["flow_vs_typical", "pH", "BOD", "COD", "TSS", "NH4", "cod_bod_ratio", "temperature"]
FORECAST_TARGETS = ["flow", "BOD", "NH4"]

# Target false-alarm rate on normal training data for the ML detector
ML_FALSE_ALARM_RATE = 0.01


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def anomaly_features(df):
    """Turn raw influent rows into the numbers the detectors look at."""
    hours = pd.to_datetime(df["timestamp"]).dt.hour.to_numpy()
    out = pd.DataFrame({
        # flow compared with what is normal for this hour of the day
        "flow_vs_typical": df["flow"].to_numpy() / (DESIGN_FLOW * DIURNAL_FLOW[hours]),
        "pH": df["pH"].to_numpy(),
        "BOD": df["BOD"].to_numpy(),
        "COD": df["COD"].to_numpy(),
        "TSS": df["TSS"].to_numpy(),
        "NH4": df["NH4"].to_numpy(),
        "cod_bod_ratio": df["COD"].to_numpy() / np.maximum(df["BOD"].to_numpy(), 1.0),
        "temperature": df["temperature"].to_numpy(),
    })
    return out[ANOMALY_FEATURES]


# ---------------------------------------------------------------------------
# Rule-based detector (the baseline)
# ---------------------------------------------------------------------------
RULES = {
    "pH below 6.0": lambda f: f["pH"] < 6.0,
    "pH above 9.0": lambda f: f["pH"] > 9.0,
    "COD/BOD above 2.6 (non-biodegradable)": lambda f: f["cod_bod_ratio"] > 2.6,
    "COD above 1000 mg/L": lambda f: f["COD"] > 1000,
    "TSS above 700 mg/L": lambda f: f["TSS"] > 700,
    "Ammonia above 60 mg/L": lambda f: f["NH4"] > 60,
    "Flow above 1.8x normal for this hour": lambda f: f["flow_vs_typical"] > 1.8,
}


def rule_check(features_row):
    """Return the list of rules broken by one row of features (a dict)."""
    return [name for name, rule in RULES.items() if rule(features_row)]


def classify_event(f):
    """Name the most likely cause of an anomaly, from the features.
    The control agents choose their action from this."""
    if f["pH"] < 6.0 and f["cod_bod_ratio"] > 2.6:
        return "industrial_discharge"
    if f["pH"] < 6.0:
        return "acid_dump"
    if f["pH"] > 9.0:
        return "alkaline_dump"
    if f["cod_bod_ratio"] > 2.6:
        return "industrial_discharge"
    if f["BOD"] > 700 or f["TSS"] > 1000 or f["NH4"] > 80:
        return "septage_dumping"
    if f["flow_vs_typical"] > 1.8:
        return "storm_inflow"
    return "unusual_load"


# ---------------------------------------------------------------------------
# ML anomaly detector (Isolation Forest)
# ---------------------------------------------------------------------------
class AnomalyDetector:
    def __init__(self, seed=42):
        self.model = IsolationForest(n_estimators=150, random_state=seed)
        self.threshold = None

    def fit(self, df):
        X = anomaly_features(df).to_numpy()
        self.model.fit(X)
        # Higher score = more unusual. Set the alarm level so that only
        # ML_FALSE_ALARM_RATE of normal training hours would raise an alarm.
        scores = -self.model.score_samples(X)
        self.threshold = float(np.quantile(scores, 1 - ML_FALSE_ALARM_RATE))
        self.train_score_range = (float(scores.min()), float(scores.max()))
        return self

    def scores(self, df):
        """Anomaly score for every row (vectorised; fast)."""
        return -self.model.score_samples(anomaly_features(df).to_numpy())

    def normalised(self, score):
        """Map a raw score to 0..1 where 0.5 = the alarm threshold. For display only."""
        lo = self.train_score_range[0]
        return float(max(0.0, min(1.0, 0.5 * (score - lo) / max(self.threshold - lo, 1e-9))))


# ---------------------------------------------------------------------------
# Load forecaster (Ridge regression)
# ---------------------------------------------------------------------------
def forecast_features(history, next_time):
    """Features for predicting the hour at next_time.
    history: DataFrame of past rows (most recent last), at least 24 rows."""
    h = next_time.hour
    feats = [math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
             1.0 if next_time.dayofweek == 6 else 0.0, DIURNAL_FLOW[h]]
    for target in FORECAST_TARGETS:
        s = history[target].to_numpy()
        feats += [s[-1], s[-2], s[-24], s[-24:].mean()]
    return feats


class LoadForecaster:
    def __init__(self):
        self.models = {t: Ridge(alpha=1.0) for t in FORECAST_TARGETS}

    def fit(self, df):
        X, Y = [], {t: [] for t in FORECAST_TARGETS}
        times = pd.to_datetime(df["timestamp"])
        for i in range(24, len(df)):
            X.append(forecast_features(df.iloc[i - 24:i], times.iloc[i]))
            for t in FORECAST_TARGETS:
                Y[t].append(df[t].iloc[i])
        X = np.array(X)
        for t in FORECAST_TARGETS:
            self.models[t].fit(X, np.array(Y[t]))
        return self

    def predict(self, history, next_time):
        x = np.array([forecast_features(history, next_time)])
        return {t: max(0.0, float(self.models[t].predict(x)[0])) for t in FORECAST_TARGETS}


# ---------------------------------------------------------------------------
# Train once and reuse (training takes about a second)
# ---------------------------------------------------------------------------
_CACHE = {}


def get_models():
    if "models" not in _CACHE:
        train = training_dataset()
        _CACHE["models"] = (AnomalyDetector().fit(train), LoadForecaster().fit(train))
    return _CACHE["models"]
