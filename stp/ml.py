"""
ml.py - The AI / machine-learning parts of the plant.

Two models, both trained ONLY on a separate month of normal sewage
(influent.training_dataset). They never see the demo month or its events.

1. Anomaly detector (Isolation Forest + robust z-scores, unsupervised)
   Learns what normal raw sewage looks like and flags hours that look
   unusual. The MonitoringAgent uses it; when it raises an alarm the
   EqualizationAgent can divert the shock to the holding tank.

2. Load forecaster (Ridge regression, supervised)
   Predicts next hour's inflow, BOD and ammonia from the recent past and
   the time of day. Used only for quantities where it beats "same as last
   hour" on held-out training data. The AerationAgent uses it to set the blowers BEFORE the
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
# Default limits, chosen for typical Indian domestic sewage (config.RAW_SEWAGE).
DEFAULT_RULE_LIMITS = {
    "pH_low": 6.0,        # below this, bacteria are harmed
    "pH_high": 9.0,
    "cod_bod": 2.6,       # domestic sewage is ~1.8-2.2; higher = non-biodegradable (industrial) COD
    "COD": 1000.0,        # mg/L
    "TSS": 700.0,         # mg/L
    "NH4": 60.0,          # mg/L
    "flow": 1.8,          # x normal flow for this hour of the day
}

# Limits that describe the LOAD can be re-tuned to a plant's own normal data
# (calibrate_rule_limits). pH limits protect the bacteria, so they never change.
CALIBRATABLE = {"cod_bod": "cod_bod_ratio", "COD": "COD", "TSS": "TSS", "NH4": "NH4", "flow": "flow_vs_typical"}


def make_rules(limits=None):
    L = {**DEFAULT_RULE_LIMITS, **(limits or {})}
    return {
        f"pH below {L['pH_low']:g}": lambda f: f["pH"] < L["pH_low"],
        f"pH above {L['pH_high']:g}": lambda f: f["pH"] > L["pH_high"],
        f"COD/BOD above {L['cod_bod']:g} (non-biodegradable)": lambda f: f["cod_bod_ratio"] > L["cod_bod"],
        f"COD above {L['COD']:g} mg/L": lambda f: f["COD"] > L["COD"],
        f"TSS above {L['TSS']:g} mg/L": lambda f: f["TSS"] > L["TSS"],
        f"Ammonia above {L['NH4']:g} mg/L": lambda f: f["NH4"] > L["NH4"],
        f"Flow above {L['flow']:g}x normal for this hour": lambda f: f["flow_vs_typical"] > L["flow"],
    }


RULES = make_rules()


def calibrate_rule_limits(train, quantile=0.995):
    """Site calibration: raise each load limit to the level that the plant's own
    NORMAL data reaches (its 99.5th percentile), never lowering the default.
    Example: the real Manresa sewage often has COD/BOD 2.6-3.0 on ordinary days,
    so the limit moves up from 2.6 to what is normal there."""
    f = anomaly_features(train)
    limits = dict(DEFAULT_RULE_LIMITS)
    for key, col in CALIBRATABLE.items():
        limits[key] = round(max(DEFAULT_RULE_LIMITS[key], float(f[col].quantile(quantile))), 2)
    return limits


_RULE_CACHE = {}


def rule_check(features_row, limits=None):
    """Return the list of rules broken by one row of features (a dict)."""
    key = tuple(sorted((limits or {}).items()))
    if key not in _RULE_CACHE:
        _RULE_CACHE[key] = make_rules(limits)
    return [name for name, rule in _RULE_CACHE[key].items() if rule(features_row)]


def classify_event(f, limits=None):
    """Name the most likely cause of an anomaly, from the features.
    The control agents choose their action from this."""
    L = {**DEFAULT_RULE_LIMITS, **(limits or {})}
    if f["pH"] < L["pH_low"] and f["cod_bod_ratio"] > L["cod_bod"]:
        return "industrial_discharge"
    if f["pH"] < L["pH_low"]:
        return "acid_dump"
    if f["pH"] > L["pH_high"]:
        return "alkaline_dump"
    if f["cod_bod_ratio"] > L["cod_bod"]:
        return "industrial_discharge"
    if f["BOD"] > 700 or f["TSS"] > 1000 or f["NH4"] > 80:
        return "septage_dumping"
    if f["flow_vs_typical"] > L["flow"]:
        return "storm_inflow"
    return "unusual_load"


# ---------------------------------------------------------------------------
# ML anomaly detector (Isolation Forest)
# ---------------------------------------------------------------------------
class AnomalyDetector:
    """Two learned checks, both trained only on normal data:

    1. Isolation Forest (multivariate): catches unusual COMBINATIONS, e.g.
       strong BOD with normal pH and flow.
    2. Robust z-score per feature (univariate): how many 'typical deviations'
       each reading is from its normal median. An Isolation Forest scores a
       value far outside the training range no worse than the most extreme
       training point, so on its own it misses single-parameter shocks such
       as an acid dump (pH 3). This check covers that gap.

    Each check gets half of the allowed false-alarm rate. The combined score is
    'times the alarm threshold': above 1.0 = anomaly."""

    def __init__(self, seed=42):
        self.model = IsolationForest(n_estimators=150, random_state=seed)

    def _zmax(self, X):
        return np.max(np.abs(X - self.median) / self.scale, axis=1)

    def fit(self, df):
        X = anomaly_features(df).to_numpy()
        self.model.fit(X)
        rate = ML_FALSE_ALARM_RATE / 2
        if_scores = -self.model.score_samples(X)
        self.if_threshold = float(np.quantile(if_scores, 1 - rate))
        self.median = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median), axis=0)
        self.scale = np.maximum(1.4826 * mad, 1e-6)
        self.z_threshold = float(np.quantile(self._zmax(X), 1 - rate))
        self.threshold = 1.0
        return self

    def parts(self, df):
        """Isolation-Forest ratio and z-score ratio for every row (1.0 = threshold)."""
        X = anomaly_features(df).to_numpy()
        return (-self.model.score_samples(X) / self.if_threshold, self._zmax(X) / self.z_threshold)

    def scores(self, df):
        """Combined anomaly score for every row (vectorised; fast). > 1.0 = alarm."""
        a, b = self.parts(df)
        return np.maximum(a, b)

    def normalised(self, score):
        """Map a score to 0..1 where 0.5 = the alarm threshold. For display only."""
        return float(max(0.0, min(1.0, 0.5 * score)))


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
    """Ridge regression per quantity. It checks itself on the last quarter of its
    training data and is only USED for a quantity if it beats the simple
    'same as last hour' guess there (self.use_ml). Otherwise the plant uses the
    last hour's value for that quantity."""

    def __init__(self):
        self.models = {t: Ridge(alpha=1.0) for t in FORECAST_TARGETS}
        self.use_ml = {t: True for t in FORECAST_TARGETS}
        self.validation = {}

    @staticmethod
    def _xy(df):
        X, Y, last = [], {t: [] for t in FORECAST_TARGETS}, {t: [] for t in FORECAST_TARGETS}
        times = pd.to_datetime(df["timestamp"])
        for i in range(24, len(df)):
            X.append(forecast_features(df.iloc[i - 24:i], times.iloc[i]))
            for t in FORECAST_TARGETS:
                Y[t].append(df[t].iloc[i])
                last[t].append(df[t].iloc[i - 1])
        return np.array(X), {t: np.array(v) for t, v in Y.items()}, {t: np.array(v) for t, v in last.items()}

    def fit(self, df):
        X, Y, last = self._xy(df)
        cut = int(len(X) * 0.75)
        for t in FORECAST_TARGETS:
            trial = Ridge(alpha=1.0).fit(X[:cut], Y[t][:cut])
            ml_err = float(np.mean(np.abs(trial.predict(X[cut:]) - Y[t][cut:])))
            naive_err = float(np.mean(np.abs(last[t][cut:] - Y[t][cut:])))
            self.use_ml[t] = ml_err < naive_err
            self.validation[t] = {"ml_mae": round(ml_err, 2), "last_hour_mae": round(naive_err, 2),
                                  "used": "ML" if self.use_ml[t] else "last hour"}
            self.models[t].fit(X, Y[t])
        return self

    def predict(self, history, next_time):
        x = np.array([forecast_features(history, next_time)])
        out = {}
        for t in FORECAST_TARGETS:
            if self.use_ml[t]:
                out[t] = max(0.0, float(self.models[t].predict(x)[0]))
            else:
                out[t] = float(history[t].iloc[-1])
        return out

    def source(self):
        ml = [t for t in FORECAST_TARGETS if self.use_ml[t]]
        last = [t for t in FORECAST_TARGETS if not self.use_ml[t]]
        parts = []
        if ml:
            parts.append("ML: " + ", ".join(ml))
        if last:
            parts.append("last hour: " + ", ".join(last))
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Train once and reuse (training takes about a second)
# ---------------------------------------------------------------------------
_CACHE = {}


def get_models(train=None, key="simulated"):
    """Return (detector, forecaster), trained once per training set and cached.

    train=None -> the simulated normal month (influent.training_dataset).
    For real data pass the real training days and a name for them, e.g.
    get_models(real_data.uci_train(), key="uci"), so the models learn what
    normal sewage looks like AT THAT PLANT."""
    if key not in _CACHE:
        if train is None:
            train = training_dataset()
        _CACHE[key] = (AnomalyDetector().fit(train), LoadForecaster().fit(train))
    return _CACHE[key]
