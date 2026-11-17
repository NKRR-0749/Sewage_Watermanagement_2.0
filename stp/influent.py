"""
influent.py - The data that is FED into the plant instead of live sensors.

Every hour the plant needs to know what raw sewage is arriving: how much
(flow) and how dirty (pH, BOD, COD, TSS, ammonia, coliforms, temperature).
In a real plant that comes from inlet analysers. Here it comes from a table
of hourly readings (a CSV file). Two sources are supported:

1. generate_influent()  - builds a realistic dataset for an Indian 10 MLD
                          plant: daily flow peaks, monsoon storms and
                          "shock" events such as an industrial discharge.
                          Every row is labelled with the event that caused
                          it, so the anomaly detectors can be graded honestly.
2. load_influent_csv()  - reads your own CSV with the same columns.

IMPORTANT: the generated data is SIMULATED, built from typical published
Indian raw sewage values (see config.RAW_SEWAGE). It is not measured data.
"""

import io
import math

import numpy as np
import pandas as pd

from .config import RAW_SEWAGE, DESIGN_FLOW
from .water import Water

COLUMNS = ["timestamp", "flow", "pH", "temperature", "BOD", "COD", "TSS", "NH4", "FC", "event"]
REQUIRED = ["flow", "pH", "BOD", "COD", "TSS", "NH4"]

# Share of the daily average flow arriving in each hour (00:00 ... 23:00).
# Indian towns: very low at night, a big morning peak, a smaller evening peak.
DIURNAL_FLOW = np.array([
    0.55, 0.48, 0.45, 0.45, 0.50, 0.70, 1.05, 1.50, 1.65, 1.55, 1.35, 1.20,
    1.10, 1.05, 0.98, 0.92, 0.92, 0.98, 1.10, 1.28, 1.32, 1.18, 0.95, 0.74,
])
DIURNAL_FLOW = DIURNAL_FLOW / DIURNAL_FLOW.mean()

# What each kind of shock event does to the sewage. These are the events the
# anomaly detectors must catch.
EVENT_TYPES = {
    "industrial_discharge": "Factory effluent: non-biodegradable COD, acidic, toxic to bacteria",
    "acid_dump": "Acid poured into the sewer (very low pH)",
    "alkaline_dump": "Alkaline / caustic discharge (very high pH)",
    "storm_inflow": "Monsoon rain entering the sewer: 2-3x flow, diluted sewage",
    "septage_dumping": "Septic-tank trucks emptying into a manhole: extremely strong sewage",
    "high_strength_food": "Food-processing waste: stronger organic load (subtle)",
}

# The demo month (July = monsoon). (day, start hour, duration in hours, event)
DEMO_EVENTS = [
    (3, 9, 6, "industrial_discharge"),
    (6, 15, 8, "storm_inflow"),
    (9, 2, 3, "acid_dump"),
    (12, 11, 4, "septage_dumping"),
    (15, 13, 5, "high_strength_food"),
    (18, 22, 4, "alkaline_dump"),
    (21, 5, 14, "storm_inflow"),
    (24, 10, 5, "industrial_discharge"),
    (27, 1, 3, "acid_dump"),
]


def apply_event(row, event, hour_into_event=0):
    """Change one hourly reading (a dict) to reflect a shock event."""
    r = dict(row)
    if event == "industrial_discharge":
        r["BOD"] *= 1.3
        r["COD"] = r["COD"] * 1.3 + 900.0       # 900 mg/L of COD bacteria cannot eat
        r["TSS"] *= 1.2
        r["pH"] = 5.2
    elif event == "acid_dump":
        r["pH"] = 3.2
    elif event == "alkaline_dump":
        r["pH"] = 10.8
    elif event == "storm_inflow":
        r["flow"] *= 2.6
        first_flush = 1.5 if hour_into_event < 2 else 1.0   # first rain washes sewer grit
        for p in ("BOD", "COD", "NH4"):
            r[p] /= 2.3
        r["TSS"] = r["TSS"] / 2.3 * first_flush
        r["FC"] /= 1.5
        r["temperature"] -= 2.0
    elif event == "septage_dumping":
        r.update(BOD=1100.0, COD=2400.0, TSS=1800.0, NH4=140.0, FC=5e8)
    elif event == "high_strength_food":
        r["BOD"] *= 1.7
        r["COD"] *= 1.8
        r["NH4"] *= 1.1
    else:
        raise ValueError(f"Unknown event: {event}")
    r["event"] = event
    return r


def generate_influent(days=30, seed=7, start="2026-07-01", events=None):
    """Build an hourly influent dataset. events=None gives normal sewage only."""
    rng = np.random.default_rng(seed)
    hours = days * 24
    times = pd.date_range(start, periods=hours, freq="h")

    # Slowly wandering random factors (AR(1) noise) so the data is not too regular
    def wander(sigma, persistence=0.85):
        x = np.zeros(hours)
        for i in range(1, hours):
            x[i] = persistence * x[i - 1] + rng.normal(0, sigma)
        return np.exp(x)

    flow_noise = wander(0.05)
    strength_noise = wander(0.06)
    nh4_noise = wander(0.05)
    tss_noise = wander(0.08)

    rows = []
    for i, t in enumerate(times):
        h = t.hour
        weekend = 1.06 if t.dayofweek == 6 else 1.0
        flow = DESIGN_FLOW * DIURNAL_FLOW[h] * weekend * flow_noise[i]
        # sewage is a bit stronger at the morning and evening peaks
        strength = (0.85 + 0.15 * DIURNAL_FLOW[h]) * strength_noise[i]
        bod = RAW_SEWAGE["BOD"] * strength
        rows.append({
            "timestamp": t,
            "flow": flow,
            "pH": RAW_SEWAGE["pH"] + 0.25 * math.sin(2 * math.pi * (h - 9) / 24) + rng.normal(0, 0.08),
            "temperature": RAW_SEWAGE["temperature"] + 1.5 * math.sin(2 * math.pi * (h - 9) / 24) + rng.normal(0, 0.2),
            "BOD": bod,
            "COD": bod * RAW_SEWAGE["COD_BOD_ratio"] * rng.normal(1.0, 0.03),
            "TSS": RAW_SEWAGE["TSS"] * strength * tss_noise[i],
            "NH4": RAW_SEWAGE["NH4"] * (0.85 + 0.15 * DIURNAL_FLOW[h]) * nh4_noise[i],
            "FC": RAW_SEWAGE["FC"] * math.exp(rng.normal(0, 0.4)),
            "event": "normal",
        })

    for day, start_hour, duration, event in (events or []):
        first = (day - 1) * 24 + start_hour
        for k in range(duration):
            idx = first + k
            if 0 <= idx < hours:
                rows[idx] = apply_event(rows[idx], event, k)

    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ["flow", "pH", "temperature", "BOD", "COD", "TSS", "NH4"]:
        df[c] = df[c].round(2)
    df["FC"] = df["FC"].round(-3)
    return df


def demo_dataset():
    """One monsoon month with nine labelled shock events."""
    return generate_influent(days=30, seed=7, start="2026-07-01", events=DEMO_EVENTS)


def training_dataset():
    """A separate, NORMAL-only month (different random seed) that the ML models
    learn from. The models never see the demo month or its events."""
    return generate_influent(days=30, seed=101, start="2026-06-01", events=None)


def load_influent_csv(source):
    """Read a CSV (path, file object or text). Required columns:
    flow, pH, BOD, COD, TSS, NH4. Optional: timestamp, temperature, FC, event."""
    if isinstance(source, str) and "\n" in source:
        source = io.StringIO(source)
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing column(s): {', '.join(missing)}. "
                         f"Required: {', '.join(REQUIRED)}")
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="h")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "temperature" not in df.columns:
        df["temperature"] = RAW_SEWAGE["temperature"]
    if "FC" not in df.columns:
        df["FC"] = RAW_SEWAGE["FC"]
    if "event" not in df.columns:
        df["event"] = "unknown"
    df["event"] = df["event"].fillna("normal").astype(str)
    df = df[COLUMNS].dropna(subset=REQUIRED).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("CSV has no usable rows.")
    return df


class InfluentFeed:
    """Hands out one hour of influent at a time. Also lets the dashboard
    inject a shock event into the coming hours (for live demos)."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True).copy()
        self.injected = set()

    def __len__(self):
        return len(self.df)

    def row(self, i):
        return self.df.iloc[i % len(self.df)]

    def water(self, i):
        r = self.row(i)
        return Water(flow=float(r["flow"]), pH=float(r["pH"]), temperature=float(r["temperature"]),
                     BOD=float(r["BOD"]), COD=float(r["COD"]), TSS=float(r["TSS"]),
                     NH4=float(r["NH4"]), NO3=0.0, FC=float(r["FC"]))

    def event(self, i):
        return str(self.row(i)["event"])

    def timestamp(self, i):
        return pd.Timestamp(self.row(i)["timestamp"])

    def inject(self, start, event, hours):
        """Overwrite rows start .. start+hours-1 with a shock event."""
        for k in range(hours):
            idx = (start + k) % len(self.df)
            base = self.df.loc[idx].to_dict()
            if base["event"] != "normal":
                continue  # do not stack two events on one hour
            new = apply_event(base, event, k)
            for col in ["flow", "pH", "temperature", "BOD", "COD", "TSS", "NH4", "FC"]:
                self.df.at[idx, col] = round(float(new[col]), 2)
            self.df.at[idx, "event"] = event
            self.injected.add(idx)
