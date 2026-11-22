"""
real_data.py - Run the plant on REAL measured sewage data from a public dataset.

Dataset: "Water Treatment Plant", UCI Machine Learning Repository
    Poch, M. (1993). Water Treatment Plant [Dataset]. UCI Machine Learning
    Repository. https://doi.org/10.24432/C5FS4C   (licence: CC BY 4.0)
    https://archive.ics.uci.edu/dataset/106/water+treatment+plant

What it is: daily sensor measurements from a real urban sewage treatment plant
in Manresa (near Barcelona, Spain), about 35,000 m³/day, mostly domestic sewage
with some industrial discharge. 527 days in the original; the copy kept in
data/real/ has 520 days plus the plant-state label of each day.

We use only the INFLUENT columns (what arrives at the plant), because our own
agents do the treatment:

    UCI column   meaning                         our column
    Q-E          input flow (m³/day)             flow (converted to m³/hour)
    PH-E         input pH                        pH
    DBO-E        input BOD (mg/L)                BOD
    DQO-E        input COD (mg/L)                COD
    SS-E         input suspended solids (mg/L)   TSS

What we have to add or change, and why (say this in the viva):

1. DAILY -> HOURLY. The plant is simulated hour by hour, but the data has one
   reading per day. Each day's value is placed at noon and values in between
   are linearly interpolated. Flow is then shaped by a standard daily pattern
   (night low, morning peak), as in influent.DIURNAL_FLOW. So the day-to-day
   changes are REAL; the hour-to-hour shape inside a day is ASSUMED.
2. SCALING. The Manresa plant is ~3.7x bigger than our 10 MLD plant, so flow is
   scaled so that the median day equals our design flow. Concentrations (mg/L)
   are NOT scaled: they are used exactly as measured.
3. AMMONIA (NH4) is not in the dataset. It is estimated from COD using the
   typical domestic NH4/COD ratio (config.RAW_SEWAGE). It follows the real COD
   trend but is not a measurement.
4. TEMPERATURE and FAECAL COLIFORMS are not in the dataset; typical constant
   values from config.RAW_SEWAGE are used.
5. MISSING VALUES (a few days have no BOD or flow reading) are filled by linear
   interpolation between the neighbouring days. Counts are reported.
6. No dates are kept in this copy, so the timestamps are placeholders starting
   1 Jan 2000; the ORDER of the days is the original order.
7. The dataset also records what the REAL plant discharged (output BOD, COD,
   SS, pH). Those columns are never fed into our plant; they are only used to
   compare our simulated effluent with real measured effluent.
8. The data has NO labels for shock events, so detection precision/recall
   cannot be measured on it directly. evaluation_real.py therefore also injects
   labelled shocks on top of the real data to grade the detectors.
"""

import os

import numpy as np
import pandas as pd

from .config import DESIGN_FLOW, RAW_SEWAGE
from .influent import COLUMNS, DIURNAL_FLOW, EVENT_TYPES, apply_event

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(os.path.dirname(HERE), "data", "real", "uci_water_treatment.data")

SOURCE = ("UCI Water Treatment Plant dataset (Poch, 1993; doi:10.24432/C5FS4C), "
          "daily influent of a real municipal plant in Manresa, Spain")

# The 38 attribute names in the official UCI file (it has no header row)
UCI_ATTRIBUTES = [
    "Q-E", "ZN-E", "PH-E", "DBO-E", "DQO-E", "SS-E", "SSV-E", "SED-E", "COND-E",
    "PH-P", "DBO-P", "SS-P", "SSV-P", "SED-P", "COND-P",
    "PH-D", "DBO-D", "DQO-D", "SS-D", "SSV-D", "SED-D", "COND-D",
    "PH-S", "DBO-S", "DQO-S", "SS-S", "SSV-S", "SED-S", "COND-S",
    "RD-DBO-P", "RD-SS-P", "RD-SED-P", "RD-DBO-S", "RD-DQO-S",
    "RD-DBO-G", "RD-DQO-G", "RD-SS-G", "RD-SED-G",
]
INFLUENT_MAP = {"Q-E": "flow_m3_day", "PH-E": "pH", "DBO-E": "BOD", "DQO-E": "COD", "SS-E": "TSS"}
# The real plant's own measured OUTPUT, kept only to compare with our simulated effluent
EFFLUENT_MAP = {"PH-S": "real_eff_pH", "DBO-S": "real_eff_BOD", "DQO-S": "real_eff_COD", "SS-S": "real_eff_TSS"}

# Ammonia estimate: typical domestic NH4 / COD ratio (25 / (250 * 1.9) ~ 0.053)
NH4_PER_COD = RAW_SEWAGE["NH4"] / (RAW_SEWAGE["BOD"] * RAW_SEWAGE["COD_BOD_ratio"])

# First part of the record is used to TRAIN the ML models, the rest to TEST them.
TRAIN_DAYS = 150


# ---------------------------------------------------------------------------
# 1. Read the file (works with the official UCI file and with the labelled copy)
# ---------------------------------------------------------------------------
def load_uci_daily(path=DEFAULT_PATH):
    """Return one row per day: day, plant_state, flow_m3_day, pH, BOD, COD, TSS,
    plus a 'filled' column listing which values were missing and interpolated."""
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    if first.lower().startswith("class"):
        # Labelled copy: header row, first column = plant state
        raw = pd.read_csv(path)
        raw.columns = [c.strip() for c in raw.columns]
        raw = raw.dropna(how="all").rename(columns={"class": "plant_state"})
    else:
        # Official UCI file: no header, first column = date like "D-1/3/90", '?' = missing
        raw = pd.read_csv(path, header=None, names=["date"] + UCI_ATTRIBUTES, na_values="?")
        raw["plant_state"] = "unlabelled"
    raw = raw.reset_index(drop=True)

    daily = raw[list(INFLUENT_MAP)].rename(columns=INFLUENT_MAP).apply(pd.to_numeric, errors="coerce")
    missing = daily.isna()
    daily = daily.interpolate(limit_direction="both")
    daily.insert(0, "day", np.arange(1, len(daily) + 1))
    daily.insert(1, "plant_state", raw["plant_state"].astype(str).str.strip())
    daily["filled"] = missing.apply(lambda r: ",".join(c for c in missing.columns if r[c]), axis=1)
    # real measured effluent: NOT filled (only days with a real measurement are compared)
    eff = raw[list(EFFLUENT_MAP)].rename(columns=EFFLUENT_MAP).apply(pd.to_numeric, errors="coerce")
    daily = pd.concat([daily, eff], axis=1)
    if "date" in raw.columns:
        daily["date"] = raw["date"]
    return daily


def missing_counts(daily):
    counts = {}
    for cell in daily["filled"]:
        for c in filter(None, cell.split(",")):
            counts[c] = counts.get(c, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# 2. Daily -> hourly influent in the plant's own format
# ---------------------------------------------------------------------------
def daily_to_hourly(daily, label="unknown", start="2000-01-01"):
    """Turn daily readings into hourly influent rows (influent.COLUMNS + day, plant_state)."""
    n = len(daily)
    hours = n * 24
    t = np.arange(hours)
    day_centres = np.arange(n) * 24 + 12          # each day's reading sits at noon

    def interp(col):
        return np.interp(t, day_centres, daily[col].to_numpy(dtype=float))

    scale = (DESIGN_FLOW * 24) / float(daily["flow_m3_day"].median())
    hour_of_day = t % 24
    shape = DIURNAL_FLOW[hour_of_day]
    strength = 0.85 + 0.15 * shape                # same daily strength pattern as influent.py

    cod = interp("COD") * strength
    df = pd.DataFrame({
        "timestamp": pd.date_range(start, periods=hours, freq="h"),
        "flow": interp("flow_m3_day") * scale / 24 * shape,
        "pH": interp("pH"),
        "temperature": RAW_SEWAGE["temperature"],
        "BOD": interp("BOD") * strength,
        "COD": cod,
        "TSS": interp("TSS") * strength,
        "NH4": cod * NH4_PER_COD,
        "FC": RAW_SEWAGE["FC"],
        "event": label,
    })
    for c in ["flow", "pH", "BOD", "COD", "TSS", "NH4"]:
        df[c] = df[c].round(2)
    df["day"] = np.repeat(daily["day"].to_numpy(), 24)
    df["plant_state"] = np.repeat(daily["plant_state"].to_numpy(), 24)
    df.attrs["flow_scale"] = scale
    return df[COLUMNS + ["day", "plant_state"]]


# ---------------------------------------------------------------------------
# 3. Ready-made datasets
# ---------------------------------------------------------------------------
_CACHE = {}


def uci_hourly(path=DEFAULT_PATH):
    """All 520 days as hourly influent (cached)."""
    if path not in _CACHE:
        _CACHE[path] = daily_to_hourly(load_uci_daily(path))
    return _CACHE[path].copy()


def uci_train(path=DEFAULT_PATH):
    """First TRAIN_DAYS days: what the ML models learn 'normal' from."""
    df = uci_hourly(path)
    df = df[df["day"] <= TRAIN_DAYS].reset_index(drop=True)
    df["event"] = "normal"
    return df


def uci_test(days=None, path=DEFAULT_PATH):
    """The days after the training period (never seen by the ML models)."""
    df = uci_hourly(path)
    df = df[df["day"] > TRAIN_DAYS].reset_index(drop=True)
    if days is not None:
        df = df[df["day"] <= TRAIN_DAYS + days].reset_index(drop=True)
    return df


# Shocks injected on top of the real test data so detection can be graded.
# Cycle through every event type; one event roughly every 4 days.
SHOCK_CYCLE = list(EVENT_TYPES)
SHOCK_DURATION = {"industrial_discharge": 6, "acid_dump": 3, "alkaline_dump": 4, "storm_inflow": 10,
                  "septage_dumping": 4, "high_strength_food": 5}


def uci_test_with_shocks(days=None, every_hours=96, seed=3, path=DEFAULT_PATH):
    """Real test data + labelled shock events. Real hours are labelled 'normal'.
    (If the real data itself contains an unusual day, an alarm on it counts as a
    false alarm, so the scores are, if anything, pessimistic.)"""
    rng = np.random.default_rng(seed)
    df = uci_test(days, path)
    df["event"] = "normal"
    rows = df.to_dict("records")
    start, k = 24, 0
    while start < len(rows) - 24:
        event = SHOCK_CYCLE[k % len(SHOCK_CYCLE)]
        begin = start + int(rng.integers(0, 24))
        for j in range(SHOCK_DURATION[event]):
            if begin + j < len(rows):
                rows[begin + j] = apply_event(rows[begin + j], event, j)
        start += every_hours
        k += 1
    out = pd.DataFrame(rows)
    for c in ["flow", "pH", "temperature", "BOD", "COD", "TSS", "NH4"]:
        out[c] = out[c].astype(float).round(2)
    return out[COLUMNS + ["day", "plant_state"]]


def describe(path=DEFAULT_PATH):
    """Short facts about the real data, for reports and the dashboard."""
    daily = load_uci_daily(path)
    return {
        "source": SOURCE,
        "days": int(len(daily)),
        "train_days": TRAIN_DAYS,
        "test_days": int(len(daily) - TRAIN_DAYS),
        "flow_scale": round((DESIGN_FLOW * 24) / float(daily["flow_m3_day"].median()), 3),
        "median_flow_m3_day_original": round(float(daily["flow_m3_day"].median())),
        "missing_values_filled": missing_counts(daily),
        "plant_states": daily["plant_state"].value_counts().to_dict(),
        "means": {c: round(float(daily[c].mean()), 1) for c in ["pH", "BOD", "COD", "TSS"]},
        "ranges": {c: [round(float(daily[c].min()), 1), round(float(daily[c].max()), 1)]
                   for c in ["flow_m3_day", "pH", "BOD", "COD", "TSS"]},
    }
