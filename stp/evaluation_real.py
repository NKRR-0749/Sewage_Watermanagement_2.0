"""
evaluation_real.py - How the plant and its AI behave on REAL measured sewage
(the UCI Water Treatment Plant dataset, see real_data.py).

Everything is measured on the TEST days (after the first real_data.TRAIN_DAYS).
The training days are used for two kinds of "site calibration":
    - the ML models (Isolation Forest + forecaster) learn this plant's normal sewage
    - the detection-rule limits are raised to what is normal at this plant
      (ml.calibrate_rule_limits)

Questions answered:
1. How do the four control modes do on real influent?
2. Does site calibration matter? (calibrated vs default Indian rules vs ML
   trained on simulated data)
3. How often does each rule fire on real data, before and after calibration?
4. How good are the detectors? The real data has no shock labels, so labelled
   shocks are injected on top of the real data and the detectors are graded.
5. How does our simulated effluent compare with what the REAL plant discharged
   on the same days?
6. Results per real plant state, and the forecast vs "same as last hour".

Used by scripts/evaluate_real.py (writes results/EVALUATION_REAL.md).
"""

import pandas as pd

from . import real_data
from .config import DISCHARGE_NORMS
from .evaluation import detection_by_event, forecast_accuracy
from .ml import DEFAULT_RULE_LIMITS, anomaly_features, calibrate_rule_limits, make_rules
from .model import STPModel, MODES

REAL_KEY = "uci"


def site_setup():
    """Training days, the calibrated rule limits and the STPModel keyword arguments."""
    train = real_data.uci_train()
    limits = calibrate_rule_limits(train)
    return train, limits, dict(train_data=train, train_key=REAL_KEY, rule_limits=limits)


def _summary_row(label, m):
    s = m.summary()
    worst = s["violation_hours_by_parameter"]
    return {
        "setup": label,
        "hours_meeting_norms_pct": s["compliant_hours_pct"],
        "volume_meeting_norms_pct": s["compliant_volume_pct"],
        "violation_hours": m.stats["hours"] - m.stats["compliant_hours"],
        "worst_parameter": max(worst, key=worst.get) if worst else "-",
        "alarms": s["alarms"],
        "alarm_hours_pct": round(100 * s["alarms"] / max(s["hours"], 1), 1),
        "energy_kwh": s["energy_kwh"],
        "diverted_m3": s["diverted_m3"],
        "overflow_m3": s["overflow_m3"],
    }


def _hourly(model):
    df = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
    feed = model.feed.df.iloc[:len(df)]
    df["day"] = feed["day"].to_numpy()
    df["plant_state"] = feed["plant_state"].to_numpy()
    df["alarm"] = df["alarm"].astype(bool)
    return df


def by_plant_state(model):
    """Compliance and alarm rate for each plant-state label in the real data."""
    g = _hourly(model).groupby("plant_state").agg(
        hours=("compliant", "size"), meets_norms_pct=("compliant", "mean"),
        alarm_hours_pct=("alarm", "mean"), mean_influent_COD=("influent_COD", "mean"),
        mean_effluent_COD=("effluent_COD", "mean"))
    g["meets_norms_pct"] = (g["meets_norms_pct"] * 100).round(1)
    g["alarm_hours_pct"] = (g["alarm_hours_pct"] * 100).round(1)
    return g.round(1).reset_index().to_dict(orient="records")


def rule_breakdown(data, limits):
    """Hours each rule fires on the real data, with default and calibrated limits."""
    f = anomaly_features(data)
    rows = []
    for (name, rule), (_, cal_rule), key in zip(make_rules().items(), make_rules(limits).items(),
                                                DEFAULT_RULE_LIMITS):
        rows.append({"rule": name, "default_limit": DEFAULT_RULE_LIMITS[key], "calibrated_limit": limits[key],
                     "hours_default": int(rule(f).sum()), "hours_calibrated": int(cal_rule(f).sum())})
    return rows


def vs_real_plant(model, norms="ngt2019"):
    """Our simulated effluent vs the REAL plant's measured effluent, day by day.
    Only days where the real plant has a measurement are compared."""
    daily = real_data.load_uci_daily().set_index("day")
    ours = _hourly(model).groupby("day")[["effluent_BOD", "effluent_COD", "effluent_TSS"]].mean()
    lim = DISCHARGE_NORMS[norms]["limits"]
    rows = []
    for p in ["BOD", "COD", "TSS"]:
        real_eff = daily.loc[ours.index, f"real_eff_{p}"]
        real_in = daily.loc[ours.index, p]
        ok = real_eff.notna()
        ours_p = ours.loc[ok, f"effluent_{p}"]
        hi = lim.get(p, (None, None))[1]
        rows.append({
            "parameter": p,
            "days_compared": int(ok.sum()),
            "real_influent_mean": round(float(real_in[ok].mean()), 1),
            "REAL_plant_effluent_mean": round(float(real_eff[ok].mean()), 1),
            "our_effluent_mean": round(float(ours_p.mean()), 1),
            "REAL_plant_removal_pct": round(100 * (1 - real_eff[ok].mean() / real_in[ok].mean()), 1),
            "our_removal_pct": round(100 * (1 - ours_p.mean() / real_in[ok].mean()), 1),
            "limit": hi if hi is not None else "-",
            "REAL_plant_days_within_limit_pct": round(100 * float((real_eff[ok] <= hi).mean()), 1) if hi else "-",
            "our_days_within_limit_pct": round(100 * float((ours_p <= hi).mean()), 1) if hi else "-",
        })
    return rows


def full_real_evaluation(norms="ngt2019", days=None, log=print):
    """days=None uses all test days (about 3 minutes on a laptop)."""
    train, limits, site = site_setup()
    test = real_data.uci_test(days)
    shocked = real_data.uci_test_with_shocks(days)

    log(f"1/3  Real test data ({len(test)} hours) in all four modes ...")
    models = {mode: STPModel(mode=mode, norms=norms, data=test, **site).run() for mode in MODES}

    log("2/3  Without site calibration ...")
    ai_default_rules = STPModel(mode="ai", norms=norms, data=test, train_data=train, train_key=REAL_KEY).run()
    ai_uncalibrated = STPModel(mode="ai", norms=norms, data=test).run()
    calibration = [
        _summary_row("AI, fully site-calibrated (ML + rule limits)", models["ai"]),
        _summary_row("AI, ML retrained, default Indian rule limits", ai_default_rules),
        _summary_row("AI, no calibration (ML from simulated data, default rules)", ai_uncalibrated),
    ]

    log("3/3  Real data + injected labelled shocks (grading the detectors) ...")
    shock_cal = STPModel(mode="ai", norms=norms, data=shocked, **site).run()
    shock_uncal = STPModel(mode="ai", norms=norms, data=shocked).run()
    det_table, false_alarms = detection_by_event(shock_cal)
    _, false_alarms_uncal = detection_by_event(shock_uncal)

    return {
        "data": real_data.describe(),
        "test_hours": len(test),
        "rule_limits": limits,
        "modes": [_summary_row(mode, m) for mode, m in models.items()],
        "calibration": calibration,
        "rules": rule_breakdown(test, limits),
        "shock_events": int((shocked["event"] != "normal").sum()),
        "detection": det_table.to_dict(orient="records"),
        "false_alarms": false_alarms,
        "detector_scores": shock_cal.summary()["detection"],
        "detector_scores_uncalibrated": shock_uncal.summary()["detection"],
        "false_alarms_uncalibrated": false_alarms_uncal,
        "vs_real_plant": vs_real_plant(models["ai"], norms),
        "by_plant_state": by_plant_state(models["ai"]),
        "forecast": forecast_accuracy(models["ai"], models["rules"]).to_dict(orient="records"),
    }
