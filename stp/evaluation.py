"""
evaluation.py - Honest measurement of how well the plant and its AI work.

Three questions, each answered against data the models never trained on:

1. Does autonomous control beat a conventional plant?
   Run the same month through "fixed", "rules" and "ai" modes and compare
   the share of hours that meet the norms, energy and chemicals.

2. How good are the anomaly detectors?
   Compare each detector's alarms with the TRUE event labels written into
   the data when it was generated (not with another detector's output).

3. Is the ML forecast better than assuming "same as last hour"?
   Mean absolute error of both, on the demo month.

Used by scripts/evaluate.py and by the dashboard's "Compare modes" button.
"""

import numpy as np
import pandas as pd

from .model import STPModel, MODES


def compare_modes(data=None, norms="ngt2019"):
    rows, models = [], {}
    for mode in MODES:
        m = STPModel(mode=mode, norms=norms, data=data).run()
        models[mode] = m
        s = m.summary()
        rows.append({
            "mode": mode,
            "description": MODES[mode],
            "hours_meeting_norms_pct": s["compliant_hours_pct"],
            "volume_meeting_norms_pct": s["compliant_volume_pct"],
            "violation_hours": m.stats["hours"] - m.stats["compliant_hours"],
            "worst_parameter": max(s["violation_hours_by_parameter"], key=s["violation_hours_by_parameter"].get)
            if s["violation_hours_by_parameter"] else "-",
            "energy_kwh": s["energy_kwh"],
            "chlorine_kg": s["chlorine_kg"],
            "diverted_m3": s["diverted_m3"],
            "overflow_m3": s["overflow_m3"],
        })
    return pd.DataFrame(rows), models


def detection_by_event(model):
    """Share of each event's hours that each detector flagged."""
    det = pd.DataFrame(model.detections)
    events = det[~det["true_event"].isin(["normal", "unknown"])]
    table = events.groupby("true_event")[["rules_alarm", "ml_alarm", "alarm"]].mean().mul(100).round(0)
    table.columns = ["rules_caught_pct", "ml_caught_pct", "ai_mode_caught_pct"]
    normal = det[det["true_event"] == "normal"]
    false_alarms = {"rules": int(normal["rules_alarm"].sum()), "ml": int(normal["ml_alarm"].sum()),
                    "ai_mode": int(normal["alarm"].sum()), "normal_hours": int(len(normal))}
    return table.reset_index(), false_alarms


def forecast_accuracy(ai_model, rules_model):
    ml = pd.DataFrame(ai_model.forecast_errors).mean()
    naive = pd.DataFrame(rules_model.forecast_errors).mean()
    out = pd.DataFrame({"ml_forecast_mae": ml.round(2), "last_hour_mae": naive.round(2)})
    out["improvement_pct"] = ((1 - out["ml_forecast_mae"] / out["last_hour_mae"]) * 100).round(1)
    return out.reset_index().rename(columns={"index": "quantity"})


def full_evaluation(data=None, norms="ngt2019"):
    modes, models = compare_modes(data, norms)
    det_table, false_alarms = detection_by_event(models["ai"])
    fc = forecast_accuracy(models["ai"], models["rules"])
    do_error = {mode: round(float(np.mean(np.abs(m.datacollector.get_model_vars_dataframe()["DO"] - 2.0))), 2)
                for mode, m in models.items()}
    return {
        "modes": modes.to_dict(orient="records"),
        "detection": det_table.to_dict(orient="records"),
        "false_alarms": false_alarms,
        "detector_scores": models["ai"].summary()["detection"],
        "forecast": fc.to_dict(orient="records"),
        "do_tracking_error": do_error,
    }
