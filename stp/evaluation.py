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


def compare_modes(data=None, norms="ngt2019", train_data=None, train_key="simulated", rule_limits=None):
    rows, models = [], {}
    for mode in MODES:
        m = STPModel(mode=mode, norms=norms, data=data, train_data=train_data, train_key=train_key,
                     rule_limits=rule_limits).run()
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


def full_evaluation(data=None, norms="ngt2019", train_data=None, train_key="simulated", rule_limits=None):
    modes, models = compare_modes(data, norms, train_data, train_key, rule_limits)
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


# ---------------------------------------------------------------------------
# Many random months (is the result a lucky month?) and unseen shock types
# ---------------------------------------------------------------------------
def _month_rows(data, label, norms):
    rows = []
    for mode in MODES:
        m = STPModel(mode=mode, norms=norms, data=data).run()
        s = m.summary()
        det = s["detection"]["used_by_plant"]
        rows.append({"month": label, "mode": mode, "hours_meeting_norms_pct": s["compliant_hours_pct"],
                     "recall": det["recall"], "precision": det["precision"],
                     "false_alarms": det["false_alarms"], "energy_kwh": s["energy_kwh"]})
    return rows


def _mean_sd(df):
    g = df.groupby("mode", sort=False)
    out = []
    for mode, x in g:
        out.append({"mode": mode,
                    "hours_meeting_norms_pct": f"{x['hours_meeting_norms_pct'].mean():.1f} ± {x['hours_meeting_norms_pct'].std():.1f}",
                    "worst_month_pct": round(float(x["hours_meeting_norms_pct"].min()), 1),
                    "detector_recall": f"{x['recall'].mean():.2f} ± {x['recall'].std():.2f}",
                    "false_alarms_per_month": f"{x['false_alarms'].mean():.1f}",
                    "energy_kwh": f"{x['energy_kwh'].mean():,.0f}"})
    return out


def _paired(df, a="ai", b="rules"):
    p = df.pivot(index="month", columns="mode", values="hours_meeting_norms_pct")
    d = p[a] - p[b]
    return {"compare": f"{a} minus {b}", "mean_difference_pct_points": round(float(d.mean()), 2),
            "sd": round(float(d.std()), 2), "months_better": int((d > 0.05).sum()),
            "months_equal": int((d.abs() <= 0.05).sum()), "months_worse": int((d < -0.05).sum()),
            "months": int(len(d))}


def multi_month(n=10, norms="ngt2019", log=print):
    """n months, each with new random sewage AND a new random schedule of the six
    known shock types. Mean ± standard deviation over months."""
    from .influent import random_month
    rows = []
    for k in range(n):
        log(f"  random month {k + 1}/{n}")
        rows += _month_rows(random_month(seed=k + 1), f"m{k + 1}", norms)
    df = pd.DataFrame(rows)
    return {"summary": _mean_sd(df), "ai_vs_rules": _paired(df, "ai", "rules"),
            "ai_vs_operator": _paired(df, "ai", "operator"), "per_month": df.to_dict(orient="records")}


def unseen_events(n=5, norms="ngt2019", log=print):
    """Months containing ONLY shock types never used for tuning (influent.UNSEEN_EVENT_TYPES)."""
    from .influent import UNSEEN_DURATION, UNSEEN_EVENT_TYPES, random_month
    rows, caught = [], []
    for k in range(n):
        log(f"  unseen-shock month {k + 1}/{n}")
        data = random_month(seed=100 + k, n_events=9, event_types=list(UNSEEN_EVENT_TYPES),
                            durations=UNSEEN_DURATION)
        rows += _month_rows(data, f"u{k + 1}", norms)
        table, _ = detection_by_event(STPModel(mode="ai", norms=norms, data=data).run())
        caught.append(table)
    df = pd.DataFrame(rows)
    by_event = pd.concat(caught).groupby("true_event").mean().round(0).reset_index()
    return {"summary": _mean_sd(df), "ai_vs_rules": _paired(df, "ai", "rules"),
            "by_event": by_event.to_dict(orient="records")}


def divert_policy_test(n_known=8, n_unseen=4, norms="ngt2019", log=print):
    """Should the plant divert water on an alarm with no identified cause
    ('unusual_load', usually ML-only)? Run AI mode both ways on the same months."""
    from . import agents
    from .influent import UNSEEN_DURATION, UNSEEN_EVENT_TYPES, random_month
    known = [random_month(seed=k + 1) for k in range(n_known)]
    unseen = [random_month(seed=100 + k, event_types=list(UNSEEN_EVENT_TYPES), durations=UNSEEN_DURATION)
              for k in range(n_unseen)]
    base = set(agents.DIVERT_TYPES)
    rows = []
    try:
        for label, types in [("divert unusual_load too (old)", base | {"unusual_load"}),
                             ("report only, do not divert (used)", base)]:
            log(f"  policy: {label}")
            agents.DIVERT_TYPES = types
            k = [STPModel(mode="ai", norms=norms, data=d).run().summary()["compliant_hours_pct"] for d in known]
            u = [STPModel(mode="ai", norms=norms, data=d).run().summary()["compliant_hours_pct"] for d in unseen]
            rows.append({"policy": label, "known_shock_months_pct": round(float(np.mean(k)), 2),
                         "worst_known_month": min(k), "unseen_shock_months_pct": round(float(np.mean(u)), 2),
                         "worst_unseen_month": min(u)})
    finally:
        agents.DIVERT_TYPES = base
    return rows


def operator_delay_sensitivity(delays=(0, 1, 2, 3, 6), norms="ngt2019"):
    """How much of the AI's advantage is just SPEED of response?"""
    from . import agents
    saved = agents.OPERATOR_DELAY_HOURS
    rows = []
    try:
        for d in delays:
            agents.OPERATOR_DELAY_HOURS = d
            m = STPModel(mode="operator", norms=norms).run()
            s = m.summary()
            health = m.datacollector.get_model_vars_dataframe()["bacteria_health"].mean()
            rows.append({"operator_response_delay_h": d, "hours_meeting_norms_pct": s["compliant_hours_pct"],
                         "TN_violation_hours": s["violation_hours_by_parameter"].get("TN", 0),
                         "mean_bacteria_health": round(float(health), 3)})
    finally:
        agents.OPERATOR_DELAY_HOURS = saved
    return rows
