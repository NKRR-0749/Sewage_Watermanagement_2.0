"""
bridge.py - The only link between the web page (app.js) and the Python plant.

The page cannot call Mesa directly, so it calls these small functions
through Pyodide. Every function returns JSON text. No plant logic lives
here: it just creates the STPModel, steps it and reports what happened.
"""

import json

from stp import STPModel, MODES, DISCHARGE_NORMS
from stp.config import PLANT, UNITS
from stp.evaluation import full_evaluation
from stp.influent import EVENT_TYPES, UNSEEN_EVENT_TYPES, load_influent_csv
from stp import real_data
from stp.evaluation_real import site_setup

REAL_DAYS = 60   # days of the real test period shown in the browser (keeps it quick)

state = {"model": None, "data": None, "site": {}, "real": False,
         "data_label": "Built-in demo month (simulated July influent)"}


def _dump(obj):
    return json.dumps(obj, default=float)


def options():
    return _dump({
        "modes": MODES,
        "norms": {k: {"label": v["label"], "limits": v["limits"]} for k, v in DISCHARGE_NORMS.items()},
        "events": {**EVENT_TYPES, **UNSEEN_EVENT_TYPES},
        "units": UNITS,
        "plant": {"design_flow": PLANT["design_flow"]},
    })


def use_demo():
    state["data"] = None
    state["site"] = {}
    state["real"] = False
    state["data_label"] = "Built-in demo month (simulated July influent)"
    return _dump({"label": state["data_label"]})


def use_csv(text, name):
    df = load_influent_csv(text)
    state["data"] = df
    state["site"] = {}
    state["real"] = False
    state["data_label"] = f"Uploaded: {name} ({len(df)} hours)"
    return _dump({"label": state["data_label"], "hours": len(df)})


def use_real():
    """Real measured influent (UCI Water Treatment Plant, Manresa, Spain).
    ML models and rule limits are calibrated on the first real days."""
    _, limits, site = site_setup()
    state["data"] = real_data.uci_test(REAL_DAYS)
    state["site"] = site
    state["real"] = True
    info = real_data.describe()
    state["data_label"] = (f"REAL data: UCI Water Treatment Plant (Manresa, Spain), {REAL_DAYS} test days "
                           f"({REAL_DAYS * 24} h). ML and rule limits calibrated on the first "
                           f"{info['train_days']} real days. Daily readings spread to hours; NH4 estimated from COD.")
    return _dump({"label": state["data_label"], "hours": len(state["data"]), "rule_limits": limits})


def reset(mode, norms):
    state["model"] = STPModel(mode=mode, norms=norms, data=state["data"], **state["site"])
    return _dump({"total_hours": len(state["model"].feed), "data_label": state["data_label"]})


def _compact(m):
    e, r, a = m.effluent, m.raw, m.unit_info["aeration"]
    eq = m.unit_info["equalization"]
    return {
        "t": str(m.timestamp)[:16], "event": m.raw_event,
        "alarm": m.detection["alarm"], "type": m.detection["type"],
        "eff": {"pH": e.pH, "BOD": e.BOD, "COD": e.COD, "TSS": e.TSS, "TN": e.TN, "FC": e.FC},
        "inf": {"flow": r.flow, "pH": r.pH, "BOD": r.BOD, "COD": r.COD, "TSS": r.TSS, "NH4": r.NH4},
        "DO": a["DO"], "blower": a["blower_pct"], "health": a["health"],
        "eq": eq["level_pct"], "hold": eq["holding_pct"],
        "passed": m.compliance["passed"], "failing": m.compliance["failing"],
        "msgs": [msg.to_dict() for msg in m.bus.messages_at(m.hour - 1)],
    }


def advance(n):
    m = state["model"]
    hours = []
    for _ in range(int(n)):
        if m.finished:
            break
        m.step()
        hours.append(_compact(m))
    snap = m.snapshot() if m.hour > 0 else None
    return _dump({"hours": hours, "snapshot": snap, "finished": m.finished})


def inject(event, hours):
    m = state["model"]
    m.inject(event, int(hours))
    return _dump({"ok": True, "from_hour": m.hour})


def evaluate(norms):
    if state["real"]:
        # Real data has no shock labels: grade the detectors on real data + injected shocks
        data = real_data.uci_test_with_shocks(REAL_DAYS)
        out = full_evaluation(data=data, norms=norms, **state["site"])
        out["note"] = ("Real influent with labelled shocks injected about every 4 days, so the detectors "
                       "can be graded. Real hours count as normal.")
        return _dump(out)
    return _dump(full_evaluation(data=state["data"], norms=norms))
