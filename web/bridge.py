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
from stp.influent import EVENT_TYPES, load_influent_csv

state = {"model": None, "data": None, "data_label": "Built-in demo month (simulated July influent)"}


def _dump(obj):
    return json.dumps(obj, default=float)


def options():
    return _dump({
        "modes": MODES,
        "norms": {k: {"label": v["label"], "limits": v["limits"]} for k, v in DISCHARGE_NORMS.items()},
        "events": EVENT_TYPES,
        "units": UNITS,
        "plant": {"design_flow": PLANT["design_flow"]},
    })


def use_demo():
    state["data"] = None
    state["data_label"] = "Built-in demo month (simulated July influent)"
    return _dump({"label": state["data_label"]})


def use_csv(text, name):
    df = load_influent_csv(text)
    state["data"] = df
    state["data_label"] = f"Uploaded: {name} ({len(df)} hours)"
    return _dump({"label": state["data_label"], "hours": len(df)})


def reset(mode, norms):
    state["model"] = STPModel(mode=mode, norms=norms, data=state["data"])
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
    return _dump(full_evaluation(data=state["data"], norms=norms))
