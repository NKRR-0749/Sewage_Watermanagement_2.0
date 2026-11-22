"""
model.py - The Mesa model: the whole sewage treatment plant.

STPModel creates the eleven agents (agents.py), the message bus
(messages.py) and the data feed (influent.py). Each call to step()
simulates ONE HOUR: every agent acts once, in the order water flows.

    model = STPModel(mode="ai", norms="ngt2019")
    for _ in range(24):
        model.step()
    print(model.summary())

The same file runs on a laptop (python scripts/run_simulation.py) and in
the browser dashboard (through Pyodide).
"""

import json

import mesa

from .agents import (InfluentFeedAgent, MonitoringAgent, ForecastAgent, PreliminaryAgent,
                     EqualizationAgent, PrimaryAgent, AerationAgent, SecondaryAgent,
                     FilterAgent, DisinfectionAgent, ComplianceAgent)
from .config import DISCHARGE_NORMS, DEFAULT_NORMS
from .influent import InfluentFeed, demo_dataset
from .messages import MessageBus
from .ml import get_models

MODES = {
    "ai": "Autonomous - AI (ML detection + ML forecasting)",
    "rules": "Autonomous - rules only (no ML)",
    "operator": "Well-run conventional plant (DO control + human operator, 3 h delay)",
    "fixed": "Conventional plant (fixed settings, no automation)",
}

# Order in which the agents act every hour = order the water flows
AGENT_ORDER = [InfluentFeedAgent, ForecastAgent, MonitoringAgent, PreliminaryAgent,
               EqualizationAgent, PrimaryAgent, AerationAgent, SecondaryAgent,
               FilterAgent, DisinfectionAgent, ComplianceAgent]

STREAM_ORDER = ["raw", "preliminary", "equalization", "primary", "aeration",
                "secondary", "filter", "chlorination", "effluent"]


class STPModel(mesa.Model):
    def __init__(self, mode="ai", norms=DEFAULT_NORMS, data=None, seed=1, train_data=None, train_key="simulated",
                 rule_limits=None):
        """data: hourly influent (default: simulated demo month).
        train_data / train_key: what the ML models learn 'normal' from
        (default: the simulated normal month). For real data use
        real_data.uci_train() with train_key="uci".
        rule_limits: detection-rule limits (default: ml.DEFAULT_RULE_LIMITS);
        use ml.calibrate_rule_limits(train_data) to tune them to a site."""
        super().__init__(rng=seed)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if norms not in DISCHARGE_NORMS:
            raise ValueError(f"norms must be one of {list(DISCHARGE_NORMS)}")
        self.mode = mode
        self.norms = norms
        self.feed = InfluentFeed(data if data is not None else demo_dataset())
        self.bus = MessageBus()
        self.detector, self.forecaster = get_models(train_data, train_key)
        self.rule_limits = rule_limits
        # score every row of the feed in one go (fast); injected rows are re-scored live
        self.precomputed_scores = dict(enumerate(self.detector.scores(self.feed.df)))

        self.hour = 0
        self.history = []
        self.streams = {}
        self.unit_info = {}
        self.detection = {}
        self.detections = []
        self.forecast = None
        self.forecast_errors = []
        self.compliance = {}
        self.raw = self.raw_event = self.timestamp = None
        self.effluent = self.last_effluent = None
        self.recirc_water = self.backwash_water = None
        self.aeration_health = 1.0
        self.alarm_streak = 0
        self.stats = {"hours": 0, "compliant_hours": 0, "discharged_m3": 0.0, "compliant_m3": 0.0,
                      "energy_kwh": 0.0, "chlorine_kg": 0.0, "ph_chemical_kg": 0.0,
                      "diverted_m3": 0.0, "recirculated_m3": 0.0, "overflow_m3": 0.0,
                      "alarms": 0, "violations": {}}

        self.plant_agents = [cls(self) for cls in AGENT_ORDER]
        self.agent_by_label = {a.label: a for a in self.plant_agents}

        self.datacollector = mesa.DataCollector(model_reporters={
            "time": lambda m: str(m.timestamp),
            "event": lambda m: m.raw_event,
            "alarm": lambda m: m.detection.get("alarm"),
            "influent_flow": lambda m: m.raw.flow,
            "influent_BOD": lambda m: m.raw.BOD,
            "influent_COD": lambda m: m.raw.COD,
            "influent_pH": lambda m: m.raw.pH,
            "effluent_BOD": lambda m: m.effluent.BOD,
            "effluent_COD": lambda m: m.effluent.COD,
            "effluent_TSS": lambda m: m.effluent.TSS,
            "effluent_TN": lambda m: m.effluent.TN,
            "effluent_FC": lambda m: m.effluent.FC,
            "effluent_pH": lambda m: m.effluent.pH,
            "compliant": lambda m: m.compliance["passed"],
            "failing": lambda m: ",".join(m.compliance["failing"]),
            "DO": lambda m: m.unit_info["aeration"]["DO"],
            "bacteria_health": lambda m: m.aeration_health,
            "blower_pct": lambda m: m.unit_info["aeration"]["blower_pct"],
            "eq_level_pct": lambda m: m.unit_info["equalization"]["level_pct"],
            "holding_pct": lambda m: m.unit_info["equalization"]["holding_pct"],
        })

    # ------------------------------------------------------------------
    def step(self):
        """Simulate one hour."""
        for agent in self.plant_agents:
            agent.step()
        self.streams["raw"] = self.raw
        self.streams["effluent"] = self.effluent
        self.datacollector.collect(self)
        self.hour += 1

    def run(self, hours=None):
        hours = len(self.feed) if hours is None else hours
        for _ in range(hours):
            self.step()
        return self

    @property
    def finished(self):
        return self.hour >= len(self.feed)

    def inject(self, event, hours=4):
        """Dashboard button: put a shock event into the next few hours of data."""
        self.feed.inject(self.hour, event, hours)

    # ------------------------------------------------------------------
    def summary(self):
        s = self.stats
        det = detector_scores(self.detections)
        return {
            "mode": self.mode, "norms": self.norms, "hours": s["hours"],
            "compliant_hours_pct": round(100 * s["compliant_hours"] / max(s["hours"], 1), 1),
            "compliant_volume_pct": round(100 * s["compliant_m3"] / max(s["discharged_m3"], 1e-9), 1),
            "violation_hours_by_parameter": dict(sorted(s["violations"].items())),
            "energy_kwh": round(s["energy_kwh"]), "chlorine_kg": round(s["chlorine_kg"]),
            "ph_chemical_kg": round(s["ph_chemical_kg"]),
            "diverted_m3": round(s["diverted_m3"]), "recirculated_m3": round(s["recirculated_m3"]),
            "overflow_m3": round(s["overflow_m3"]), "alarms": s["alarms"],
            "detection": det,
        }

    def snapshot(self):
        """Everything the dashboard needs for the hour just simulated."""
        streams = {}
        for name in STREAM_ORDER:
            w = self.streams.get(name)
            if w is not None:
                streams[name] = w.rounded()
        msgs = [m.to_dict() for m in self.bus.messages_at(self.hour - 1)]
        return {
            "hour": self.hour, "total_hours": len(self.feed), "time": str(self.timestamp),
            "event": self.raw_event, "mode": self.mode, "norms": self.norms,
            "streams": streams, "units": self.unit_info, "detection": self.detection,
            "forecast": self.forecast, "compliance": self.compliance, "messages": msgs,
            "summary": self.summary(),
        }

    def snapshot_json(self):
        return json.dumps(self.snapshot(), default=float)


def detector_scores(detections):
    """Precision / recall of each detector against the TRUE event labels in the
    data (hours labelled 'normal' vs any shock). Hours with unknown labels are skipped."""
    rows = [d for d in detections if d["true_event"] not in ("unknown",)]
    out = {}
    for key in ("rules_alarm", "ml_alarm", "alarm"):
        tp = sum(1 for d in rows if d[key] and d["true_event"] != "normal")
        fp = sum(1 for d in rows if d[key] and d["true_event"] == "normal")
        fn = sum(1 for d in rows if not d[key] and d["true_event"] != "normal")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[key.replace("_alarm", "") if key != "alarm" else "used_by_plant"] = {
            "true_positives": tp, "false_alarms": fp, "missed": fn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
    return out
