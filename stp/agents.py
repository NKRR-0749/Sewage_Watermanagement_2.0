"""
agents.py - The BRAINS of the plant: eleven Mesa agents.

Each agent is a mesa.Agent with a step() method that runs once per simulated
hour, in the order water flows through the plant. Agents read their inbox,
look at the water reaching them, make a decision, and pass water and
messages on.

Data & intelligence agents
    InfluentFeedAgent    reads the next hour of influent data (replaces sensors)
    MonitoringAgent      detects shock loads (rules + Isolation Forest) and names them
    ForecastAgent        predicts next hour's load (Ridge regression)

Treatment-unit agents (each controls one piece of equipment in units.py)
    PreliminaryAgent     screen & grit
    EqualizationAgent    evens out flow, corrects pH, diverts shocks to the holding tank
    PrimaryAgent         primary clarifier
    AerationAgent        sets blowers and internal recycle
    SecondaryAgent       secondary clarifier
    FilterAgent          decides when to backwash
    DisinfectionAgent    sets the chlorine dose
    ComplianceAgent      checks the Indian norms; sends failing water back

Control modes (set on the model):
    "ai"     - smart control, ML + rule detection, ML forecasting
    "rules"    - smart control, rule detection only, no forecasting
    "operator" - a WELL-RUN conventional plant: DO feedback control on the
                 blowers (standard SCADA), recycle fixed at its design value, fixed chlorine and filter
                 timer, and a human who acts on alarms after
                 config.OPERATOR_DELAY_HOURS (divert, pH dosing, storm pumping)
    "fixed"    - a conventional plant: fixed settings, nobody acts on alarms
"""

import math

import mesa
import pandas as pd

from .config import PLANT, DESIGN_FLOW, OPERATOR_DELAY_HOURS, OPERATOR_RECYCLE
from .ml import anomaly_features, rule_check, classify_event
from .units import (Preliminary, EqualizationTank, HoldingTank, PrimaryClarifier,
                    AerationTank, SecondaryClarifier, TertiaryFilter, Chlorination,
                    do_saturation)
from .water import Water, mix, check_norms

# Shock types that should be kept OUT of the biology (diverted to holding).
# Storm water is NOT diverted: it is dilute, the plant has to take it.
# "unusual_load" (anomaly with no identified cause, usually an ML-only alarm)
# is reported but NOT diverted: tested over many random months, diverting on
# uncertain alarms wastes holding-tank space needed for real toxic shocks and
# lowers compliance (see results/EVALUATION.md, section 5).
DIVERT_TYPES = {"industrial_discharge", "acid_dump", "alkaline_dump", "septage_dumping"}

EVENT_NAMES = {
    "industrial_discharge": "industrial discharge", "acid_dump": "acid dump",
    "alkaline_dump": "alkaline dump", "storm_inflow": "storm inflow",
    "septage_dumping": "septage dumping", "unusual_load": "unusual load",
    "high_strength_food": "high-strength food waste",
}


class PlantAgent(mesa.Agent):
    """Common helpers for all agents."""
    label = "Agent"

    def send(self, receiver, kind, text, **data):
        return self.model.bus.send(self.model.hour, self.label, receiver, kind, text, **data)

    def inbox(self, kind=None):
        return self.model.bus.read(self.label, kind)

    @property
    def smart(self):
        return self.model.mode in ("ai", "rules")

    @property
    def operator_acts(self):
        """Operator mode: True once the current alarm has lasted long enough
        for the human operator to respond."""
        return self.model.mode == "operator" and self.model.alarm_streak > OPERATOR_DELAY_HOURS


# ===========================================================================
# Data & intelligence agents
# ===========================================================================
class InfluentFeedAgent(PlantAgent):
    """Replaces the inlet sensors: each hour it reads the next row of data."""
    label = "InfluentFeed"

    def step(self):
        m = self.model
        i = m.hour
        m.raw = m.feed.water(i)
        m.raw_event = m.feed.event(i)
        m.timestamp = m.feed.timestamp(i)
        row = m.feed.row(i).to_dict()
        m.history.append(row)


class MonitoringAgent(PlantAgent):
    """Watches the raw sewage. Two detectors:
        rules - fixed thresholds (pH, COD/BOD ratio, flow ...)
        ML    - Isolation Forest trained on a month of normal sewage
    In "ai" mode an alarm from either detector counts; in "rules" and
    "fixed" mode only the rules are used. The cause is named by
    ml.classify_event so the other agents know what to do."""
    label = "Monitoring"

    def step(self):
        m = self.model
        row = pd.DataFrame([m.history[-1]])
        feats = anomaly_features(row).iloc[0].to_dict()
        broken = rule_check(feats, m.rule_limits)
        rules_alarm = bool(broken)

        if m.hour in m.precomputed_scores and m.hour not in m.feed.injected:
            score = m.precomputed_scores[m.hour]
        else:
            score = float(m.detector.scores(row)[0])
        ml_alarm = score > m.detector.threshold

        alarm = (rules_alarm or ml_alarm) if m.mode == "ai" else rules_alarm
        kind = classify_event(feats, m.rule_limits) if alarm else None

        m.detection = {
            "rules": broken, "rules_alarm": rules_alarm,
            "ml_score": round(m.detector.normalised(score), 3), "ml_alarm": bool(ml_alarm),
            "alarm": alarm, "type": kind, "true_event": m.raw_event,
        }
        m.detections.append({"hour": m.hour, "true_event": m.raw_event,
                             "rules_alarm": rules_alarm, "ml_alarm": bool(ml_alarm), "alarm": alarm})

        m.alarm_streak = m.alarm_streak + 1 if alarm else 0
        if alarm:
            m.stats["alarms"] += 1
            why = ", ".join(broken) if broken else f"ML anomaly score {m.detection['ml_score']:.2f}"
            self.send("Equalization", "ALERT",
                      f"Shock load: {EVENT_NAMES.get(kind, kind)} ({why})", type=kind)


class ForecastAgent(PlantAgent):
    """Predicts this hour's influent from the hours before it (decided at the
    start of the hour, before the water is measured). Only used in "ai" mode;
    otherwise the plant assumes 'same as last hour'."""
    label = "Forecast"

    def step(self):
        m = self.model
        past = m.history[:-1]          # do not peek at the hour being predicted
        if len(past) < 24:
            m.forecast = None
            return
        last = past[-1]
        if m.mode == "ai":
            hist = pd.DataFrame(past[-24:])
            pred = m.forecaster.predict(hist, m.timestamp)
            source = m.forecaster.source()
        else:
            pred = {"flow": last["flow"], "BOD": last["BOD"], "NH4": last["NH4"]}
            source = "last hour"
        m.forecast = {**pred, "source": source}
        actual = m.history[-1]
        m.forecast_errors.append({k: abs(pred[k] - actual[k]) for k in ("flow", "BOD", "NH4")})
        if m.mode == "ai":
            self.send("Aeration", "FORECAST",
                      f"Expect {pred['flow']:.0f} m³/h, BOD {pred['BOD']:.0f}, NH4 {pred['NH4']:.1f} mg/L",
                      **pred)


# ===========================================================================
# Treatment-unit agents
# ===========================================================================
class PreliminaryAgent(PlantAgent):
    label = "Preliminary"

    def __init__(self, model):
        super().__init__(model)
        self.unit = Preliminary()

    def step(self):
        m = self.model
        m.streams["preliminary"], m.unit_info["preliminary"] = self.unit.process(m.raw)


class EqualizationAgent(PlantAgent):
    """Runs the equalization tank and the emergency holding tank.
    Smart decisions:
      - divert a toxic/strong shock into the holding tank (on ALERT)
      - feed the holding tank back slowly once the biology is healthy
      - dose lime/acid to keep pH near 7.2
    Always (plant hardware): level control and overflow."""
    label = "Equalization"

    def __init__(self, model):
        super().__init__(model)
        self.tank = EqualizationTank()
        self.holding = HoldingTank()
        self.avg_inflow = DESIGN_FLOW

    def step(self):
        m = self.model
        inflow = m.streams["preliminary"]
        info = {"diverted": 0.0, "released": 0.0, "ph_shift": 0.0, "overflow": 0.0}

        alerts = self.inbox("ALERT")
        shock = next((a for a in alerts if a.data.get("type") in DIVERT_TYPES), None)

        acts = self.smart or self.operator_acts

        # 1. divert a shock load into the holding tank
        if acts and shock is not None:
            before = inflow.flow
            inflow = self.holding.add(inflow)
            info["diverted"] = before - inflow.flow
            m.stats["diverted_m3"] += info["diverted"]
            if inflow.flow > 0:
                self.send("Compliance", "HOLDING_FULL",
                          f"Holding tank full - {inflow.flow:.0f} m³ of shock load entering the plant")
            else:
                self.send("Monitoring", "DIVERTED",
                          f"Diverted {info['diverted']:.0f} m³ to holding tank ({self.holding.level:.0%} full)")

        # 2. returned water: failed effluent and filter backwash from last hour
        returned = [w for w in (m.recirc_water, m.backwash_water) if w is not None and w.flow > 0]
        for w in returned:
            inflow = mix(inflow, w)
        m.recirc_water = m.backwash_water = None

        # 3. slowly release the holding tank when it is safe
        if (self.smart or m.mode == "operator") and shock is None and self.holding.volume > 0 \
                and m.aeration_health > 0.8:
            # release faster when the stored water is mild, slower when it is strong
            held = self.holding.contents
            rate = 0.12 * DESIGN_FLOW * min(1.0, 300.0 / max(held.inert_COD, 1.0), 600.0 / max(held.BOD, 1.0))
            # never release so fast that the blend becomes toxic again (bacteria are
            # harmed above 150 mg/L non-biodegradable COD; keep a margin at 120)
            safe_inert = 120.0
            if held.inert_COD > safe_inert:
                rate = min(rate, inflow.flow * max(0.0, safe_inert - inflow.inert_COD)
                           / (held.inert_COD - safe_inert))
            released = self.holding.take(rate)
            info["released"] = released.flow
            inflow = mix(inflow, released)

        # 4. level control: send forward about the average inflow, nudged by the level
        self.avg_inflow = 0.9 * self.avg_inflow + 0.1 * inflow.flow
        outflow = self.avg_inflow + (self.tank.level - 0.5) * self.tank.capacity / 6
        storm = any(a.data.get("type") == "storm_inflow" for a in alerts)
        if acts and storm:
            # storm: pump forward at full capacity to keep room in the tank
            outflow = PLANT["max_forward_flow"]
            self.send("Monitoring", "STORM_MODE", "Storm inflow - forward pumps at maximum to avoid overflow")
        outflow = max(0.3 * DESIGN_FLOW, min(PLANT["max_forward_flow"], outflow))

        self.tank.capacity += outflow            # water leaves while water arrives
        rest = self.tank.add(inflow)
        self.tank.capacity -= outflow

        # 5. pH correction (smart only)
        # (the operator also neutralises while draining the holding tank)
        releasing = m.mode == "operator" and info["released"] > 0
        if (acts or releasing) and not 6.5 <= self.tank.contents.pH <= 8.5:
            before_ph = self.tank.contents.pH
            info["ph_shift"] = self.tank.correct_ph(7.2)
            m.stats["ph_chemical_kg"] += abs(info["ph_shift"]) * outflow * 0.04
            self.send("Equalization", "PH_DOSING",
                      f"pH {before_ph:.1f} -> {self.tank.contents.pH:.1f} (lime/acid dosing)")

        out = self.tank.take(outflow)
        overflow = rest
        if self.tank.volume > self.tank.capacity:
            overflow = mix(overflow, self.tank.take(self.tank.volume - self.tank.capacity))
        if overflow.flow > 0:
            info["overflow"] = overflow.flow
            m.stats["overflow_m3"] += overflow.flow
            self.send("Compliance", "OVERFLOW",
                      f"Equalization tank full: {overflow.flow:.0f} m³ raw sewage bypassing treatment")
        m.streams["bypass"] = overflow
        m.streams["equalization"] = out
        info.update(level_pct=round(self.tank.level * 100, 1), holding_pct=round(self.holding.level * 100, 1),
                    outflow=round(out.flow, 1))
        m.unit_info["equalization"] = info

    def predict_outflow(self, predicted_inflow):
        """What the tank will send forward if the inflow is as predicted.
        Used by the AerationAgent to set the blowers in advance."""
        contents = self.tank.contents.copy(flow=self.tank.volume)
        blended = mix(contents, predicted_inflow)
        return blended.copy(flow=max(0.3 * DESIGN_FLOW, min(PLANT["max_forward_flow"], self.avg_inflow)))


class PrimaryAgent(PlantAgent):
    label = "Primary"

    def __init__(self, model):
        super().__init__(model)
        self.unit = PrimaryClarifier()

    def step(self):
        m = self.model
        m.streams["primary"], m.unit_info["primary"] = self.unit.process(m.streams["equalization"])


class AerationAgent(PlantAgent):
    """Controls the blowers and the internal recycle pump.

    Smart: decides the blower power at the START of the hour by predicting
    the oxygen the bacteria will need (feed-forward), aiming for DO 2.0 mg/L,
    plus a correction for last hour's DO error (feedback). Adjusts the
    internal recycle to keep nitrate low.
    Operator: standard DO feedback controller (no forecast), recycle at the
    design value config.OPERATOR_RECYCLE.
    Fixed: blowers at 60% (sized for the average load), recycle at 3x."""
    label = "Aeration"
    DO_SETPOINT = 2.0

    def __init__(self, model):
        super().__init__(model)
        self.unit = AerationTank()
        self.trim = 0.0
        self.recycle = 3.0
        self.last_inlet = None
        self.op_blower = 0.60

    def planned_blower(self):
        m = self.model
        eq = m.agent_by_label["Equalization"]
        fc = m.forecast
        if fc is not None and len(m.history) >= 2:
            prev = m.history[-2]     # last hour's reading (this hour is not known yet)
            pred_raw = Water(flow=fc["flow"], pH=prev["pH"], temperature=prev["temperature"],
                             BOD=fc["BOD"], COD=fc["BOD"] * 1.9, TSS=prev["TSS"],
                             NH4=fc["NH4"], NO3=0.0, FC=prev["FC"])
            pred_raw = m.agent_by_label["Preliminary"].unit.process(pred_raw)[0]
            inlet = PrimaryClarifier().process(eq.predict_outflow(pred_raw))[0]
        elif self.last_inlet is not None:
            inlet = self.last_inlet
        else:
            return 0.75
        demand, *_ = self.unit.oxygen_demand(inlet, self.DO_SETPOINT, self.unit.health)
        sat = do_saturation(inlet.temperature)
        blower = demand / (PLANT["max_oxygen_transfer"] * (1 - self.DO_SETPOINT / sat))
        return max(0.1, min(1.0, blower + self.trim))

    def step(self):
        m = self.model
        inlet = m.streams["primary"]
        self.inbox("FORECAST")  # the forecast itself is read from m.forecast

        if self.smart:
            blower = self.planned_blower()
            last_no3 = m.last_effluent.NO3 if m.last_effluent is not None else 5.0
            self.recycle = max(PLANT["internal_recycle_range"][0],
                               min(PLANT["internal_recycle_range"][1], self.recycle + 0.3 * (last_no3 - 5.0) / 5.0))
        elif m.mode == "operator":
            # plain feedback: adjust last hour's blower setting by the DO error
            self.op_blower = max(0.1, min(1.0, self.op_blower + 0.12 * (self.DO_SETPOINT - self.unit.DO)))
            blower, self.recycle = self.op_blower, OPERATOR_RECYCLE
        else:
            blower, self.recycle = 0.60, 3.0

        health_before = self.unit.health
        out, info = self.unit.process(inlet, blower, self.recycle)
        if self.smart:
            self.trim = max(-0.3, min(0.3, self.trim + 0.05 * (self.DO_SETPOINT - self.unit.DO)))
        self.last_inlet = inlet
        m.aeration_health = self.unit.health
        m.stats["energy_kwh"] += info["kwh"]

        if info["toxicity"] > 0:
            self.send("Compliance", "TOXIC_SHOCK",
                      f"Bacteria poisoned: health {health_before:.0%} -> {self.unit.health:.0%}")
        if blower >= 0.99 and self.unit.DO < 1.0:
            self.send("Compliance", "BLOWERS_MAXED", f"Blowers at 100% but DO only {self.unit.DO:.1f} mg/L")
        m.streams["aeration"] = out
        m.unit_info["aeration"] = info


class SecondaryAgent(PlantAgent):
    label = "Secondary"

    def __init__(self, model):
        super().__init__(model)
        self.unit = SecondaryClarifier()

    def step(self):
        m = self.model
        m.streams["secondary"], m.unit_info["secondary"] = self.unit.process(m.streams["aeration"], m.aeration_health)


class FilterAgent(PlantAgent):
    """Smart: backwash when the filter is getting clogged, preferably at a
    low-flow hour. Fixed: backwash on a timer once a day at 02:00."""
    label = "Filter"

    def __init__(self, model):
        super().__init__(model)
        self.unit = TertiaryFilter()

    def step(self):
        m = self.model
        w = m.streams["secondary"]
        if self.smart:
            backwash = self.unit.clog > 0.85 or (self.unit.clog > 0.55 and w.flow < 0.8 * DESIGN_FLOW)
        else:
            backwash = m.timestamp.hour == 2
        out, info = self.unit.process(w, backwash)
        if backwash:
            m.backwash_water = m.streams["secondary"].copy(flow=info["backwash_water"])
            self.send("Filter", "BACKWASH", "Filter backwashed")
        m.streams["filter"] = out
        m.unit_info["filter"] = info


class DisinfectionAgent(PlantAgent):
    """Smart: works out the chlorine needed from the solids and ammonia it
    measures and an ESTIMATE of the coliforms (coliforms cannot be measured
    online, so the estimate carries some error). Fixed: 5 mg/L always."""
    label = "Disinfection"

    def __init__(self, model):
        super().__init__(model)
        self.unit = Chlorination()

    def step(self):
        m = self.model
        w = m.streams["filter"]
        if self.smart:
            fc_estimate = w.FC * 10 ** m.random.gauss(0, 0.25)
            logs_needed = max(0.0, math.log10(max(fc_estimate, 1.0) / 100.0))
            dose = self.unit.chlorine_demand(w) + logs_needed / 1.2 + 0.3
            dose = min(PLANT["max_chlorine_dose"], dose)
        else:
            dose = 5.0
        out, info = self.unit.process(w, dose)
        m.stats["chlorine_kg"] += info["chlorine_kg"]
        m.streams["chlorination"] = out
        m.unit_info["chlorination"] = info


class ComplianceAgent(PlantAgent):
    """Checks treated water against the chosen Indian norms. Smart: if it
    fails, sends up to 30% of design flow back to the equalization tank
    instead of the river (if the tank has room)."""
    label = "Compliance"

    def step(self):
        m = self.model
        treated = m.streams["chlorination"]
        _, ok = check_norms(treated, m.norms)
        self.inbox()   # warnings from other agents are already in the log

        recirc = 0.0
        eq = m.agent_by_label["Equalization"].tank
        if self.smart and not ok and eq.level < 0.85:
            recirc = min(treated.flow, PLANT["max_recirculation"])
            m.recirc_water = treated.copy(flow=recirc)
            m.stats["recirculated_m3"] += recirc
            self.send("Equalization", "RECIRCULATE",
                      f"Effluent fails norms - sending {recirc:.0f} m³ back for re-treatment")

        discharged = mix(treated.copy(flow=treated.flow - recirc), m.streams["bypass"])
        result, passed = check_norms(discharged, m.norms)
        if discharged.flow <= 1e-6:
            passed = True
        failing = [p for p, (_, good) in result.items() if not good] if discharged.flow > 1e-6 else []

        m.effluent = discharged
        m.last_effluent = treated
        m.compliance = {"passed": passed, "failing": failing,
                        "values": {p: v for p, (v, _) in result.items()},
                        "discharged_m3": round(discharged.flow, 1), "recirculated_m3": round(recirc, 1)}
        s = m.stats
        s["hours"] += 1
        s["discharged_m3"] += discharged.flow
        if passed:
            s["compliant_hours"] += 1
            s["compliant_m3"] += discharged.flow
        else:
            for p in failing:
                s["violations"][p] = s["violations"].get(p, 0) + 1
            self.send("Monitoring", "NON_COMPLIANT", f"Discharge fails: {', '.join(failing)}")
