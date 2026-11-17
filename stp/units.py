"""
units.py - The PHYSICAL plant: what each treatment unit does to the water.

This file has no decision-making. Each unit is a piece of equipment that
takes water in and gives treated water out, using simplified engineering
formulas. The decisions (how hard to run the blowers, how much chlorine to
add, when to divert a shock load) are made by the agents in agents.py.

Every unit has LIMITS, so treatment can fail:
    - the equalization tank can overflow,
    - the primary clarifier removes less when flow is high,
    - bacteria in the aeration tank are poisoned by acid, alkali or
      industrial chemicals and take days to recover,
    - blowers have a maximum oxygen capacity,
    - the filter clogs,
    - chlorine is used up by dirt and ammonia before it can kill bacteria.

Treatment train (a typical Indian activated sludge plant):
    Screen & grit -> Equalization -> Primary clarifier -> Aeration (with
    anoxic zone) -> Secondary clarifier -> Filter -> Chlorination -> River
"""

import math

from .config import PLANT, DESIGN_FLOW
from .water import Water, mix, empty_water, BIODEGRADABLE_COD_PER_BOD


def do_saturation(temperature):
    """Maximum dissolved oxygen (mg/L) water can hold at a temperature (°C)."""
    t = temperature
    return 14.62 - 0.3898 * t + 0.006969 * t ** 2 - 0.00005897 * t ** 3


def remove_particles(w: Water, tss_fraction_removed: float) -> Water:
    """Removing solids also removes the organic matter stuck to them.
    Rule of thumb: half of the BOD and COD in sewage is in particles."""
    r = max(0.0, min(1.0, tss_fraction_removed))
    inert = w.inert_COD * (1 - 0.5 * r)
    bod = w.BOD * (1 - 0.5 * r)
    return w.copy(TSS=w.TSS * (1 - r), BOD=bod, COD=BIODEGRADABLE_COD_PER_BOD * bod + inert)


# ---------------------------------------------------------------------------
class Preliminary:
    """Bar screens + grit chamber. Removes rags, plastic, sand."""
    name = "Screen & Grit"

    def process(self, w: Water):
        out = remove_particles(w, 0.08)
        return out, {"tss_removed_pct": 8.0}


# ---------------------------------------------------------------------------
class Tank:
    """A well-mixed tank: water coming in mixes with what is already there."""

    def __init__(self, capacity, fill_fraction=0.0, contents=None):
        self.capacity = capacity
        self.volume = capacity * fill_fraction
        self.contents = contents or empty_water()

    def add(self, w: Water):
        """Add water (the flow value is treated as a volume for one hour).
        Returns any water that did not fit."""
        space = max(0.0, self.capacity - self.volume)
        accepted = min(w.flow, space)
        if accepted > 0:
            self.contents = mix(self.contents.copy(flow=self.volume), w.copy(flow=accepted)).copy(flow=0.0)
            self.volume += accepted
        return w.copy(flow=w.flow - accepted)

    def take(self, amount):
        q = max(0.0, min(amount, self.volume))
        self.volume -= q
        return self.contents.copy(flow=q)

    @property
    def level(self):
        return self.volume / self.capacity


class EqualizationTank(Tank):
    """Evens out the morning and evening peaks so the biology gets a steady
    feed. A pH correction pump can add lime or acid to the tank."""
    name = "Equalization"

    def __init__(self):
        super().__init__(PLANT["eq_volume"], fill_fraction=0.5)
        self.contents = Water(flow=0.0, pH=7.4, temperature=28.0, BOD=230.0, COD=440.0,
                              TSS=275.0, NH4=25.0, NO3=0.0, FC=1e7)

    def correct_ph(self, target):
        """Move the tank pH toward target, limited by the dosing pump size.
        Returns the pH shift achieved."""
        gap = target - self.contents.pH
        shift = max(-PLANT["max_ph_correction"], min(PLANT["max_ph_correction"], gap))
        self.contents = self.contents.copy(pH=self.contents.pH + shift)
        return shift


class HoldingTank(Tank):
    """Emergency tank. Shock loads are diverted here and fed back slowly."""
    name = "Holding tank"

    def __init__(self):
        super().__init__(PLANT["holding_volume"], fill_fraction=0.0)


# ---------------------------------------------------------------------------
class PrimaryClarifier:
    """Settling tank. Removes about 60% of solids at design flow; less when
    the flow is high because particles have less time to sink."""
    name = "Primary clarifier"

    def process(self, w: Water):
        sor = w.flow * 24 / PLANT["primary_area"]          # surface overflow rate, m³/m²/day
        if sor <= 30:
            r = 0.60
        else:
            r = max(0.30, 0.60 - 0.30 * (sor - 30) / 30)
        out = remove_particles(w, r)
        out = out.copy(FC=w.FC * 0.7)
        return out, {"overflow_rate": round(sor, 1), "tss_removed_pct": round(r * 100, 1)}


# ---------------------------------------------------------------------------
class AerationTank:
    """Activated sludge with an anoxic zone. Bacteria eat BOD, turn ammonia
    into nitrate (needs oxygen), and turn nitrate into nitrogen gas in the
    anoxic zone (needs the internal recycle). Toxic water poisons them.

    health: 1.0 = healthy bacteria, 0.05 = almost dead. Recovers slowly
    (about 3% of the gap per hour, so several days after a bad shock)."""
    name = "Aeration"

    def __init__(self):
        self.health = 1.0
        self.DO = 2.0
        self.last = {}

    @staticmethod
    def toxicity(w: Water):
        """How toxic the incoming water is to the bacteria (0 = harmless)."""
        s = 0.0
        if w.pH < 6.0:
            s += (6.0 - w.pH) / 2.0
        if w.pH > 9.0:
            s += (w.pH - 9.0) / 2.0
        if w.inert_COD > 150:
            s += (w.inert_COD - 150) / 400
        return min(s, 1.0)

    def oxygen_demand(self, w: Water, do, health):
        """kg O2/h the bacteria need to treat water w at dissolved oxygen do."""
        theta = min(48.0, PLANT["aeration_volume"] / max(w.flow, 1.0))
        k = 8.0 * health * do / (0.2 + do) * min(1.25, 1.04 ** (w.temperature - 20))
        bod_out = w.BOD / (1 + k * theta)
        ml_ph = 7.2 + (w.pH - 7.2) * 0.35
        f_ph = 1.0 if 6.8 <= ml_ph <= 8.4 else max(0.1, 1 - 0.6 * min(abs(ml_ph - 6.8), abs(ml_ph - 8.4)))
        kn = 3.0 * health * do / (1.0 + do) * min(1.4, 1.08 ** (w.temperature - 20)) * f_ph
        nh4_out = w.NH4 / (1 + kn * theta)
        demand = w.flow * ((w.BOD - bod_out) * 1.1 + 4.57 * (w.NH4 - nh4_out)) / 1000 + 15.0
        return demand, bod_out, nh4_out, ml_ph, theta

    def process(self, w: Water, blower: float, internal_recycle: float):
        # 1. poisoning, then slow recovery
        tox = self.toxicity(w)
        if tox > 0:
            self.health = max(0.05, self.health * (1 - 0.35 * tox))
        self.health += 0.03 * (1 - self.health)

        # 2. find the dissolved oxygen where blower supply = bacteria demand
        sat = do_saturation(w.temperature)
        supply_max = max(blower, 0.0) * PLANT["max_oxygen_transfer"]
        do = self.DO
        for _ in range(12):
            demand, bod_out, nh4_out, ml_ph, theta = self.oxygen_demand(w, max(do, 0.01), self.health)
            target = sat * (1 - demand / supply_max) if supply_max > 0 else 0.0
            do = 0.5 * do + 0.5 * max(0.0, min(sat, target))
        self.DO = do
        demand, bod_out, nh4_out, ml_ph, theta = self.oxygen_demand(w, max(do, 0.01), self.health)

        # 3. denitrification in the anoxic zone (needs recycle and food)
        no3_formed = w.NO3 + (w.NH4 - nh4_out)
        carbon = min(1.0, w.BOD / (4 * no3_formed + 1e-9))
        r_dn = internal_recycle / (1 + internal_recycle) * 0.98 * self.health * (1 - min(0.5, 0.05 * do)) * carbon
        no3_out = no3_formed * (1 - r_dn)

        # 4. what leaves: biology cannot remove inert COD (only ~half sticks to the sludge)
        inert_out = w.inert_COD * 0.5
        out = w.copy(
            BOD=bod_out, COD=BIODEGRADABLE_COD_PER_BOD * bod_out + inert_out,
            NH4=nh4_out, NO3=no3_out, FC=w.FC * 0.1, TSS=3000.0,   # 3000 = mixed liquor solids
            pH=max(0.0, min(14.0, ml_ph - 0.01 * (w.NH4 - nh4_out))),
        )
        kwh = max(blower, 0.0) * PLANT["max_oxygen_transfer"] / PLANT["aeration_kg_o2_per_kwh"]
        self.last = {
            "DO": round(do, 2), "health": round(self.health, 3), "toxicity": round(tox, 2),
            "oxygen_demand": round(demand, 1), "blower_pct": round(blower * 100, 1),
            "internal_recycle": round(internal_recycle, 2), "hrt_hours": round(theta, 2),
            "bod_removed_pct": round(100 * (1 - bod_out / max(w.BOD, 1e-9)), 1),
            "nh4_removed_pct": round(100 * (1 - nh4_out / max(w.NH4, 1e-9)), 1),
            "kwh": round(kwh, 1),
        }
        return out, self.last


# ---------------------------------------------------------------------------
class SecondaryClarifier:
    """Settles the bacteria (sludge) out. Unhealthy sludge settles badly and
    high flow washes it over the weir, so effluent solids go up."""
    name = "Secondary clarifier"

    def process(self, w: Water, health: float):
        hydraulic = max(1.0, w.flow / DESIGN_FLOW) ** 1.5
        tss = min(400.0, 8.0 * (1 / max(health, 0.05)) ** 0.8 * hydraulic)
        out = w.copy(TSS=tss, BOD=w.BOD + 0.3 * tss, COD=w.COD + 0.9 * tss, FC=w.FC * 0.5)
        return out, {"effluent_tss": round(tss, 1)}


# ---------------------------------------------------------------------------
class TertiaryFilter:
    """Two dual-media filters. They clog as they catch solids and must be
    backwashed (cleaned). During a backwash one filter is offline."""
    name = "Filter"

    def __init__(self):
        self.clog = 0.0

    def process(self, w: Water, backwash: bool):
        if backwash:
            self.clog = 0.0
            r = 0.40
        elif self.clog >= 1.0:
            r = 0.0          # fully clogged: water bypasses the media
        else:
            r = 0.80 * (1 - 0.7 * self.clog)
        removed = w.TSS * r
        self.clog = min(1.0, self.clog + removed * w.flow / 1000 / PLANT["filter_solids_capacity"])
        bod = max(0.0, w.BOD - 0.3 * removed)
        cod = max(w.inert_COD * 0.5, w.COD - 0.9 * removed)
        out = w.copy(TSS=w.TSS - removed, BOD=bod, COD=cod, FC=w.FC * 10 ** (-0.5 * r / 0.8))
        return out, {"clog_pct": round(self.clog * 100, 1), "tss_removed_pct": round(r * 100, 1),
                     "backwash": backwash, "backwash_water": 0.05 * w.flow if backwash else 0.0}


# ---------------------------------------------------------------------------
class Chlorination:
    """Chlorine contact tank. Dirt (TSS) and ammonia use up chlorine first;
    only what is left over kills coliform bacteria."""
    name = "Chlorination"

    @staticmethod
    def chlorine_demand(w: Water):
        return 0.5 + 0.05 * w.TSS + 0.3 * w.NH4

    def process(self, w: Water, dose: float):
        effective = max(0.0, dose - self.chlorine_demand(w))
        log_kill = min(6.0, 1.2 * effective)
        out = w.copy(FC=w.FC * 10 ** (-log_kill))
        return out, {"dose": round(dose, 2), "log_kill": round(log_kill, 2),
                     "residual_chlorine": round(0.6 * effective, 2),
                     "chlorine_kg": round(dose * w.flow / 1000, 2)}
