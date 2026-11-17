"""
water.py - The "packet" of water that moves through the plant every hour.

A Water object holds the flow (how much water) and its quality (what is in it).
Every treatment unit takes a Water object in and gives a new, cleaner Water
object out. The helper functions here mix two streams and check a stream
against the discharge norms.
"""

from dataclasses import dataclass, asdict, replace
import math

from .config import DISCHARGE_NORMS

# COD that bacteria can eat is about 1.7 x BOD. Anything above that is
# "inert" COD - mostly industrial chemicals that biology cannot remove.
BIODEGRADABLE_COD_PER_BOD = 1.7

QUALITY_FIELDS = ["pH", "temperature", "BOD", "COD", "TSS", "NH4", "NO3", "FC"]


@dataclass
class Water:
    flow: float          # m³/h
    pH: float
    temperature: float   # °C
    BOD: float           # mg/L  biochemical oxygen demand (organic pollution)
    COD: float           # mg/L  chemical oxygen demand (all oxidisable pollution)
    TSS: float           # mg/L  total suspended solids
    NH4: float           # mg/L  ammonia nitrogen
    NO3: float = 0.0     # mg/L  nitrate nitrogen (made from ammonia in aeration)
    FC: float = 0.0      # MPN/100 mL faecal coliform bacteria

    @property
    def TN(self):
        """Total nitrogen = ammonia + nitrate (organic nitrogen ignored)."""
        return self.NH4 + self.NO3

    @property
    def inert_COD(self):
        """COD that bacteria cannot break down."""
        return max(0.0, self.COD - BIODEGRADABLE_COD_PER_BOD * self.BOD)

    def copy(self, **changes):
        return replace(self, **changes)

    def to_dict(self):
        d = asdict(self)
        d["TN"] = self.TN
        return d

    def rounded(self):
        d = self.to_dict()
        out = {}
        for k, v in d.items():
            if k == "FC":
                out[k] = float(f"{v:.3g}")
            else:
                out[k] = round(v, 2)
        return out


def empty_water(temperature=28.0):
    return Water(flow=0.0, pH=7.0, temperature=temperature, BOD=0.0, COD=0.0,
                 TSS=0.0, NH4=0.0, NO3=0.0, FC=0.0)


def mix(a: Water, b: Water) -> Water:
    """Mix two streams. Concentrations are flow-weighted averages.
    pH is mixed through hydrogen-ion concentration, not averaged directly."""
    total = a.flow + b.flow
    if total <= 0:
        return a.copy(flow=0.0)
    wa, wb = a.flow / total, b.flow / total
    values = {}
    for f in ["temperature", "BOD", "COD", "TSS", "NH4", "NO3", "FC"]:
        values[f] = getattr(a, f) * wa + getattr(b, f) * wb
    values["pH"] = mix_ph(a.pH, b.pH, wa)
    return Water(flow=total, **values)


def mix_ph(ph_a, ph_b, fraction_a):
    """pH of a mixture. Acid (H+) and base (OH-) neutralise each other, so we
    mix the ion concentrations, then solve [H+] - Kw/[H+] = net excess acid.
    (Simplification: the natural buffering of sewage is ignored, so this is
    the worst case for an acid or alkaline shock.)"""
    kw = 1e-14
    net = 0.0
    for ph, w in ((ph_a, fraction_a), (ph_b, 1 - fraction_a)):
        net += w * (10 ** (-ph) - kw / 10 ** (-ph))
    h = (net + math.sqrt(net * net + 4 * kw)) / 2
    return min(14.0, max(0.0, -math.log10(h)))


def check_norms(water: Water, norms_key: str):
    """Return a dict {parameter: (value, passed)} and an overall pass/fail."""
    limits = DISCHARGE_NORMS[norms_key]["limits"]
    result = {}
    for param, (low, high) in limits.items():
        value = water.TN if param == "TN" else getattr(water, param)
        ok = (low is None or value >= low) and (high is None or value <= high)
        result[param] = (value, ok)
    return result, all(ok for _, ok in result.values())
