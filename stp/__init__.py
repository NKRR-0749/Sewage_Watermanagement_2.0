"""stp - Autonomous Sewage Treatment Plant simulation (Mesa multi-agent system).

Read the files in this order:
    config.py    numbers: Indian norms, plant size, raw sewage
    water.py     the water "packet" and the norms check
    influent.py  the data fed into the plant (replaces sensors)
    units.py     the physical treatment units
    ml.py        the AI: anomaly detector + load forecaster
    messages.py  how agents talk
    agents.py    the eleven Mesa agents (the decisions)
    model.py     the Mesa model that ties it all together
"""

from .model import STPModel, MODES
from .config import DISCHARGE_NORMS

__all__ = ["STPModel", "MODES", "DISCHARGE_NORMS"]
