"""
config.py - Every number the plant uses, in one place.

Three groups of settings live here:

1. DISCHARGE_NORMS  - the Indian limits that treated sewage must meet before it
                      leaves the plant (NGT 2019 order and MoEFCC 2017 notification).
2. PLANT            - the size and capacity of each treatment unit of a
                      10 MLD (million litres per day) sewage treatment plant.
3. RAW_SEWAGE       - typical Indian domestic raw sewage quality, used to
                      generate the influent data that is fed into the plant.

Keeping them here means a viva question like "where does BOD 10 come from?"
can be answered by pointing at one line.
"""

# ---------------------------------------------------------------------------
# 1. Discharge norms for treated sewage (India)
# ---------------------------------------------------------------------------
# Each parameter is (lowest allowed, highest allowed). None = no limit.
#
# NGT 2019: National Green Tribunal order dated 30 April 2019 (OA 1069/2018),
#   as quoted in CPCB compliance reports: pH 5.5-9.0, BOD < 10 mg/L,
#   TSS < 20 mg/L, COD < 50 mg/L, Total Nitrogen < 10 mg/L,
#   Faecal Coliform desirable < 100, permissible < 230 MPN/100 mL.
#   (Total Phosphorus < 1 mg/L applies only to discharge into lakes/ponds
#   and is not modelled here.)
#
# MoEFCC 2017: notification dated 13 October 2017 (G.S.R. 1265(E)).
#   Metro cities and state capitals: BOD 20, TSS 50. Other areas: BOD 30,
#   TSS 100. pH 6.5-9.0 and Faecal Coliform < 1000 MPN/100 mL for both.
#   No COD or nitrogen limit.
DISCHARGE_NORMS = {
    "ngt2019": {
        "label": "NGT 2019 (strict, all cities)",
        "limits": {
            "pH": (5.5, 9.0),
            "BOD": (None, 10.0),
            "COD": (None, 50.0),
            "TSS": (None, 20.0),
            "TN": (None, 10.0),
            "FC": (None, 230.0),
        },
    },
    "moefcc2017_metro": {
        "label": "MoEFCC 2017 - metro cities & state capitals",
        "limits": {
            "pH": (6.5, 9.0),
            "BOD": (None, 20.0),
            "TSS": (None, 50.0),
            "FC": (None, 1000.0),
        },
    },
    "moefcc2017_other": {
        "label": "MoEFCC 2017 - other areas",
        "limits": {
            "pH": (6.5, 9.0),
            "BOD": (None, 30.0),
            "TSS": (None, 100.0),
            "FC": (None, 1000.0),
        },
    },
}
DEFAULT_NORMS = "ngt2019"

# Units shown in reports and the dashboard
UNITS = {
    "flow": "m³/h", "pH": "", "temperature": "°C", "BOD": "mg/L", "COD": "mg/L",
    "TSS": "mg/L", "NH4": "mg/L", "NO3": "mg/L", "TN": "mg/L", "FC": "MPN/100mL",
}


# ---------------------------------------------------------------------------
# 2. Plant design: a 10 MLD activated sludge plant
# ---------------------------------------------------------------------------
DESIGN_FLOW = 10_000 / 24          # 10 MLD = 10,000 m³/day = ~417 m³/hour

PLANT = {
    "design_flow": DESIGN_FLOW,

    # Equalization tank: evens out the daily peaks. 6 hours of average flow.
    "eq_volume": 6 * DESIGN_FLOW,
    # Forward pumps after the equalization tank can push at most 1.8x design flow.
    "max_forward_flow": 1.8 * DESIGN_FLOW,
    # Emergency holding tank: a shock load can be diverted here and fed back slowly.
    "holding_volume": 4000.0,
    # pH correction pump at the equalization tank: max pH shift it can achieve per hour.
    "max_ph_correction": 1.5,

    # Primary clarifier sized for a surface overflow rate of 30 m³/m²/day at design flow.
    "primary_area": 10_000 / 30,

    # Aeration tank: 6 hours hydraulic retention time at design flow.
    "aeration_volume": 6 * DESIGN_FLOW,
    # Blowers: max oxygen they can transfer (kg O2/hour) when the tank has zero DO.
    "max_oxygen_transfer": 320.0,
    # Aeration efficiency: kg of oxygen delivered per kWh of blower electricity.
    "aeration_kg_o2_per_kwh": 1.8,
    # Internal recycle (nitrate back to the anoxic zone) range, as a multiple of flow.
    "internal_recycle_range": (1.0, 6.0),

    # Tertiary filter: kg of solids it can hold before it is fully clogged.
    "filter_solids_capacity": 150.0,

    # Chlorine dosing pump: maximum dose in mg/L.
    "max_chlorine_dose": 10.0,

    # Treated water that fails the norms can be sent back to the equalization
    # tank instead of the river, up to this fraction of design flow.
    "max_recirculation": 0.3 * DESIGN_FLOW,
}


# ---------------------------------------------------------------------------
# 3. Typical Indian domestic raw sewage
# ---------------------------------------------------------------------------
# Typical fresh domestic sewage in India: BOD 200-300 mg/L, COD 400-600 mg/L,
# suspended solids 200-400 mg/L, pH about 6.8-8.0 (CPHEEO Manual on Sewerage and
# Sewage Treatment, 2013). Average nitrogen in Indian municipal wastewater is
# about 30 mg/L; most of it is ammonia. Raw sewage typically carries
# 10^6 - 10^8 faecal coliforms per 100 mL. Indian sewage is warm: 24-32 °C.
RAW_SEWAGE = {
    "BOD": 250.0,          # mg/L, daily average
    "COD_BOD_ratio": 1.9,  # domestic sewage COD is roughly 1.8-2.2 x BOD
    "TSS": 300.0,          # mg/L
    "NH4": 25.0,           # mg/L ammonia nitrogen (rest of the ~30 mg/L total N is organic)
    "pH": 7.4,
    "temperature": 28.0,   # °C
    "FC": 1.0e7,           # MPN/100 mL
}

# "operator" mode: a well-run conventional plant. Standard DO feedback control
# on the blowers, and a human operator who acts on alarms (divert a shock,
# dose lime/acid, storm pumping) only after the alarm has lasted this long:
# time to notice, walk to the panel, confirm with a grab sample.
OPERATOR_DELAY_HOURS = 3
# The operator's fixed internal recycle, set at design time high enough for the
# NGT nitrogen limit (the "fixed" plant keeps a generic 3x).
OPERATOR_RECYCLE = 5.0

# Simulation time step: one step = one hour of plant operation.
HOURS_PER_STEP = 1
