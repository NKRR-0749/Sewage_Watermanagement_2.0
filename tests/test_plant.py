"""
Tests: run with   python -m pytest -q

Each test checks one thing the project claims, so a failing test points at
a broken claim.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stp import STPModel  # noqa: E402
from stp.influent import demo_dataset, generate_influent, load_influent_csv  # noqa: E402
from stp.units import AerationTank  # noqa: E402
from stp.water import Water, check_norms, mix  # noqa: E402

CLEAN = Water(flow=400, pH=7.2, temperature=28, BOD=5, COD=30, TSS=5, NH4=1, NO3=5, FC=50)


def test_norms_pass_clean_water_and_fail_dirty_water():
    _, ok = check_norms(CLEAN, "ngt2019")
    assert ok
    result, ok = check_norms(CLEAN.copy(BOD=15), "ngt2019")
    assert not ok and result["BOD"][1] is False
    # BOD 15 is allowed under the 2017 metro norms (limit 20)
    _, ok = check_norms(CLEAN.copy(BOD=15), "moefcc2017_metro")
    assert ok


def test_mixing_conserves_load_and_neutralises_ph():
    a = CLEAN.copy(flow=100, BOD=100)
    b = CLEAN.copy(flow=300, BOD=20)
    m = mix(a, b)
    assert m.flow == 400
    assert abs(m.BOD - 40) < 1e-9
    acid = CLEAN.copy(flow=100, pH=3.0)
    base = CLEAN.copy(flow=100, pH=11.0)
    assert abs(mix(acid, base).pH - 7.0) < 0.05


def test_toxic_water_poisons_bacteria():
    tank = AerationTank()
    acid = Water(flow=417, pH=3.0, temperature=28, BOD=170, COD=330, TSS=120, NH4=25)
    for _ in range(3):
        tank.process(acid, blower=0.8, internal_recycle=3)
    assert tank.health < 0.5


def test_treatment_can_fail():
    """The old project certified sludge as pure water. This one must not."""
    df = generate_influent(days=3, seed=3)
    df.loc[30:40, ["pH", "COD"]] = [2.0, 3000.0]
    df.loc[30:40, "event"] = "industrial_discharge"
    m = STPModel(mode="fixed", data=df).run()
    assert m.summary()["compliant_hours_pct"] < 90


def test_autonomous_beats_fixed_on_demo_month():
    fixed = STPModel(mode="fixed").run().summary()
    ai = STPModel(mode="ai").run().summary()
    assert ai["compliant_hours_pct"] > fixed["compliant_hours_pct"] + 30
    assert ai["diverted_m3"] > 0 and fixed["diverted_m3"] == 0


def test_detectors_are_graded_against_true_labels():
    m = STPModel(mode="ai").run(24 * 10)
    det = m.summary()["detection"]
    assert det["rules"]["recall"] > 0.8
    assert det["used_by_plant"]["true_positives"] >= det["rules"]["true_positives"]


def test_runs_are_repeatable():
    a = STPModel(mode="ai", seed=5).run(100).summary()
    b = STPModel(mode="ai", seed=5).run(100).summary()
    assert a == b


def test_values_stay_physical():
    m = STPModel(mode="ai").run(24 * 7)
    df = m.datacollector.get_model_vars_dataframe()
    for col in ["effluent_BOD", "effluent_COD", "effluent_TSS", "effluent_TN", "effluent_FC", "DO"]:
        assert (df[col] >= 0).all(), col
    assert df["effluent_pH"].between(0, 14).all()
    assert df["bacteria_health"].between(0.05, 1.0).all()


def test_csv_upload_validation():
    text = demo_dataset().head(30).to_csv(index=False)
    assert len(load_influent_csv(text)) == 30
    with pytest.raises(ValueError):
        load_influent_csv("flow,pH\n400,7\n")


def test_injected_event_is_detected_and_diverted():
    m = STPModel(mode="ai").run(30)
    m.inject("acid_dump", hours=3)
    m.step()
    assert m.detection["alarm"] and m.detection["type"] == "acid_dump"
    assert m.unit_info["equalization"]["diverted"] > 0


# ---------------------------------------------------------------------------
# Real data (UCI Water Treatment Plant, Manresa)
# ---------------------------------------------------------------------------
from stp import real_data  # noqa: E402
from stp.config import DESIGN_FLOW  # noqa: E402
from stp.ml import DEFAULT_RULE_LIMITS, calibrate_rule_limits  # noqa: E402


def test_real_dataset_loads_and_converts_to_hourly():
    daily = real_data.load_uci_daily()
    assert len(daily) == 520
    hourly = real_data.uci_hourly()
    assert len(hourly) == 520 * 24
    assert hourly[["flow", "pH", "BOD", "COD", "TSS", "NH4"]].isna().sum().sum() == 0
    # concentrations are used as measured: daily means are preserved
    assert abs(hourly["BOD"].mean() - daily["BOD"].mean()) / daily["BOD"].mean() < 0.02
    # flow is scaled so the median day equals our 10 MLD design flow
    assert abs(hourly.groupby("day")["flow"].sum().median() - DESIGN_FLOW * 24) / (DESIGN_FLOW * 24) < 0.03


def test_real_train_and_test_days_do_not_overlap():
    train, test = real_data.uci_train(), real_data.uci_test()
    assert train["day"].max() == real_data.TRAIN_DAYS
    assert test["day"].min() == real_data.TRAIN_DAYS + 1


def test_rule_calibration_never_loosens_ph_and_never_goes_below_default():
    limits = calibrate_rule_limits(real_data.uci_train())
    assert limits["pH_low"] == DEFAULT_RULE_LIMITS["pH_low"]
    assert limits["pH_high"] == DEFAULT_RULE_LIMITS["pH_high"]
    assert all(limits[k] >= DEFAULT_RULE_LIMITS[k] for k in DEFAULT_RULE_LIMITS)


def test_plant_runs_on_real_data_with_site_calibration():
    from stp.evaluation_real import site_setup
    _, _, site = site_setup()
    m = STPModel(mode="ai", data=real_data.uci_test(3), **site).run()
    s = m.summary()
    assert s["hours"] == 72
    assert 0 <= s["compliant_hours_pct"] <= 100


def test_injected_shocks_on_real_data_are_caught():
    from stp.evaluation_real import site_setup
    _, _, site = site_setup()
    data = real_data.uci_test_with_shocks(days=20)
    assert (data["event"] != "normal").sum() > 0
    m = STPModel(mode="ai", data=data, **site).run()
    assert m.summary()["detection"]["used_by_plant"]["recall"] > 0.7


# ---------------------------------------------------------------------------
# Baselines, overflow, unseen shocks, forecaster self-check
# ---------------------------------------------------------------------------
def test_overflow_counts_against_compliance():
    """Raw sewage bypassing treatment must make the discharge fail."""
    from stp.influent import apply_event
    df = demo_dataset().iloc[:72].copy()
    rows = df.to_dict("records")
    for i in range(30, 60):                     # a very long, very big storm
        rows[i] = apply_event(rows[i], "storm_inflow", 5)
        rows[i]["flow"] *= 2.0
    m = STPModel(mode="ai", data=pd.DataFrame(rows)).run()
    assert m.stats["overflow_m3"] > 0
    assert m.summary()["compliant_hours_pct"] < 100


def test_operator_mode_sits_between_fixed_and_ai():
    res = {mode: STPModel(mode=mode).run().summary()["compliant_hours_pct"]
           for mode in ("ai", "operator", "fixed")}
    assert res["fixed"] < res["operator"] < res["ai"]


def test_ml_catches_unseen_ammonia_spike_that_rules_miss():
    from stp.influent import UNSEEN_DURATION, random_month
    data = random_month(seed=100, n_events=6, event_types=["ammonia_spike"], durations=UNSEEN_DURATION)
    det = STPModel(mode="ai", data=data).run().summary()["detection"]
    assert det["ml"]["recall"] > 0.9
    assert det["rules"]["recall"] < 0.5


def test_forecaster_only_uses_ml_where_it_beat_last_hour():
    from stp.ml import get_models
    fc = get_models()[1]
    for t, v in fc.validation.items():
        assert fc.use_ml[t] == (v["ml_mae"] < v["last_hour_mae"])
