"""
Tests: run with   python -m pytest -q

Each test checks one thing the project claims, so a failing test points at
a broken claim.
"""

import os
import sys

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
