"""
Run the plant from the command line.

    python scripts/run_simulation.py                       # AI mode, NGT 2019 norms, demo month
    python scripts/run_simulation.py --mode fixed
    python scripts/run_simulation.py --norms moefcc2017_metro
    python scripts/run_simulation.py --data my_readings.csv --hours 48
    python scripts/run_simulation.py --real                 # REAL data (UCI, Manresa plant)
    python scripts/run_simulation.py --real --days 60

Prints a summary and saves every hour to results/run_<mode>.csv
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stp import STPModel, MODES, DISCHARGE_NORMS  # noqa: E402
from stp.influent import load_influent_csv  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Autonomous sewage treatment plant (Mesa)")
    p.add_argument("--mode", choices=list(MODES), default="ai")
    p.add_argument("--norms", choices=list(DISCHARGE_NORMS), default="ngt2019")
    p.add_argument("--data", help="CSV of hourly influent readings (default: built-in demo month)")
    p.add_argument("--hours", type=int, help="hours to simulate (default: whole dataset)")
    p.add_argument("--real", action="store_true",
                   help="use the REAL UCI Water Treatment Plant data (test days; ML + rules calibrated on training days)")
    p.add_argument("--days", type=int, help="with --real: only the first N test days")
    args = p.parse_args()

    site, label = {}, args.data or "built-in demo month (simulated, July)"
    if args.real:
        from stp import real_data
        from stp.evaluation_real import site_setup
        data = real_data.uci_test(args.days)
        _, limits, site = site_setup()
        label = f"REAL - {real_data.SOURCE}; calibrated rule limits {limits}"
    else:
        data = load_influent_csv(args.data) if args.data else None
    model = STPModel(mode=args.mode, norms=args.norms, data=data, **site)
    print(f"Mode:  {MODES[args.mode]}")
    print(f"Norms: {DISCHARGE_NORMS[args.norms]['label']}")
    print(f"Data:  {label} - {len(model.feed)} hours")
    model.run(args.hours)

    s = model.summary()
    print("\n--- Result ---")
    print(f"Hours meeting the norms:   {s['compliant_hours_pct']}%")
    print(f"Volume meeting the norms:  {s['compliant_volume_pct']}%")
    print(f"Violation hours by param:  {s['violation_hours_by_parameter'] or 'none'}")
    print(f"Blower energy:             {s['energy_kwh']:,} kWh")
    print(f"Chlorine used:             {s['chlorine_kg']:,} kg")
    print(f"Shock load diverted:       {s['diverted_m3']:,} m³")
    print(f"Raw sewage overflow:       {s['overflow_m3']:,} m³")
    print(f"Alarms raised:             {s['alarms']}")
    if args.real:
        print("\nThe real data has no shock labels, so detectors are not graded here.")
        print("Run  python scripts/evaluate_real.py  (grades them on real data + injected shocks).")
    else:
        print("\nDetector scores against the true event labels:")
        print(json.dumps(s["detection"], indent=2))

    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"run_{args.mode}{'_real' if args.real else ''}.csv")
    model.datacollector.get_model_vars_dataframe().to_csv(path, index=False)
    print(f"\nHourly results saved to {path}")


if __name__ == "__main__":
    main()
