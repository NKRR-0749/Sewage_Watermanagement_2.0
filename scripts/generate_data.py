"""
Generate the two influent datasets and save them as CSV in data/.

    python scripts/generate_data.py

data/influent_demo_july.csv      the month the plant is tested on (9 labelled shock events)
data/influent_training_june.csv  a separate normal-only month the ML models learn from
data/sample_upload.csv           a 3-day file showing the format for your own data
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stp.influent import demo_dataset, training_dataset  # noqa: E402


def main():
    out = os.path.join(ROOT, "data")
    os.makedirs(out, exist_ok=True)
    demo = demo_dataset()
    train = training_dataset()
    demo.to_csv(os.path.join(out, "influent_demo_july.csv"), index=False)
    train.to_csv(os.path.join(out, "influent_training_june.csv"), index=False)
    demo.head(72).to_csv(os.path.join(out, "sample_upload.csv"), index=False)
    print(f"Demo month:     {len(demo)} hours, {int((demo['event'] != 'normal').sum())} event hours")
    print(f"Training month: {len(train)} hours, all normal")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
