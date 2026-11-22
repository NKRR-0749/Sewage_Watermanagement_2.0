"""
Build the website that GitHub Pages serves, into the folder _site/.

    python scripts/build_site.py

What goes in _site/:
    index.html, style.css, app.js, worker.js   the dashboard (from web/)
    stp_code.zip                                the Python package stp/ + web/bridge.py + real dataset
    wheels/mesa-*.whl + manifest.json           Mesa, installed into the browser's Python
    sample_upload.csv                           example data file
    summary.json                                key results for the start screen

The GitHub Actions workflow (.github/workflows/pages.yml) runs this script
on every push, so you never need to run it yourself unless you want to test
the site locally:

    python scripts/build_site.py
    python -m http.server 8000 --directory _site
    then open http://localhost:8000
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "_site")
MESA_VERSION = "3.5.2"


def main():
    if os.path.exists(SITE):
        shutil.rmtree(SITE)
    os.makedirs(os.path.join(SITE, "wheels"))

    for name in ["index.html", "style.css", "app.js", "worker.js"]:
        shutil.copy(os.path.join(ROOT, "web", name), SITE)
    shutil.copy(os.path.join(ROOT, "data", "sample_upload.csv"), SITE)
    summary = os.path.join(ROOT, "results", "summary.json")
    if os.path.exists(summary):          # key results, shown while Python loads
        shutil.copy(summary, SITE)
    open(os.path.join(SITE, ".nojekyll"), "w").close()

    with zipfile.ZipFile(os.path.join(SITE, "stp_code.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(os.path.join(ROOT, "stp"))):
            if f.endswith(".py"):
                z.write(os.path.join(ROOT, "stp", f), f"stp/{f}")
        z.write(os.path.join(ROOT, "web", "bridge.py"), "bridge.py")
        # the real dataset (read by stp/real_data.py from ../data/real/)
        z.write(os.path.join(ROOT, "data", "real", "uci_water_treatment.data"), "data/real/uci_water_treatment.data")

    wheel_dir = os.path.join(SITE, "wheels")
    subprocess.run([sys.executable, "-m", "pip", "download", f"mesa=={MESA_VERSION}", "--no-deps",
                    "--only-binary=:all:", "-d", wheel_dir, "-q"], check=True)
    wheel = next(f for f in os.listdir(wheel_dir) if f.startswith("mesa-") and f.endswith(".whl"))
    with open(os.path.join(wheel_dir, "manifest.json"), "w") as f:
        json.dump({"mesa": wheel}, f)

    print(f"Site built in {SITE}")
    for f in sorted(os.listdir(SITE)):
        print("  ", f)


if __name__ == "__main__":
    main()
