# Autonomous Sewage Treatment Plant — Mesa Multi-Agent System

**Sewage water quality monitoring, anomaly detection and autonomous treatment for an Indian sewage treatment plant, built with Mesa agents.**

▶ **Live dashboard:** https://nkrr-0749.github.io/Sewage_Watermanagement_2.0/
(opens in any browser, phone included. The Python and Mesa code runs inside the page, so first load takes about 20–40 s.)

> New to the project? Read **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** first. It explains every part in plain language.

---

## What it does

A 10 MLD (10 million litres/day) activated-sludge plant is simulated hour by hour. **Eleven Mesa agents** run it:

- Real-world-style **influent data is fed in instead of sensors**: daily flow peaks, monsoon storms, and shock events like industrial discharge, acid dumps and septage dumping.
- A **Monitoring agent** detects shocks with **rules + an Isolation Forest (ML)** and names the cause.
- **Control agents** act: divert toxic water to a holding tank, dose lime or acid, run the blowers to an oxygen target (using an **ML load forecast**), adjust nitrate recycle, decide filter backwashes, set the chlorine dose, and send failing water back for re-treatment.
- A **Compliance agent** checks every hour against Indian norms: **NGT 2019** (BOD ≤ 10, COD ≤ 50, TSS ≤ 20, TN ≤ 10 mg/L, FC ≤ 230 MPN/100 mL, pH 5.5–9) or **MoEFCC 2017**.

The treatment **can fail**. Bacteria get poisoned, blowers max out, tanks overflow, filters clog. The results are therefore measured, not guaranteed.

## Results (demo month, NGT 2019 norms)

| Control mode | Hours meeting norms | Bad hours | Energy (kWh) | Chlorine (kg) |
|---|---|---|---|---|
| **AI** (ML detection + ML forecast) | **95.7 %** | 31 | 75,539 | 1,296 |
| Rules only (no ML) | 95.6 % | 32 | 75,708 | 1,298 |
| Conventional (fixed settings) | 15.8 % | 606 | 76,824 | 1,514 |

| Detector (graded against the true labels in the data) | Precision | Recall | False alarms |
|---|---|---|---|
| Rules | 1.00 | 0.90 | 0 |
| Isolation Forest (ML) | 0.91 | 0.75 | 4 |
| Both (used in AI mode) | 0.93 | **0.94** | 4 |

Rules catch every pH shock that ML misses. ML catches the subtle food-waste load that no rule catches. The ML forecast cuts the flow-prediction error by 37 % compared with "same as last hour". Full details are in [results/EVALUATION.md](results/EVALUATION.md). Regenerate it with `python scripts/evaluate.py`.

## Run it on your laptop

```bash
pip install -r requirements.txt          # Python 3.12 or newer
python scripts/run_simulation.py         # one month, AI mode
python scripts/run_simulation.py --mode fixed --norms moefcc2017_metro
python scripts/evaluate.py               # compare all modes -> results/EVALUATION.md
python -m pytest -q                      # 10 tests
```

Your own data: `python scripts/run_simulation.py --data your_file.csv`. The format is in [data/sample_upload.csv](data/sample_upload.csv). Required columns are `flow, pH, BOD, COD, TSS, NH4`. The dashboard also accepts a CSV upload.

## Project layout

```
stp/                  the plant (Python package, runs on laptop AND in the browser)
  config.py           Indian norms, plant sizes, typical raw sewage
  water.py            the water "packet" + norms check
  influent.py         data fed into the plant (replaces sensors)
  units.py            physical treatment units (no decisions)
  ml.py               Isolation Forest detector + Ridge forecaster
  messages.py         message bus between agents
  agents.py           the 11 Mesa agents (all decisions)
  model.py            the Mesa model
  evaluation.py       honest comparison of modes and detectors
web/                  the dashboard (index.html, app.js, worker.js, bridge.py)
scripts/              run, evaluate, generate data, build the website
tests/                pytest tests
data/                 generated influent CSVs
results/              EVALUATION.md
.github/workflows/    auto-deploys the dashboard to GitHub Pages
```

## Honest limitations

- **The influent data is simulated** from typical Indian raw-sewage values (CPHEEO manual). It is not measured at a real plant.
- **Process formulas are simplified** first-order models, not a calibrated ASM1-type model.
- Faecal coliforms can't be measured online, so the disinfection agent uses an estimate with deliberate error. Phosphorus isn't modelled.
- It demonstrates **control logic and multi-agent coordination**. It is not a design tool for a real plant.

## Sources

- NGT order 30-04-2019 (OA 1069/2018) STP norms, as quoted in CPCB compliance reports to the NGT
- MoEFCC notification G.S.R. 1265(E), 13-10-2017
- CPHEEO *Manual on Sewerage and Sewage Treatment Systems* (2013): raw sewage characteristics
- Mesa: https://mesa.readthedocs.io · Pyodide: https://pyodide.org
