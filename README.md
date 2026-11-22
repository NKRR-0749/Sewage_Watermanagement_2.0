# Autonomous Sewage Treatment Plant — Mesa Multi-Agent System

**Autonomous Control of a Sewage Treatment Plant Using Mesa Multi-Agent Systems with ML-Based Anomaly Detection**: sewage water quality monitoring, anomaly detection and autonomous treatment for an Indian sewage treatment plant.

▶ **Live dashboard:** https://nkrr-0749.github.io/Sewage_Watermanagement_2.0/
(opens in any browser, phone included. The Python and Mesa code runs inside the page, so first load takes about 15–40 s.)

> New to the project? Read **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** first. It explains every part in plain language.

---

## What it does

A 10 MLD (10 million litres/day) activated-sludge plant is simulated hour by hour. **Eleven Mesa agents** run it:

- Real-world-style **influent data is fed in instead of sensors**: daily flow peaks, monsoon storms, and shock events like industrial discharge, acid dumps and septage dumping.
- A **Monitoring agent** detects shocks with **rules + ML (Isolation Forest + robust z-scores)** and names the cause.
- **Control agents** act: divert toxic water to a holding tank, dose lime or acid, run the blowers to an oxygen target (using an **ML load forecast**), adjust nitrate recycle, decide filter backwashes, set the chlorine dose, and send failing water back for re-treatment.
- A **Compliance agent** checks every hour against Indian norms: **NGT 2019** (BOD ≤ 10, COD ≤ 50, TSS ≤ 20, TN ≤ 10 mg/L, FC ≤ 230 MPN/100 mL, pH 5.5–9) or **MoEFCC 2017**.

The treatment **can fail**. Bacteria get poisoned, blowers max out, tanks overflow, filters clog. The results are therefore measured, not guaranteed.

## Results (simulated data, NGT 2019 norms)

Four ways of running the same plant:
- **AI:** the agents with ML and rule detection plus the ML forecast.
- **Rules only:** the agents without ML.
- **Operator:** a *well-run* conventional plant. It has DO feedback control on the blowers and a
  recycle set to its design value. A human operator diverts shocks, doses pH and switches storm
  pumping **3 hours** after an alarm.
- **Fixed:** fixed settings, and nobody acts.

| Control mode | Demo month | 10 random months (mean ± sd) | Worst month |
|---|---|---|---|
| **AI** (ML + rules) | **95.8 %** | **94.9 ± 3.6 %** | 88.5 % |
| Rules only (no ML) | 95.6 % | 94.7 ± 3.5 % | 88.5 % |
| Operator (well-run conventional, 3 h response) | 45.3 % | 55.8 ± 6.7 % | 42.5 % |
| Fixed settings | 15.8 % | 24.0 ± 4.5 % | 18.8 % |

Each random month has new random sewage and a new random schedule of 9 shocks. In paired
comparisons on the same months, AI beats the operator in **10 of 10** months by 39 ± 5 points.
AI beats rules-only by only 0.2 ± 0.3 points.

**Why the operator loses:** a toxic shock that reaches the bacteria before anyone acts poisons
them, and the nitrifying bacteria take days to recover, so total nitrogen keeps failing. With an
instant (0 h) response the operator gets 91.4 %; with 1 h it gets 67.6 %. **Speed of response is
the main value of the agents.**

**Where the ML earns its place: shock types it was never tuned for** (solvent discharge, ammonia
spike, slow 30-hour drift; 5 months):

| Detector | Ammonia spike | Slow drift | Solvent | Overall recall |
|---|---|---|---|---|
| Rules | 30 % | 0 % | 100 % | 0.20 |
| ML | **100 %** | **63 %** | 100 % | **0.75** |

Compliance on these months is the same with and without ML (91.0 % vs 91.1 %), because the plant's
feedback controls absorb these loads. ML gives *awareness*, not extra treatment.

Detectors on the demo month, graded against the true labels:

| Detector | Precision | Recall | False alarms (668 normal hours) |
|---|---|---|---|
| Rules | 1.00 | 0.90 | 0 |
| ML (Isolation Forest + robust z-score) | 0.88 | **1.00** | 7 |

An Isolation Forest alone missed every acid and alkaline dump, because it can't tell "far outside
the normal range" from "at the edge of it". Adding a learned per-parameter z-score fixed that.

Two design choices were *tested*, not assumed:
- **An alarm with no identified cause is reported, not diverted.** Diverting on uncertain alarms
  filled the holding tank with ordinary sewage. It lowered compliance from 95.4 % to 94.1 % on known
  shocks, and from 90.0 % to 86.6 % on unseen ones.
- **The ML forecast is used only where it beat "same as last hour" on held-out training data.**
  It cuts flow-prediction error by 37 %.

Full details: [results/EVALUATION.md](results/EVALUATION.md). Regenerate with `python scripts/evaluate.py` (~2 min).

## Real data: UCI Water Treatment Plant dataset

Besides the simulated month, the plant runs on **real measured sewage**: the public
[UCI Water Treatment Plant dataset](https://archive.ics.uci.edu/dataset/106/water+treatment+plant)
(Poch, 1993, doi:10.24432/C5FS4C, CC BY 4.0). It holds 520 days of daily inflow, pH, BOD, COD and
suspended solids from a working municipal plant in Manresa, Spain. Details are in
[data/real/SOURCE.md](data/real/SOURCE.md) and `stp/real_data.py`.

- **Train/test split:** the first 150 days are used only to calibrate the plant. The Isolation Forest
  and forecaster are retrained on them, and the detection-rule limits are tuned to what is normal at
  that plant. All results are on the other 370 days.
- **What is real and what is assumed:** day-to-day flow and concentrations are real. The hourly
  shape within each day, ammonia (estimated from COD), temperature and coliforms are assumed.
- **Validation:** the dataset also records what the real plant discharged. Our simulated effluent
  is compared with it on the same days. Those numbers are never fed into the model.

Results on the 370 real test days (NGT 2019 norms):

| Setup | Hours meeting norms | Alarms (% of hours) | False alarms (injected-shock test) |
|---|---|---|---|
| AI, site-calibrated | 68.5 % | 4.3 % | 358 |
| Rules only, site-calibrated | 68.5 % | 2.7 % | 224 |
| Operator (well-run conventional, 3 h response) | 63.1 % | 2.7 % | – |
| Fixed settings | 54.9 % | – | – |
| AI, **not** calibrated (simulated-data ML, Indian rule limits) | 70.0 % | 90.9 % | 7,631 |

| Measured on the same real days | Real plant effluent | Our simulated effluent |
|---|---|---|
| BOD (mg/L) | 17.8 | 3.9 |
| COD (mg/L) | 82.0 | 40.2 |
| TSS (mg/L) | 19.3 | 4.7 |

What the real data shows:
- **COD is the hard parameter.** Manresa sewage carries more non-biodegradable COD than typical
  Indian domestic sewage, so NGT's 50 mg/L COD limit causes most of the failing hours (about 32 % of hours fail in total).
- **Site calibration is essential.** Without it, 91 % of real hours raise an alarm. With it, false
  alarms drop by 95 %. The uncalibrated plant scores slightly higher (70.0 %) only because it
  diverts so much ordinary sewage.
- **On real data the ML adds alarms, not compliance.** Rules-only does as well as AI mode. The ML
  forecaster still beats "same as last hour".
- **Our effluent is cleaner than the real plant's.** Our design adds a tertiary filter and nitrogen
  removal, and our process formulas are simplified, so these numbers are probably optimistic.

Full report: [results/EVALUATION_REAL.md](results/EVALUATION_REAL.md). On the dashboard, choose
**Data → Real plant data (UCI, Spain)**. The dashboard uses 60 test days to stay quick.

## Use the dashboard

Open the [live dashboard](https://nkrr-0749.github.io/Sewage_Watermanagement_2.0/) and wait for Python,
Mesa, and the ML models to load in your browser. The first visit usually takes 15–40 seconds.

- Use **Run**, **+1 hour**, or **+1 day** to advance the plant. **Speed** controls how many hours each
  run step advances.
- Choose a **Control** mode or **Norms** standard to compare operating strategies; changing either
  resets the simulation.
- Choose **Data** to use the simulated demo, the 60-day real-data demo, or upload your own CSV.
  The CSV needs `flow, pH, BOD, COD, TSS, NH4`; see [data/sample_upload.csv](data/sample_upload.csv).
- **Inject (4 h)** adds the selected shock to the next four hours. **Run all 4 modes on this data**
  compares the control modes on the selected dataset.

## Run it on your laptop

```bash
pip install -r requirements.txt          # Python 3.12 or newer
python scripts/run_simulation.py         # one month, AI mode
python scripts/run_simulation.py --mode operator --norms moefcc2017_metro
python scripts/evaluate.py               # 4 modes, 10 random months, unseen shocks -> results/EVALUATION.md
python scripts/run_simulation.py --real  # REAL data (UCI Manresa plant), site-calibrated
python scripts/evaluate_real.py          # full real-data evaluation -> results/EVALUATION_REAL.md (~3 min)
python -m pytest -q                      # 19 tests
```

Your own data: `python scripts/run_simulation.py --data your_file.csv`. The format is in [data/sample_upload.csv](data/sample_upload.csv). Required columns are `flow, pH, BOD, COD, TSS, NH4`. The dashboard also accepts a CSV upload.

## Project layout

```
stp/                  the plant (Python package, runs on laptop AND in the browser)
  config.py           Indian norms, plant sizes, typical raw sewage
  water.py            the water "packet" + norms check
  influent.py         data fed into the plant (replaces sensors)
  real_data.py        loads the REAL UCI dataset and converts it to hourly influent
  units.py            physical treatment units (no decisions)
  ml.py               anomaly detector (Isolation Forest + z-scores), Ridge forecaster, rules + site calibration
  messages.py         message bus between agents
  agents.py           the 11 Mesa agents (all decisions)
  model.py            the Mesa model
  evaluation.py       honest comparison of modes and detectors
  evaluation_real.py  the same on real data + comparison with the real plant's effluent
web/                  the dashboard (index.html, app.js, worker.js, bridge.py)
scripts/              run, evaluate, generate data, build the website
tests/                pytest tests
data/                 generated influent CSVs; data/real/ = the UCI real dataset + SOURCE.md
results/              EVALUATION.md, EVALUATION_REAL.md, summary.json (start screen)
.github/workflows/    auto-deploys the dashboard to GitHub Pages
```

## Honest limitations

- **The demo influent is simulated** from typical Indian raw-sewage values (CPHEEO manual). The real dataset is from a Spanish plant and is **daily**, so the hourly shape within each day is assumed. Ammonia is not in it and is estimated from COD. No public hourly Indian STP dataset was available.
- The real data has **no shock labels**, so detectors are graded on shocks injected on top of it.
- **Process formulas are simplified** first-order models, not a calibrated ASM1-type model.
- Faecal coliforms can't be measured online, so the disinfection agent uses an estimate with deliberate error. Phosphorus isn't modelled.
- It demonstrates **control logic and multi-agent coordination**. It is not a design tool for a real plant.

## Sources

- NGT order 30-04-2019 (OA 1069/2018) STP norms, as quoted in CPCB compliance reports to the NGT
- MoEFCC notification G.S.R. 1265(E), 13-10-2017
- CPHEEO *Manual on Sewerage and Sewage Treatment Systems* (2013): raw sewage characteristics
- Poch, M. (1993). *Water Treatment Plant* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5FS4C
- Mesa: https://mesa.readthedocs.io · Pyodide: https://pyodide.org
