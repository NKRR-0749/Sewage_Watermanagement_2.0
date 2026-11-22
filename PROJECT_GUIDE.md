# Project Guide: understand the whole project in 20 minutes

This file explains **what the project does, how every part works, where the AI is, and how to demo and defend it**. Read it top to bottom once. Each section stands on its own, so you can come back to any one later.

---

## 1. The project in one minute

We simulate a **sewage treatment plant (STP) for an Indian town** that runs itself.

- Every simulated **hour**, a new reading of the raw sewage arrives from a data file. That file plays the role of the sensors.
- **Eleven Mesa agents** look at the water, talk to each other, and make decisions: divert poison, add lime, run the blowers, add chlorine, re-treat bad water, and so on.
- At the end, the treated water is checked against **Indian discharge norms** (NGT 2019 or MoEFCC 2017).
- Two pieces of **machine learning** help: an **Isolation Forest** spots unusual sewage, and a **Ridge regression** forecasts the next hour's load.
- A **web dashboard** shows it all live. The *same Python code* runs inside the browser (Pyodide), so the dashboard displays the real logic, not a copy of it.

**Main result (one month, strict NGT norms):** the autonomous plant meets the norms **95.7 %** of hours. The same plant with fixed settings meets them only **15.8 %** of hours.

---

## 2. How a real sewage treatment plant works

Water flows left to right through these units. The dashboard draws them in the same order.

| # | Unit | What it does in real life | What can go wrong (and is simulated) |
|---|---|---|---|
| 1 | **Screen & grit** | Bar screens catch plastic and rags; the grit chamber settles sand | — |
| 2 | **Equalization tank** | A big tank that evens out the morning and evening peaks | Overflows in a storm and raw sewage escapes |
| 2b | **Holding tank** | Emergency tank for toxic or strong sewage | Fills up if shocks come too often |
| 3 | **Primary clarifier** | Settling tank: about 60 % of the solids sink | Removes less when flow is high |
| 4 | **Aeration tank** | Bacteria eat the organic pollution (BOD), turn ammonia into nitrate (needs oxygen), and an anoxic zone turns nitrate into harmless N₂ gas | Acid, alkali or factory chemicals **poison the bacteria**, and they take days to recover. Blowers have a maximum capacity. |
| 5 | **Secondary clarifier** | The bacteria (sludge) settle out | Sick sludge settles badly, so solids escape |
| 6 | **Filter** | Sand filter polishes out the last solids | Clogs and must be backwashed |
| 7 | **Chlorination** | Kills faecal coliform bacteria | Dirt and ammonia use up the chlorine first |
| 8 | **Outlet** | Treated water goes to the river | Fails the norms |

### The water quality numbers (glossary)

| Name | Meaning | Raw sewage | NGT 2019 limit |
|---|---|---|---|
| **BOD** | Biochemical Oxygen Demand: organic pollution bacteria can eat | ~250 mg/L | ≤ 10 mg/L |
| **COD** | Chemical Oxygen Demand: all pollution that can be oxidised, including chemicals bacteria can't eat | ~475 mg/L | ≤ 50 mg/L |
| **TSS** | Total Suspended Solids: particles floating in the water | ~300 mg/L | ≤ 20 mg/L |
| **NH₄** | Ammonia nitrogen (toxic to fish) | ~25 mg/L | part of TN |
| **TN** | Total Nitrogen = ammonia + nitrate | ~25 mg/L | ≤ 10 mg/L |
| **FC** | Faecal coliform bacteria (disease) | ~10 million per 100 mL | ≤ 230 MPN/100 mL |
| **pH** | Acidity: 7 is neutral | ~7.4 | 5.5 – 9.0 |
| **DO** | Dissolved oxygen in the aeration tank | target 2 mg/L | — |
| **MLD** | Million litres per day. 10 MLD ≈ the sewage of about 90,000 people | | |

---

## 3. What happens in ONE simulated hour

`model.step()` in `stp/model.py` runs the 11 agents **in the order the water flows**:

1. **InfluentFeedAgent** reads the next row of the data file: flow, pH, BOD, COD, TSS, NH₄, FC, temperature.
2. **ForecastAgent** predicts this hour's load *using only past hours* (it doesn't peek).
3. **MonitoringAgent** checks the raw sewage with the **rules** and the **Isolation Forest**. If something is wrong it names the cause ("acid dump", "storm inflow"…) and sends an **ALERT** message to the EqualizationAgent.
4. **PreliminaryAgent** runs screening and grit removal.
5. **EqualizationAgent** reads its inbox. On an ALERT for a toxic or strong shock it **diverts** the water into the holding tank. For a storm it runs the pumps at full speed to avoid overflow. It doses lime or acid if pH is off, and slowly feeds held water back when the bacteria are healthy.
6. **PrimaryAgent** runs the primary clarifier.
7. **AerationAgent** sets the blower power *before* the water arrives, from the forecast, aiming for DO = 2 mg/L. It then corrects using last hour's error. It also sets the nitrate recycle. If the bacteria get poisoned it messages the ComplianceAgent.
8. **SecondaryAgent** runs the secondary clarifier.
9. **FilterAgent** decides whether to backwash now (when clogged, preferably at a low-flow hour).
10. **DisinfectionAgent** calculates the chlorine dose needed.
11. **ComplianceAgent** checks the treated water against the norms. If it fails, it sends up to 30 % of the flow **back to the equalization tank** instead of the river, and records the result.

Then the Mesa `DataCollector` saves the hour, and the next hour begins.

---

## 4. The eleven agents

All are `mesa.Agent` subclasses in `stp/agents.py`.

| Agent | Smart decision ("ai"/"rules" modes) | Conventional plant ("fixed" mode) |
|---|---|---|
| InfluentFeed | reads the data (same in all modes) | same |
| Forecast | ML forecast (ai) or "same as last hour" (rules) | not used |
| Monitoring | rules (+ Isolation Forest in ai mode); names the cause | rules run, but nobody acts on the alarms |
| Preliminary | — (equipment only) | same |
| Equalization | divert shocks, storm mode, pH dosing, slow release | level control only |
| Primary | — (equipment only) | same |
| Aeration | feed-forward + feedback blower control; nitrate recycle 1–6× | blowers fixed at 60 %, recycle fixed at 3× |
| Secondary | — (equipment only) | same |
| Filter | backwash when clogged, at low flow | backwash on a timer at 02:00 |
| Disinfection | dose from measured solids/ammonia + estimated coliforms | 5 mg/L always |
| Compliance | check norms, recirculate failing water | check only |

**How they talk:** `stp/messages.py`. An agent calls `self.send("Equalization", "ALERT", "...")` and the receiver calls `self.inbox("ALERT")` on its turn. Every message appears in the dashboard's *Agent messages* panel.

---

## 5. Where the AI is, and what happens if you remove it

This is the question examiners ask most. The answer is in the numbers.

### AI piece 1: anomaly detector (`stp/ml.py`)
- **What:** two unsupervised checks that learn what *normal* sewage looks like:
  - an **Isolation Forest**, which catches unusual *combinations*, such as strong BOD with normal pH;
  - a **robust z-score per parameter**, which measures how many "typical deviations" a reading is
    from its normal median. It catches single-parameter shocks such as pH 3.
- **Why both:** an Isolation Forest scores a value far outside the training range no worse than the
  most extreme normal value. On its own, it caught **0 %** of the acid and alkaline dumps. With the
  z-score added, the ML catches **100 %**.
- **Trained on:** a separate month of normal sewage (`training_dataset()`, June). It **never sees** any test month.
- **Threshold:** each check is set so it raises false alarms on only 0.5 % of normal training hours.
- **Graded against:** the **true event labels** written into the data when it was generated. It is
  *not* graded against the rule detector's output; that mistake was in the old project.

| Shock (true label) | Rules caught | ML caught |
|---|---|---|
| Acid dump / alkaline dump | 100 % | 100 % |
| Food-processing waste (subtle) | **0 %** | **100 %** |
| Industrial, septage, storm | 100 % | 100 % |
| *Unseen:* ammonia spike | 30 % | **100 %** |
| *Unseen:* slow 30-hour drift | 0 % | **63 %** |

**What the plant does with an alarm:** a *named* toxic shock (industrial, acid, alkaline, septage)
is diverted to the holding tank. An alarm with **no identified cause** (usually ML-only) is reported
but **not diverted**. This was tested: diverting on uncertain alarms filled the holding tank with
ordinary sewage and lowered compliance (95.4 % → 94.1 %).

### AI piece 2: Ridge regression load forecaster (`stp/ml.py`)
- **What:** predicts the next hour's flow, BOD and ammonia from the last 24 hours and the time of day.
- **Used by:** the AerationAgent, to set the blowers *before* the load arrives.
- **Self-check:** it tests itself on the last quarter of its training data and is used only for a
  quantity where it beats "same as last hour". Otherwise the plant uses last hour's value.
- **Result:** flow error is **37 % lower** than "same as last hour". On the shock month, BOD and
  ammonia are slightly *worse* (3–5 %), even though they passed the self-check. Say this honestly.

### The four control modes
| Mode | What it is |
|---|---|
| `ai` | the agents + ML detection + ML forecast |
| `rules` | the agents, rule detection only, "same as last hour" forecast |
| `operator` | a **well-run conventional plant**: DO feedback control on the blowers, recycle at its design value (5×), and a human who diverts, doses pH and switches storm pumping **3 hours** after an alarm |
| `fixed` | fixed settings, and nobody acts |

The `operator` mode exists because comparing against `fixed` alone would be unfair: no real plant
fails 84 % of the time.

### What happens if you remove the AI?
Compare **ai** with **rules** over **10 random months** (`python scripts/evaluate.py`):
- Compliance: 94.9 ± 3.6 % vs 94.7 ± 3.5 %. The difference is about zero.
- Detection: recall 1.00 vs 0.88 on known shocks, and 0.75 vs 0.20 on **unseen** shock types.
  The cost is about 13 false alarms a month.
- **The biggest improvement comes from the agents acting fast**: well-run operator 55.8 % → agents 94.9 %.
  An operator who could respond instantly would get 91 %.

**One sentence for the viva:** *"The big jump in compliance comes from agents that respond within
the hour; a human 3 hours late lets toxic shocks poison the bacteria. The ML doesn't add
compliance, but it detects shock types no rule was written for, and we measured all of this over
10 random months against ground-truth labels."*

---

## 6. The data: "fed instead of sensors"

`stp/influent.py` creates the data; `data/*.csv` are copies of it.

- **Why simulated?** Live hourly sewage data from Indian plants is not publicly available. The values are built from published typical Indian raw sewage (CPHEEO manual): BOD 200–300, COD 400–600, TSS 200–400 mg/L, pH 6.8–8.0, nitrogen about 30 mg/L. **Always call it simulated.**
- **Realistic patterns:** low flow at 3 am, a big 8 am peak, an evening peak, Sunday changes, random variation.
- **Nine shock events in July** (monsoon):

| Day | Event | Effect |
|---|---|---|
| 3 | Industrial discharge (6 h) | +900 mg/L COD bacteria can't eat, pH 5.2 |
| 6 | Storm inflow (8 h) | 2.6× flow, diluted sewage |
| 9 | Acid dump (3 h) | pH 3.2 |
| 12 | Septage dumping (4 h) | BOD 1100, TSS 1800, NH₄ 140 |
| 15 | Food-processing waste (5 h) | 1.7× BOD (subtle) |
| 18 | Alkaline dump (4 h) | pH 10.8 |
| 21 | Long storm (14 h) | 2.6× flow |
| 24 | Industrial discharge (5 h) | as day 3 |
| 27 | Acid dump (3 h) | pH 3.2 |

- **Your own data:** any CSV with columns `flow, pH, BOD, COD, TSS, NH4` (optional: `timestamp, temperature, FC, event`). Upload it in the dashboard, or run `python scripts/run_simulation.py --data file.csv`. If you get real data from a local STP, use it this way.

---

### Real data (`stp/real_data.py`)

The plant also runs on the public **UCI Water Treatment Plant** dataset: 520 days of real daily
measurements from a municipal plant in Manresa, Spain. The steps:

1. **Read** the influent columns: flow, pH, BOD, COD and suspended solids. Fill the few missing
   days by interpolation.
2. **Convert daily to hourly.** Each day's value is placed at noon and the hours in between are
   interpolated. Flow is shaped with the normal daily pattern. *Day-to-day changes are real; the
   shape within a day is assumed.*
3. **Scale the flow** so that the median day equals our 10 MLD. Concentrations stay exactly as
   measured.
4. **Estimate ammonia** from COD, because it isn't in the dataset.
5. **Calibrate to the site** on the first 150 days. The ML retrains, and the rule limits rise to
   the plant's own normal levels (the COD/BOD limit goes from 2.6 to 3.65). Test on the other 370 days.
6. **Grade the detectors** by injecting labelled shocks on top of the real days, because the real
   data has no shock labels.
7. **Validate** by comparing our effluent with the real plant's measured effluent on the same days.

Results: `results/EVALUATION_REAL.md`. Run it with `python scripts/evaluate_real.py`.

## 7. The Indian norms (`stp/config.py`)

| Norm set | pH | BOD | COD | TSS | TN | FC |
|---|---|---|---|---|---|---|
| **NGT 2019** (default, strict) | 5.5–9.0 | 10 | 50 | 20 | 10 | 230 |
| MoEFCC 2017 – metro & capitals | 6.5–9.0 | 20 | – | 50 | – | 1000 |
| MoEFCC 2017 – other areas | 6.5–9.0 | 30 | – | 100 | – | 1000 |

(mg/L; FC in MPN/100 mL.) The MoEFCC notification of 13 Oct 2017 relaxed the earlier draft standards. The NGT order of 30 April 2019 overturned that relaxation and set stricter limits. Different states apply them unevenly, which is why the dashboard lets you switch.

---

## 8. The files: read them in this order

| File | Lines | What it is |
|---|---|---|
| `stp/config.py` | ~130 | **All the numbers**: norms, tank sizes, raw sewage. Start here. |
| `stp/water.py` | ~110 | The `Water` object (flow + quality), mixing two streams, the norms check |
| `stp/influent.py` | ~230 | Makes and loads the data; the `InfluentFeed` that hands out one hour at a time |
| `stp/units.py` | ~260 | The **physical** units and their limits. No decisions here. |
| `stp/ml.py` | ~170 | Isolation Forest, rule detector, event classifier, Ridge forecaster |
| `stp/messages.py` | ~60 | The message bus |
| `stp/agents.py` | ~450 | **The 11 Mesa agents: every decision is here** |
| `stp/model.py` | ~180 | The Mesa `Model`: creates agents, `step()`, DataCollector, summary |
| `stp/evaluation.py` | ~90 | Runs all modes and grades the detectors |
| `scripts/run_simulation.py` | | Command-line run |
| `scripts/evaluate.py` | | Writes `results/EVALUATION.md` |
| `scripts/generate_data.py` | | Writes `data/*.csv` |
| `scripts/build_site.py` | | Packs the website into `_site/` (GitHub does this automatically) |
| `web/bridge.py` | | The few functions the web page calls in Python |
| `web/worker.js` | | Starts Python (Pyodide) + Mesa inside the browser, in a background thread |
| `web/app.js` | | Draws the dashboard. **Displays only**; all numbers come from Python. |
| `web/index.html`, `style.css` | | Page layout and look |
| `tests/test_plant.py` | | 10 tests, one per claim (e.g. "treatment can fail") |
| `.github/workflows/pages.yml` | | On every push: run tests → build → publish the dashboard |

**Separation to remember:** `units.py` = *what the equipment does*; `agents.py` = *what the agents decide*; `ml.py` = *the AI*; `model.py` = *Mesa glue*.

---

## 9. How to run it

### On a laptop
```bash
pip install -r requirements.txt     # needs Python 3.12+
python scripts/run_simulation.py    # one month, AI mode, prints the summary
python scripts/evaluate.py          # the comparison tables
python -m pytest -q                 # all tests should pass
```

### Put it on GitHub (one-time)
1. Copy everything from this folder into your repo folder and push:
   ```bash
   git add .
   git commit -m "Autonomous STP with Mesa agents and dashboard"
   git push origin main
   ```
2. On GitHub: repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Open the **Actions** tab and wait for the green tick (about 2 minutes).
4. The dashboard is at `https://<owner>.github.io/<repo>/`. For this repo that's
   **https://nkrr-0749.github.io/Sewage_Watermanagement_2.0/**. If the repo is under another account, change the link at the top of README.md.

### Test the website on your laptop
```bash
python scripts/build_site.py
python -m http.server 8000 --directory _site
# open http://localhost:8000  (needs internet for Pyodide)
```

---

## 10. A 3-minute demo script (for the viva)

1. **Open the link** and wait for "Starting the plant" to finish. Say: *"This is Python and Mesa running inside the browser."*
2. Click **+1 day** twice. Point at the diagram: water flows left to right, numbers update each hour, and everything is green.
3. Click **Run**. On **3 July 09:00** the industrial discharge arrives. The Inlet turns red, the Monitoring agent raises an ALARM, the purple pipe shows the shock **diverted to the holding tank**, and the Agent messages panel shows the conversation. The outlet stays green.
4. Pause. Pick **Inject shock → Acid dump** and press **Inject**, then **+1 hour**: it's caught live.
5. Switch **Control → Well-run conventional plant** (this resets), click **+1 day** 3–4 times. On day 3 the operator only reacts after 3 hours. By then the bacteria are poisoned, and nitrogen fails for days afterwards.
6. Back in AI mode, inject **Ammonia spike (unseen type)**. No rule fires, but the ML raises the alarm.
7. Pick **Data → Real plant data (UCI, Spain)** to show it running on real measured sewage.
8. Scroll down and click **Evaluate**: four modes compared, and each detector graded against the true labels.

---

## 11. Likely viva questions, with honest answers

**Q: Why a multi-agent system?**
Each unit has its own local decision (blowers, chlorine, backwash), and they must coordinate through messages: the monitor warns the equalization tank, and compliance asks for recirculation. Mesa gives a standard structure (Agent, Model, step, DataCollector) for that.

**Q: Is the data real?**
The demo month is simulated from published typical Indian sewage values. The plant also runs on a
real public dataset (UCI Water Treatment Plant, Manresa, Spain); see the real-data questions below.

**Q: Can the treatment fail?**
Yes. Under fixed settings it fails 606 of 720 hours. Even autonomous, it fails 31 hours, mostly during storms when the tank overflows. There's a test for this: `test_treatment_can_fail`.

**Q: Why Isolation Forest?**
It is unsupervised, and we don't have labelled shocks from real plants. It learns "normal" from normal data only. Weakness: it misses single-feature shocks like pH, which is why we combine it with rules.

**Q: How did you evaluate the detectors?**
Against the true labels stored in the data when it was generated, on a month the model never trained on. We report precision, recall and false alarms per detector and per event type.

**Q: Isn't the conventional baseline unfair?**
That's why there are two baselines. `fixed` is a plant nobody runs. `operator` is a well-run
conventional plant with DO control, the correct recycle, and a human who acts 3 hours after an
alarm. Section 5 of EVALUATION.md shows the result for every delay from 0 to 6 hours.

**Q: Is it just one lucky month?**
No. There are 10 random months, each with new sewage and a new shock schedule. Results are given as mean ± standard deviation, and as paired month-by-month comparisons.

**Q: What if a shock happens that you never designed for?**
Section 3 of EVALUATION.md uses three shock types that were never used to tune anything. Rules
catch 20 % of those hours; the ML catches 75 %. You can also inject them live from the dashboard.

**Q: What does the AI change?**
Honest version: fast agent control gives the large gain. Over 10 months, compliance with and without ML is the same within 0.2 points. The ML adds detection of subtle and never-seen shocks, and a better flow forecast.

**Q: How does the aeration control work?**
Feed-forward: forecast the next hour's oxygen demand and set the blowers for DO = 2 mg/L. Feedback: correct by last hour's DO error (an integral term).

**Q: Why does TN fail in the conventional plant?**
Blowers fixed at 60 % and recycle fixed at 3×. At night it over-aerates, and oxygen carried into the anoxic zone stops denitrification. In the morning it under-aerates and ammonia slips through.

**Q: Did you use any real dataset?**
Yes: the UCI Water Treatment Plant dataset (Poch, 1993). It holds 520 days of real measured inflow
from a municipal plant in Manresa, Spain. The first 150 days calibrate the ML and the rule limits;
the other 370 days are the test. Its own measured effluent is used to check our output.

**Q: Why not Indian data?**
We could not find a public, downloadable influent dataset for an Indian STP. The code accepts any CSV, so a local STP's data can be plugged in.

**Q: The data is daily but your plant runs hourly. Isn't that a problem?**
The day-to-day values are real. Only the shape within each day is assumed, using a standard
diurnal pattern. We state this, and it is why the forecast results on real data are partly
optimistic.

**Q: What did the real data teach you?**
Thresholds and models tuned for one place don't transfer. Uncalibrated, about 70 % of real hours
raised alarms. After calibrating on the plant's own normal days, false alarms fell by more than
90 %. COD is the limiting parameter there, because of the non-biodegradable industrial COD.

**Q: What are the limitations?**
Simulated data; simplified first-order process formulas (not ASM1); coliforms estimated, not measured; phosphorus not modelled; no sludge handling.

**Q: How does Python run in a browser?**
Pyodide is CPython compiled to WebAssembly. `web/worker.js` loads it, installs Mesa, unpacks our `stp` package and calls `bridge.py`. It runs in a Web Worker so the page doesn't freeze.

---

## 12. Limitations (put these in your report)

1. **Demo influent is simulated.** The real dataset (UCI, Manresa, Spain) is daily, not Indian, and has no ammonia or shock labels. The hourly shape and ammonia are assumed, and detection on it is graded with injected shocks.
2. **Simplified process models**: first-order kinetics, quasi-steady oxygen, a simple clarifier formula. Good for showing control behaviour; not for designing a real plant.
3. **Faecal coliform** cannot be measured online, so the disinfection agent uses an estimate with deliberate random error.
4. **Not modelled:** phosphorus, sludge handling and wasting, energy for pumps, equipment breakdowns.
5. **Simulated test months** (10 random months + 5 with unseen shock types). They are still generated by our own code, so they share its assumptions.

## 13. Ideas for future work
- Feed real hourly data from a local Indian STP (the CSV upload and site calibration are ready for it).
- Replace the first-order formulas with the standard ASM1 activated-sludge model.
- Reinforcement learning for the blower agent, compared against the current feed-forward controller.
- Add a sludge-handling agent and pump energy.
