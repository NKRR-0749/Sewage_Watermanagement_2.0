# Real dataset: UCI Water Treatment Plant

**File:** `uci_water_treatment.data`

**Citation:** Poch, M. (1993). *Water Treatment Plant* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5FS4C. Licence: CC BY 4.0.

**Official page:** https://archive.ics.uci.edu/dataset/106/water+treatment+plant

## What it is

Daily sensor measurements from a real urban wastewater treatment plant in Manresa, a town of
about 100,000 people near Barcelona, Spain. The plant treats about 35,000 m³/day, mostly domestic
sewage plus some industrial wastewater. The 38 measured variables cover the plant inlet (`-E`),
after pretreatment (`-P`), the input to the secondary settler (`-D`) and the plant outlet (`-S`).

## Which copy this is

The official UCI file has 527 days and no header. This copy has a header row, a first column with
each day's **plant-state label** (`Normal_situation`, `Normal_situation_low_influent`,
`Normal_situation_over`) and 520 days (7 fewer than the official file). It comes from the public GitHub repository
https://github.com/ammahmoudi/water-treatment-plant (`Data.data`). Its summary statistics
closely match the official UCI description; for example, the inflow Q-E has a mean of about 37,200 m³/day
and a maximum of 60,081.

`stp/real_data.py` also reads the **official** UCI file. To use it, download `water-treatment.data`
from the UCI page and pass its path to `load_uci_daily()`.

## Columns the project uses

| UCI column | Meaning | Used as |
|---|---|---|
| Q-E | inflow, m³/day | plant influent flow (scaled to 10 MLD, spread over the hours of each day) |
| PH-E | inflow pH | influent pH |
| DBO-E | inflow BOD, mg/L | influent BOD |
| DQO-E | inflow COD, mg/L | influent COD (also used to estimate ammonia, which is not measured) |
| SS-E | inflow suspended solids, mg/L | influent TSS |
| PH-S, DBO-S, DQO-S, SS-S | the real plant's measured **output** | only compared with our simulated effluent, never fed in |
