# Cloud Workload Forecasting

Forecasting near-term aggregate demand on a 10,000-VM subset of the
Microsoft Azure Public Dataset V2. The original assignment is in
[`project_info/ASSIGNMENT.md`](project_info/ASSIGNMENT.md).

## Quick start

Requires Python 3.11+.

### 1. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install (all systems)

```bash
python -m pip install -e ".[dev,notebook]"
```

### 3. Get the data

Download `data-v1.tar.gz` from the
[data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)
and extract it into `challenge_data/` (all systems):

```bash
tar -xzf data-v1.tar.gz -C challenge_data
```

Verify the checksums from inside `challenge_data/`:

| System | Command |
|---|---|
| macOS | `shasum -a 256 -c SHA256SUMS` |
| Linux | `sha256sum -c SHA256SUMS` |
| Windows | `Get-FileHash vm_cpu_readings.csv.gz` and compare with `SHA256SUMS` |

### 4. Generate predictions (all systems)

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This writes 24 rows (`timestamp,active_vm_count,aggregate_cpu_utilization`)
for the final two-hour holdout.

### 5. Run the tests (all systems)

```bash
python -m pytest -q
```

## Repository layout

```text
src/cwfc/
  targets.py      build the two target series; 110/24 train/holdout split
  validation.py   rolling-origin folds inside the training window
  baselines.py    persistence, moving average, seasonal-naive
  models.py       AR(1) (final model) and lag-feature regression
  metrics.py      MAE, RMSE, sMAPE, normalized MAE
  predict.py      CLI that writes predictions.csv
notebooks/
  01_data_audit.ipynb    data quality, distributions, segments, target series
  02_forecasting.ipynb   validation, model selection, holdout, errors, capacity policy
tests/                   unit tests, including a holdout-leakage test
reports/
  final_report.md        final report
  figures/               figures used in the report
project_info/            original assignment and dataset contract
```

Notebooks are run from the `notebooks/` folder with the `.venv` kernel.

## Main decisions

- **Targets.** `active_vm_count` is the number of distinct VMs with a valid
`avg_cpu` reading per timestamp; `aggregate_cpu_utilization` is the mean of
per-VM `avg_cpu`. Missing readings are dropped, and duplicate readings of one
VM at one timestamp are averaged first, so each VM counts once in both targets.
- **`active_vm_count` is constant (10,000).** The release keeps only complete
134-point VM traces, so there is no churn inside the window. AR(1) reduces to
   the mean here, which forecasts it exactly.
- **Validation.** Rolling-origin evaluation inside the first 110 points:
  horizon 24, minimum 48 points of history, step 6, anchored so that the last
  fold ends at point 109 (7 folds). The holdout was used once, after model
  selection was committed.
- **Model.** AR(1): `forecast(h) = mu + phi^h * (last - mu)`, fitted by least
  squares with `phi` clipped to [0, 1]. It combines persistence (short horizon)
  and the mean (long horizon), which were each best in different validation
  folds.
- **No leakage.** `predict.py` reads only the holdout *timestamps*;
  `tests/test_predict.py` checks that overwriting all holdout values leaves the
  predictions unchanged.

## Results

`aggregate_cpu_utilization`, MAE in percentage points (pp) of CPU:

| method | validation MAE (7 folds) | holdout MAE | holdout RMSE |
|---|---|---|---|
| **AR(1)** | **0.201** | 0.153 | **0.183** |
| persistence | 0.215 | **0.147** | 0.185 |
| train mean | 0.209 | 0.175 | 0.211 |
| lag regression* | 0.221 | 0.096 | 0.121 |

\* Added after the holdout evaluation as a comparison only. It is the worst on
validation but the best on the single holdout window; switching to it would mean
selecting on the test set, so AR(1) stays the final model.

AR(1) has the best validation MAE and is never the worst in any fold.
On the holdout, persistence is 4% better on MAE because the window starts in a
low, calm regime; AR(1) is slightly better on RMSE and errs on the side of
over-forecasting, which is the safer direction for capacity.

**Main failure mode:** AR(1) under-forecasts peaks (above the training 90th
percentile) by ~0.4 pp.

**Capacity recommendation:** provision the AR(1) forecast **+ 0.44 pp**
(≈ +4.7% headroom), the 95th percentile of validation under-forecast errors.
This margin covered all 24 holdout points.

Details are in the notebooks and in the [final report](reports/final_report.md).

## Data source

Microsoft Azure Public Dataset V2 —
https://github.com/Azure/AzurePublicDataset.
The data is not committed to this repository.

## Limitations and next steps

- The window is 11 h, so daily seasonality cannot be learned; a longer trace is needed.
- AR(1) under-forecasts peaks; a horizon-dependent hybrid (persistence for the first ~30 min) looked better on validation but was not applied, to keep the holdout clean.
- Capacity should ultimately be planned on `max_cpu` and per segment (Interactive VMs).

See section 9 of the [final report](reports/final_report.md) for the full discussion.
