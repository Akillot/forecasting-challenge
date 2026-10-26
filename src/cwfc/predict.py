"""Generate predictions.csv for the final 24-step holdout.

Usage: python -m cwfc.predict --data challenge_data --output predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cwfc.models import ar1_forecast
from cwfc.targets import build_targets, split_train_holdout

HORIZON = 24
TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]


def make_predictions(cpu: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    targets = build_targets(cpu)
    train, holdout = split_train_holdout(targets, horizon)
    template = holdout[["timestamp"]].reset_index(drop=True)  # holdout targets are never read
    out = template.copy()
    for target in TARGETS:
        out[target] = ar1_forecast(train, template, target).to_numpy()
    out["active_vm_count"] = out["active_vm_count"].round().clip(lower=0).astype(int)
    out["aggregate_cpu_utilization"] = out["aggregate_cpu_utilization"].clip(0, 100)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args(argv)

    cpu = pd.read_csv(args.data / "vm_cpu_readings.csv.gz")
    predictions = make_predictions(cpu)
    predictions.to_csv(args.output, index=False)
    print(f"wrote {len(predictions)} rows to {args.output}")


if __name__ == "__main__":
    main()
