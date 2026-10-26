import numpy as np
import pandas as pd
import pytest

from cwfc.targets import build_targets, split_train_holdout


def make_cpu() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [0,      0,      0,      300,    300,    300],
        "vm_id":     ["a",    "b",    "a",    "a",    "b",    "c"],
        "avg_cpu":   [10.0,   20.0,   10.0,   30.0,   np.nan, 60.0],
    })


def test_build_targets_counts_each_vm_once_and_skips_missing():
    # Arrange
    cpu = make_cpu()

    # Act
    out = build_targets(cpu)

    # Assert
    assert out["timestamp"].tolist() == [0, 300]
    assert out["active_vm_count"].tolist() == [2, 2]
    assert out["aggregate_cpu_utilization"].tolist() == pytest.approx([15.0, 45.0])


def test_split_train_holdout_keeps_order_and_sizes():
    targets = pd.DataFrame({"timestamp": range(10), "y": range(10)})

    train, holdout = split_train_holdout(targets, holdout=3)

    assert train["timestamp"].tolist() == list(range(7))
    assert holdout["timestamp"].tolist() == [7, 8, 9]


def test_split_train_holdout_rejects_too_short_series():
    with pytest.raises(ValueError):
        split_train_holdout(pd.DataFrame({"timestamp": range(3)}), holdout=3)
