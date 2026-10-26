import numpy as np
import pandas as pd

from cwfc.predict import make_predictions


def make_cpu(n_ts: int = 40, n_vm: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [(t * 300, f"vm{v}", rng.uniform(0, 100)) for t in range(n_ts) for v in range(n_vm)]
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "avg_cpu"])


def test_predictions_follow_submission_contract():
    out = make_predictions(make_cpu())

    assert list(out.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(out) == 24
    assert out["timestamp"].tolist() == [t * 300 for t in range(16, 40)]
    assert out["aggregate_cpu_utilization"].between(0, 100).all()
    assert (out["active_vm_count"] >= 0).all()


def test_holdout_values_do_not_affect_predictions():
    cpu = make_cpu()
    changed = cpu.copy()
    in_holdout = changed["timestamp"] >= 16 * 300
    changed.loc[in_holdout, "avg_cpu"] = 99.0

    pd.testing.assert_frame_equal(make_predictions(cpu), make_predictions(changed))
