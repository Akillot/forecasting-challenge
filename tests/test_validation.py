import pandas as pd
import pytest
from cwfc.validation import rolling_origin_folds, rolling_origin_splits


def test_rolling_origin_splits_are_temporal():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")})
    splits = rolling_origin_splits(frame, horizon=4, min_train_size=8, step=4)
    assert len(splits) == 3
    assert splits[0].train_end < splits[0].validation_start
    assert splits[0].validation_end < splits[1].validation_start


def test_rolling_origin_folds_end_exactly_at_series_end():
    origins = rolling_origin_folds(110, horizon=24, min_train=48, step=6)
    assert origins == [50, 56, 62, 68, 74, 80, 86]
    assert origins[-1] + 24 == 110


def test_rolling_origin_folds_rejects_too_short_series():
    with pytest.raises(ValueError):
        rolling_origin_folds(60, horizon=24, min_train=48)
