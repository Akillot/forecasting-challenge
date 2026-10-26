import pandas as pd
import pytest

from cwfc.models import ar1_forecast, fit_ar1


def test_fit_ar1_constant_series_gives_mean_and_zero_phi():
    mu, phi = fit_ar1([5.0] * 10)
    assert mu == 5.0
    assert phi == 0.0


def test_fit_ar1_clips_negative_phi_to_zero():
    mu, phi = fit_ar1([1.0, -1.0] * 5)
    assert phi == 0.0


def test_ar1_forecast_starts_near_last_value_and_decays_to_mean():
    train = pd.DataFrame({"y": [1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5, 3.5]})
    template = pd.DataFrame(index=range(24))

    pred = ar1_forecast(train, template, "y")

    mu, phi = fit_ar1(train["y"])
    assert pred.iloc[0] == pytest.approx(mu + phi * (3.5 - mu))
    assert abs(pred.iloc[-1] - mu) < abs(pred.iloc[0] - mu)
