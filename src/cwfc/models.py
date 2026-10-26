from __future__ import annotations

import numpy as np
import pandas as pd


def fit_ar1(values) -> tuple[float, float]:
    """Fit x[t] - mu = phi * (x[t-1] - mu) by least squares; phi is clipped to [0,1]."""
    x = np.asarray(values, dtype=float)
    if len(x) < 3:
        raise ValueError("need at least 3 points to fit AR(1)")
    mu = float(x.mean())
    d = x - mu
    denom = float(np.sum(d[:-1] ** 2))
    phi = 0.0 if denom == 0 else float(np.sum(d[1:] * d[:-1]) / denom)
    return mu, float(np.clip(phi, 0.0, 1.0))


def ar1_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    mu, phi = fit_ar1(train[target])
    last = float(train[target].iloc[-1])
    steps = np.arange(1, len(template) + 1)
    return pd.Series(mu + phi ** steps * (last - mu), index=template.index, dtype=float)
