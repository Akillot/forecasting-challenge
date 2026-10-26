from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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


LAGS = (1, 2, 3)
WINDOWS = (6, 12)


def _features(history: np.ndarray) -> list[float]:
    """Features known at the forecast origin: recent lags and rolling means."""
    return [history[-lag] for lag in LAGS] + [history[-w:].mean() for w in WINDOWS]


def lag_regression_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Linear regression on lags and rolling means, applied recursively over the horizon."""
    y = train[target].to_numpy(dtype=float)
    start = max(max(LAGS), max(WINDOWS))
    if len(y) <= start + 1:
        raise ValueError("not enough history for lag regression")
    X = [_features(y[:t]) for t in range(start, len(y))]
    model = LinearRegression().fit(X, y[start:])  
    history = list(y)
    preds = []
    for _ in range(len(template)):
        nxt = float(model.predict([_features(np.asarray(history))])[0])
        preds.append(nxt)
        history.append(nxt)
    return pd.Series(preds, index=template.index, dtype=float)
