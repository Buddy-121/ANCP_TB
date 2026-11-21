"""Forecasting & predictive models with leakage-safe walk-forward validation.

Every model is benchmarked against a naive forecast. If a model cannot beat naive
out-of-sample, the UI says so – a model that loses to "tomorrow = today" should not
drive decisions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

METHODS = ["Naive (random walk)", "Linear trend", "ETS (damped trend)", "ARIMA(1,1,1)"]


def _fit_predict(train: pd.Series, h: int, method: str, alpha: float = 0.05) -> pd.DataFrame:
    """Return DataFrame [mean, lower, upper] for h steps ahead (positional index)."""
    y = train.astype(float).values
    n = len(y)
    if method == "Naive (random walk)":
        sd = np.std(np.diff(y), ddof=1) if n > 2 else 0
        z = 1.96
        steps = np.arange(1, h + 1)
        return pd.DataFrame({"mean": np.repeat(y[-1], h), "lower": y[-1] - z * sd * np.sqrt(steps),
                             "upper": y[-1] + z * sd * np.sqrt(steps)})
    if method == "Linear trend":
        import statsmodels.api as sm
        X = sm.add_constant(np.arange(n))
        m = sm.OLS(y, X).fit()
        Xf = sm.add_constant(np.arange(n, n + h), has_constant="add")
        pr = m.get_prediction(Xf).summary_frame(alpha=alpha)
        return pd.DataFrame({"mean": pr["mean"].values, "lower": pr["obs_ci_lower"].values,
                             "upper": pr["obs_ci_upper"].values})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "ETS (damped trend)":
            from statsmodels.tsa.exponential_smoothing.ets import ETSModel
            m = ETSModel(pd.Series(y), error="add", trend="add", damped_trend=True).fit(disp=False)
            pr = m.get_prediction(start=n, end=n + h - 1).summary_frame(alpha=alpha)
            return pd.DataFrame({"mean": pr["mean"].values, "lower": pr["pi_lower"].values,
                                 "upper": pr["pi_upper"].values})
        if method == "ARIMA(1,1,1)":
            from statsmodels.tsa.arima.model import ARIMA
            m = ARIMA(y, order=(1, 1, 1)).fit()
            fc = m.get_forecast(h)
            ci = fc.conf_int(alpha=alpha)
            return pd.DataFrame({"mean": fc.predicted_mean, "lower": ci[:, 0], "upper": ci[:, 1]})
    raise ValueError(method)


def walk_forward(series: pd.Series, method: str, n_test: int = 12, log: bool = False) -> dict:
    """Expanding-window one-step-ahead backtest. Each forecast uses only data before it."""
    s = series.dropna()
    t = np.log(s) if log else s
    n_test = min(n_test, max(1, len(s) // 4))
    preds, naive, actual, dates = [], [], [], []
    for i in range(len(s) - n_test, len(s)):
        train = t.iloc[:i]
        if len(train) < 8:
            continue
        p = _fit_predict(train, 1, method)["mean"].iloc[0]
        preds.append(np.exp(p) if log else p)
        naive.append(s.iloc[i - 1])
        actual.append(s.iloc[i])
        dates.append(s.index[i])
    if not preds:
        return {"error": "Not enough history for a backtest (need ≥ 8 training points)."}
    a, p, nv = np.array(actual), np.array(preds), np.array(naive)

    def m(err, a):
        return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "MAPE": float(np.mean(np.abs(err / a))) if np.all(a != 0) else np.nan}

    mm, mn = m(a - p, a), m(a - nv, a)
    return {"metrics": mm, "naive_metrics": mn,
            "beats_naive": mm["RMSE"] < mn["RMSE"],
            "theil_u": mm["RMSE"] / mn["RMSE"] if mn["RMSE"] else np.nan,
            "table": pd.DataFrame({"actual": a, "forecast": p, "naive": nv}, index=dates)}


def forecast(series: pd.Series, horizon: int, method: str, freq: str, log: bool = False) -> pd.DataFrame:
    s = series.dropna()
    t = np.log(s) if log else s
    fc = _fit_predict(t, horizon, method)
    if log:
        fc = np.exp(fc)
    idx = pd.date_range(s.index[-1], periods=horizon + 1, freq=freq)[1:]
    fc.index = idx
    return fc


# ------------------------------------------------------------------ ML return model
def ml_return_model(panel: pd.DataFrame, target: str, lags: int = 2, n_splits: int = 5) -> dict:
    """Predict next-period log return of `target` from lagged log returns of all columns.

    Leakage controls: features are shifted by ≥1 period; TimeSeriesSplit (train always
    precedes test); scaler fitted inside each training fold via a Pipeline.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    r = np.log(panel.where(panel > 0)).diff()
    feats = pd.concat({f"{c}_lag{k}": r[c].shift(k) for c in r.columns for k in range(1, lags + 1)}, axis=1)
    y = r[target]
    d = pd.concat([feats, y.rename("y")], axis=1).dropna()
    if len(d) < 40:
        return {"error": f"Only {len(d)} usable periods – need ≥ 40 for a meaningful ML backtest."}
    X, yy = d.drop(columns="y"), d["y"]
    models = {"Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
              "Random forest": RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=5,
                                                     random_state=42)}
    tss = TimeSeriesSplit(n_splits=n_splits)
    results, oos = {}, {}
    for name, mdl in models.items():
        preds, actual, base, idx = [], [], [], []
        for tr, te in tss.split(X):
            mdl.fit(X.iloc[tr], yy.iloc[tr])
            preds.extend(mdl.predict(X.iloc[te]))
            actual.extend(yy.iloc[te])
            base.extend([yy.iloc[tr].mean()] * len(te))  # baseline uses training mean only
            idx.extend(X.index[te])
        a, p, b = map(np.array, (actual, preds, base))
        ss_res, ss_base = np.sum((a - p) ** 2), np.sum((a - b) ** 2)
        results[name] = {"OOS R² vs train-mean": 1 - ss_res / ss_base,
                         "Directional accuracy": float(np.mean(np.sign(a) == np.sign(p))),
                         "Baseline directional (always up)": float(np.mean(a > 0)),
                         "RMSE": float(np.sqrt(np.mean((a - p) ** 2))), "Test periods": len(a)}
        oos[name] = pd.DataFrame({"actual": a, "predicted": p}, index=idx)
    rf = models["Random forest"].fit(X, yy)
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    latest = feats.dropna().iloc[[-1]]
    next_pred = {n: float(m.fit(X, yy).predict(latest)[0]) for n, m in models.items()}
    return {"results": pd.DataFrame(results).T, "oos": oos, "importance": importance,
            "next_period_pred": next_pred, "features_asof": latest.index[0], "n": len(d)}
