"""
Shared feature engineering for algorithms operating on OHLCV data.

Extracted into a separate module so that ~20 different ML algorithms don't
duplicate the same technical-indicator and target construction logic.
All features at time t use only data <= t (no look-ahead leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Builds the feature matrix + targets from a single instrument's OHLCV data.

    Returns a DataFrame with the original OHLCV + features + columns:
        fwd_return   - future return close(t+horizon)/close(t) - 1 (for regression)
        fwd_direction - 1 if fwd_return > 0, else 0 (for classification)

    The leading/trailing rows with NaN (from indicator windows and the horizon)
    are deliberately not dropped here - the calling code decides whether to
    trim them before or after the train/test split, so as not to lose data
    unnecessarily.
    """
    out = df.copy()
    close = out["close"]

    for window in (5, 10, 20, 60):
        out[f"return_{window}d"] = close.pct_change(window)
        out[f"sma_{window}"] = close.rolling(window).mean()
        out[f"sma_ratio_{window}"] = close / out[f"sma_{window}"] - 1.0
        out[f"volatility_{window}"] = close.pct_change().rolling(window).std()

    out["return_1d"] = close.pct_change(1)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    if "volume" in out.columns:
        out["volume_change_5d"] = out["volume"].pct_change(5)
        out["volume_zscore_20d"] = (
            out["volume"] - out["volume"].rolling(20).mean()
        ) / out["volume"].rolling(20).std()

    out["high_low_range"] = (out["high"] - out["low"]) / close

    out["fwd_return"] = close.shift(-horizon) / close - 1.0
    out["fwd_direction"] = (out["fwd_return"] > 0).astype(float)

    return out


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "sma_ratio_5",
    "sma_ratio_10",
    "sma_ratio_20",
    "sma_ratio_60",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_60",
    "rsi_14",
    "macd",
    "macd_signal",
    "volume_change_5d",
    "volume_zscore_20d",
    "high_low_range",
]


def split_features_target(
    feat_df: pd.DataFrame, target_col: str = "fwd_direction", feature_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Drops rows with NaN in the features/target and returns (X, y). Training-only:
    dropping rows with an unknown target is correct there, but the same drop silently
    discards the most recent row(s) at inference time - see select_features() below."""
    cols = feature_cols or FEATURE_COLUMNS
    cols = [c for c in cols if c in feat_df.columns]
    clean = feat_df.dropna(subset=cols + [target_col])
    return clean[cols], clean[target_col]


def select_features(feat_df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Drops rows with NaN in the features only and returns X, with no dependency on any
    target column. For inference (generate_signals), use this instead of
    split_features_target(): a regression target like fwd_return is genuinely NaN on the
    most recent row (the future outcome isn't known yet), so dropping on it there would
    always discard exactly the row live trading needs - the current tick's own signal.
    fwd_direction happens not to hit this (NaN > 0 evaluates to False, so it is never NaN),
    which is why this only ever surfaced as "always-flat" for the regression-target
    algorithms (elastic_net, lasso, gaussian_process, genetic_programming), not the
    classification-target ones."""
    cols = feature_cols or FEATURE_COLUMNS
    cols = [c for c in cols if c in feat_df.columns]
    return feat_df.dropna(subset=cols)[cols]


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split without shuffling (required for time series)."""
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]
