"""
Общая инженерия признаков для алгоритмов, работающих на OHLCV-данных.

Вынесена в отдельный модуль, чтобы ~20 разных ML-алгоритмов не дублировали
одну и ту же логику построения технических индикаторов и таргетов.
Все признаки на момент t используют только данные <= t (без утечки будущего).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Строит матрицу признаков + таргеты по OHLCV-данным одного инструмента.

    Возвращает DataFrame с исходными OHLCV + признаками + колонками:
        fwd_return   - будущая доходность close(t+horizon)/close(t) - 1 (для регрессии)
        fwd_direction - 1, если fwd_return > 0, иначе 0 (для классификации)

    Первые/последние строки с NaN (из-за окон индикаторов и горизонта) не удаляются
    здесь намеренно - вызывающий код сам решает, обрезать их до или после сплита
    на train/test, чтобы не терять данные не по делу.
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
    """Отбрасывает строки с NaN в фичах/таргете и возвращает (X, y)."""
    cols = feature_cols or FEATURE_COLUMNS
    cols = [c for c in cols if c in feat_df.columns]
    clean = feat_df.dropna(subset=cols + [target_col])
    return clean[cols], clean[target_col]


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Хронологический сплит без перемешивания (обязателен для временных рядов)."""
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]
