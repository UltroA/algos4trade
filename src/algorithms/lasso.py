"""
Lasso - факторная модель, отбор признаков через L1-разреженность.

частный случай Elastic Net с l1_ratio=1 - чистая L1-регуляризация линейной регрессии на будущую доходность
(fwd_return). Сильнее обнуляет веса нерелевантных признаков, чем Elastic Net,
что даёт более разреженную и интерпретируемую модель ценой чуть большей
неустойчивости коэффициентов при коррелированных признаках.

Реалистичный уровень (см. docs/Алгоритмы.md): низкая точность, высокая
стабильность отобранного (разреженного) набора факторов. Только линейные
зависимости.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split, make_features, split_features_target


class LassoFactorModel(SingleAssetAlgorithm):
    name = "Lasso Factor Model"
    category = AlgorithmCategory.LINEAR_FACTOR
    description = (
        "Линейная факторная модель (Lasso) прогнозирует будущую доходность как разреженную "
        "комбинацию технических признаков за счёт L1-регуляризации."
    )

    def __init__(
        self,
        horizon: int = 1,
        alpha: float = 0.001,
        signal_strength: float = 20.0,
        **kwargs,
    ):
        super().__init__(horizon=horizon, alpha=alpha, signal_strength=signal_strength, **kwargs)
        self.horizon = horizon
        self.signal_strength = signal_strength
        self.scaler = StandardScaler()
        self.model = Lasso(
            alpha=alpha,
            max_iter=10_000,
            **{k: v for k, v in kwargs.items() if k not in {"horizon", "alpha", "signal_strength"}},
        )

    def fit(self, train_data: pd.DataFrame) -> "LassoFactorModel":
        feat_df = make_features(train_data, horizon=self.horizon)
        X, y = split_features_target(feat_df, target_col="fwd_return")
        if X.empty or y.empty:
            self.is_fitted = False
            return self
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(0.0, index=data.index)

        feat_df = make_features(data, horizon=self.horizon)
        X, _ = split_features_target(feat_df, target_col="fwd_return")
        if X.empty:
            return pd.Series(0.0, index=data.index)

        X_scaled = self.scaler.transform(X)
        predicted_return = self.model.predict(X_scaled)
        raw_signal = np.tanh(predicted_return * self.signal_strength)
        signal = pd.Series(raw_signal, index=X.index).clip(-1.0, 1.0)
        return signal.reindex(data.index).fillna(0.0)


if __name__ == "__main__":
    # Быстрая самопроверка на синтетических данных (без сети).
    rng = np.random.default_rng(0)
    n = 400
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    dummy = pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )
    train_df, test_df = chronological_split(dummy, 0.7)
    algo = LassoFactorModel()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
