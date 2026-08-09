"""
Фильтр Калмана - динамический хедж-коэффициент в парном трейдинге.

Табличная строка #7. Для пары активов (A, B) фильтр Калмана онлайн оценивает
линейную зависимость close_A(t) = beta(t) * close_B(t) + alpha(t), где beta(t)
плавно меняется во времени (в отличие от статической OLS-регрессии). Спред
z(t) = close_A(t) - beta(t) * close_B(t) торгуется на возврат к среднему:
шорт спреда при z далеко выше своего скользящего среднего, лонг - при z далеко ниже.

Главная слабость (см. таблицу): разрыв коинтеграции ведёт к большому убытку -
здесь это не защищено отдельным фильтром, только ограничением плеча через
клиппинг сигнала в [-1, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.base import AlgorithmCategory, MultiAssetAlgorithm


class KalmanFilterPairsTrading(MultiAssetAlgorithm):
    name = "Kalman Filter Pairs Trading"
    category = AlgorithmCategory.PAIRS_STAT_ARB
    description = (
        "Онлайн-оценка динамического хедж-коэффициента между двумя активами фильтром Калмана "
        "и торговля возвратом спреда к среднему (mean-reversion статистического арбитража)."
    )

    def __init__(
        self,
        asset_a: str | None = None,
        asset_b: str | None = None,
        delta: float = 1e-4,
        observation_cov: float = 1e-3,
        zscore_window: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            asset_a=asset_a, asset_b=asset_b, delta=delta, observation_cov=observation_cov,
            zscore_window=zscore_window, entry_z=entry_z, exit_z=exit_z, **kwargs,
        )
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.delta = delta
        self.observation_cov = observation_cov
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        # Состояние фильтра Калмана: theta = [beta, alpha], P - ковариация оценки.
        self._theta = np.array([1.0, 0.0])
        self._P = np.eye(2)

    def _pick_pair(self, data: dict[str, pd.DataFrame]) -> tuple[str, str]:
        tickers = list(data.keys())
        a = self.asset_a or tickers[0]
        b = self.asset_b or tickers[1]
        return a, b

    def _kalman_beta_series(self, price_a: pd.Series, price_b: pd.Series) -> pd.Series:
        """Прогоняет фильтр Калмана вдоль ряда, возвращает beta(t) (без забегания вперёд)."""
        common_index = price_a.index.intersection(price_b.index)
        price_a = price_a.loc[common_index]
        price_b = price_b.loc[common_index]
        n = len(price_a)
        betas = np.empty(n)
        Q = self.delta * np.eye(2)  # шумовая ковариация процесса (плавный дрейф beta, alpha)
        R = self.observation_cov     # шум наблюдения

        theta = self._theta.copy()
        P = self._P.copy()

        for i in range(n):
            H = np.array([price_b.iloc[i], 1.0])  # observation model: price_a = beta*price_b + alpha
            # predict
            P = P + Q
            # update
            y_pred = H @ theta
            residual = price_a.iloc[i] - y_pred
            S = H @ P @ H.T + R
            K = (P @ H) / S
            theta = theta + K * residual
            P = P - np.outer(K, H) @ P
            betas[i] = theta[0]

        self._theta, self._P = theta, P
        return pd.Series(betas, index=price_a.index)

    def fit(self, train_data: dict[str, pd.DataFrame]) -> "KalmanFilterPairsTrading":
        a, b = self._pick_pair(train_data)
        self.asset_a, self.asset_b = a, b
        price_a, price_b = train_data[a]["close"], train_data[b]["close"]
        self._kalman_beta_series(price_a, price_b)  # прогреваем состояние фильтра на train
        self.is_fitted = True
        return self

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        a, b = self.asset_a, self.asset_b
        if not self.is_fitted or a not in data or b not in data:
            return {t: pd.Series(0.0, index=df.index) for t, df in data.items()}

        price_a, price_b = data[a]["close"], data[b]["close"]
        beta = self._kalman_beta_series(price_a, price_b)
        spread = price_a - beta * price_b
        spread_mean = spread.rolling(self.zscore_window).mean()
        spread_std = spread.rolling(self.zscore_window).std()
        zscore = (spread - spread_mean) / spread_std.replace(0, np.nan)

        position_a = pd.Series(0.0, index=zscore.index)
        position_a[zscore > self.entry_z] = -1.0   # спред завышен -> шорт A / лонг B
        position_a[zscore < -self.entry_z] = 1.0    # спред занижен -> лонг A / шорт B
        position_a[zscore.abs() < self.exit_z] = 0.0
        position_a = position_a.replace(0.0, np.nan).ffill().fillna(0.0)

        position_b = -position_a  # хедж-нога с противоположным знаком

        signals = {a: position_a.fillna(0.0), b: position_b.fillna(0.0)}
        for ticker, df in data.items():
            if ticker not in signals:
                signals[ticker] = pd.Series(0.0, index=df.index)
        return signals


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    n = 400
    common = np.cumsum(rng.normal(0, 0.01, n))
    price_a = 100 * np.exp(common + rng.normal(0, 0.005, n))
    price_b = 50 * np.exp(common * 0.9 + rng.normal(0, 0.005, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")

    def make_df(price):
        return pd.DataFrame(
            {"open": price, "high": price * 1.01, "low": price * 0.99, "close": price, "volume": 1_000_000},
            index=idx,
        )

    data = {"A": make_df(price_a), "B": make_df(price_b)}
    train_data = {k: v.iloc[:280] for k, v in data.items()}
    test_data = {k: v.iloc[280:] for k, v in data.items()}

    algo = KalmanFilterPairsTrading()
    algo.fit(train_data)
    signals = algo.generate_signals(test_data)
    for ticker, s in signals.items():
        print(ticker, s.describe())
