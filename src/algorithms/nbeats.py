"""
N-BEATS - чистое прогнозирование временного ряда без внешних признаков.

сильные бенчмарки на M-соревнованиях
по прогнозированию временных рядов, но, в отличие от supervised-моделей
проекта (LightGBM/XGBoost/...), работает ТОЛЬКО с самим рядом доходностей -
без технических индикаторов и объёма. Главная слабость (см. таблицу):
прогноз цены/доходности ещё не значит прибыльную торговую стратегию - здесь
это явно видно, т.к. точный прогноз следующей доходности переводится в
позицию отдельным, наивным правилом (sign * величина), а не оптимизируется
напрямую под Sharpe/PnL.

Архитектура - упрощённый N-BEATS: стек из нескольких блоков, каждый блок -
MLP над окном последних `lookback` доходностей, выдающий (backcast, forecast).
backcast вычитается из входа перед следующим блоком (residual stacking -
ключевая идея N-BEATS: каждый следующий блок видит то, что не смог объяснить
предыдущий), а forecast-ы всех блоков суммируются в финальный прогноз.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split


class _NBEATSBlock(nn.Module):
    def __init__(self, lookback: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(lookback, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.backcast_head = nn.Linear(hidden, lookback)
        self.forecast_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast


class _NBEATSNet(nn.Module):
    def __init__(self, lookback: int, n_blocks: int = 3, hidden: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList([_NBEATSBlock(lookback, hidden) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        total_forecast = torch.zeros(x.shape[0], 1, device=x.device)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            total_forecast = total_forecast + forecast
        return total_forecast.squeeze(-1)


class NBEATSForecaster(SingleAssetAlgorithm):
    name = "N-BEATS Forecaster"
    category = AlgorithmCategory.TIME_SERIES_FORECAST
    description = (
        "Чистое прогнозирование следующей доходности по ряду доходностей (без внешних фич) "
        "через стек residual-блоков N-BEATS; прогноз конвертируется в позицию наивным правилом."
    )

    def __init__(
        self,
        lookback: int = 20,
        n_blocks: int = 3,
        hidden: int = 32,
        epochs: int = 60,
        lr: float = 1e-3,
        signal_strength: float = 20.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            lookback=lookback, n_blocks=n_blocks, hidden=hidden, epochs=epochs,
            lr=lr, signal_strength=signal_strength, seed=seed, **kwargs,
        )
        self.lookback = lookback
        self.n_blocks = n_blocks
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.signal_strength = signal_strength
        torch.manual_seed(seed)
        self.model: _NBEATSNet | None = None

    def _make_windows(self, returns: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        n = len(returns)
        xs, ys, idx = [], [], []
        for t in range(self.lookback, n - 1):
            xs.append(returns[t - self.lookback : t])
            ys.append(returns[t + 1])
            idx.append(t + 1)
        if not xs:
            return torch.empty(0, self.lookback), torch.empty(0), np.array(idx)
        return (
            torch.tensor(np.array(xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.float32),
            np.array(idx),
        )

    def fit(self, train_data: pd.DataFrame) -> "NBEATSForecaster":
        returns = train_data["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float32)
        X, y, _ = self._make_windows(returns)
        if len(X) < 10:
            self.is_fitted = False
            return self

        self.model = _NBEATSNet(self.lookback, self.n_blocks, self.hidden)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted or self.model is None:
            return pd.Series(0.0, index=data.index)

        returns = data["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float32)
        X, _, idx = self._make_windows(returns)
        if len(X) == 0:
            return pd.Series(0.0, index=data.index)

        self.model.eval()
        with torch.no_grad():
            predicted_returns = self.model(X).numpy()

        raw_signal = np.sign(predicted_returns) * np.minimum(np.abs(predicted_returns) * self.signal_strength, 1.0)
        signal = pd.Series(raw_signal, index=data.index[idx])
        return signal.reindex(data.index).fillna(0.0)


if __name__ == "__main__":
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
    algo = NBEATSForecaster(epochs=30)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
