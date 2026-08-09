"""
CNN по "картинкам" графиков - распознавание паттернов на изображениях свечей.

вместо табличных признаков модель получает на вход
растеризованное изображение окна свечей (OHLC) и учится классифицировать
направление следующей свечи по визуальному паттерну. Реалистичный уровень -
52-54% direction accuracy; главная слабость - "чёрный ящик, тяжело
отлаживать" (в отличие от признаков LightGBM/RandomForest здесь нет прямой
интерпретации важности входа).

Рендеринг через matplotlib был бы на порядки медленнее и не нужен - окно
свечей растеризуется вручную в маленький numpy-массив (1, H, W): для каждой
свечи один "столбец тени" (от low до high) и один "столбец тела" (от open
до close), высота нормализована min-max по high/low всего окна.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split

_IMG_HEIGHT = 32


def _rasterize_window(window: pd.DataFrame, height: int = _IMG_HEIGHT) -> np.ndarray:
    """Превращает окно OHLC-свечей в изображение (1, height, width), width = 2 * n_candles."""
    n = len(window)
    width = 2 * n
    img = np.zeros((height, width), dtype=np.float32)

    lo = window["low"].min()
    hi = window["high"].max()
    scale = (hi - lo) if hi > lo else 1.0

    def to_row(price: float) -> int:
        return int(np.clip((price - lo) / scale * (height - 1), 0, height - 1))

    for i, (_, candle) in enumerate(window.iterrows()):
        shadow_col = 2 * i
        body_col = 2 * i + 1

        low_row, high_row = to_row(candle["low"]), to_row(candle["high"])
        img[min(low_row, high_row) : max(low_row, high_row) + 1, shadow_col] = 0.5

        open_row, close_row = to_row(candle["open"]), to_row(candle["close"])
        body_lo, body_hi = min(open_row, close_row), max(open_row, close_row)
        intensity = 1.0 if candle["close"] >= candle["open"] else 0.75
        img[body_lo : body_hi + 1, body_col] = intensity

    return img[np.newaxis, :, :]  # (1, H, W)


class _CandlestickCNNNet(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        out_h, out_w = height // 4, width // 4
        self.classifier = nn.Linear(16 * out_h * out_w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.classifier(h).squeeze(-1)


class CandlestickCNN(SingleAssetAlgorithm):
    name = "Candlestick Pattern CNN"
    category = AlgorithmCategory.PATTERN_RECOGNITION
    description = (
        "Сверточная сеть распознаёт визуальные паттерны в растеризованном окне свечей "
        "и классифицирует направление следующей свечи."
    )

    def __init__(
        self,
        window: int = 20,
        epochs: int = 15,
        lr: float = 1e-3,
        batch_size: int = 32,
        signal_strength: float = 4.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            window=window, epochs=epochs, lr=lr, batch_size=batch_size,
            signal_strength=signal_strength, seed=seed, **kwargs,
        )
        self.window = window
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.signal_strength = signal_strength
        torch.manual_seed(seed)
        self.model: _CandlestickCNNNet | None = None

    def _build_dataset(self, data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        closes = data["close"].to_numpy()
        images, labels, idx = [], [], []
        for t in range(self.window, len(data) - 1):
            window_df = data.iloc[t - self.window : t]
            images.append(_rasterize_window(window_df))
            labels.append(1.0 if closes[t + 1] > closes[t] else 0.0)
            idx.append(t + 1)
        if not images:
            return torch.empty(0), torch.empty(0), np.array(idx)
        return (
            torch.tensor(np.array(images), dtype=torch.float32),
            torch.tensor(np.array(labels), dtype=torch.float32),
            np.array(idx),
        )

    def fit(self, train_data: pd.DataFrame) -> "CandlestickCNN":
        X, y, _ = self._build_dataset(train_data)
        if len(X) < 20:
            self.is_fitted = False
            return self

        self.model = _CandlestickCNNNet(_IMG_HEIGHT, 2 * self.window)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        n = len(X)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                logits = self.model(X[batch_idx])
                loss = loss_fn(logits, y[batch_idx])
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted or self.model is None:
            return pd.Series(0.0, index=data.index)

        X, _, idx = self._build_dataset(data)
        if len(X) == 0:
            return pd.Series(0.0, index=data.index)

        self.model.eval()
        with torch.no_grad():
            proba_up = torch.sigmoid(self.model(X)).numpy()

        raw_signal = (proba_up - 0.5) * self.signal_strength
        signal = pd.Series(np.clip(raw_signal, -1.0, 1.0), index=data.index[idx])
        return signal.reindex(data.index).fillna(0.0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    idx_range = pd.date_range("2023-01-01", periods=n, freq="B")
    dummy = pd.DataFrame(
        {
            "open": price * (1 + rng.normal(0, 0.002, n)),
            "high": price * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": price * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": price,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx_range,
    )
    train_df, test_df = chronological_split(dummy, 0.7)
    algo = CandlestickCNN(epochs=5)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
