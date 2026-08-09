"""
TCN (Temporal Convolutional Network) - causal dilated-свёртки по временному ряду.

то же назначение, что у LSTM/GRU (последовательности цен/объёмов/индикаторов),
но за счёт параллельных dilated-свёрток обучается
быстрее и стабильнее рекуррентных сетей. Главная слабость - фиксированное
рецептивное поле: сеть не видит закономерности длиннее, чем позволяет глубина
и dilation слоёв (здесь рецептивное поле = 1 + 2*(1+2+4) = 15 шагов при
kernel_size=3, dilations=[1,2,4]).

Все свёртки - causal: паддинг добавляется только слева на (kernel_size-1)*dilation
шагов и затем обрезается справа, чтобы предсказание в момент t никогда не
использовало будущие значения (t+1, t+2, ...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import FEATURE_COLUMNS, chronological_split, make_features, split_features_target


class _CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]  # обрезаем "будущий" паддинг справа -> causal
        return self.relu(out)


class _TCNNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, kernel_size: int = 3, dilations: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        blocks = []
        in_channels = n_features
        for dilation in dilations:
            blocks.append(_CausalConvBlock(in_channels, hidden, kernel_size, dilation))
            in_channels = hidden
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> (batch, n_features, seq_len) для Conv1d
        x = x.permute(0, 2, 1)
        h = self.blocks(x)  # (batch, hidden, seq_len)
        last_step = h[:, :, -1]  # берём представление последнего временного шага
        return self.head(last_step).squeeze(-1)


class TCNPredictor(SingleAssetAlgorithm):
    name = "TCN Predictor"
    category = AlgorithmCategory.SEQUENCE_MODEL
    description = (
        "Temporal Convolutional Network с causal dilated-свёртками предсказывает вероятность "
        "роста цены по окну технических признаков - быстрее и стабильнее LSTM, но с фиксированным "
        "рецептивным полем."
    )

    def __init__(
        self,
        horizon: int = 1,
        seq_len: int = 20,
        hidden: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        epochs: int = 25,
        lr: float = 1e-3,
        signal_strength: float = 4.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            horizon=horizon, seq_len=seq_len, hidden=hidden, kernel_size=kernel_size,
            dilations=dilations, epochs=epochs, lr=lr, signal_strength=signal_strength, seed=seed, **kwargs,
        )
        self.horizon = horizon
        self.seq_len = seq_len
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.epochs = epochs
        self.lr = lr
        self.signal_strength = signal_strength
        torch.manual_seed(seed)
        self._scaler = StandardScaler()
        self.model: _TCNNet | None = None

    def _make_sequences(self, X: np.ndarray, y: np.ndarray | None, seq_len: int):
        """Скользящее окно длиной seq_len; таргет - значение y в последней точке окна."""
        n = len(X)
        if n <= seq_len:
            empty_x = np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.float32) if y is not None else None
            return empty_x, empty_y

        n_samples = n - seq_len + 1
        seqs = np.stack([X[i : i + seq_len] for i in range(n_samples)]).astype(np.float32)
        targets = y[seq_len - 1 :].astype(np.float32) if y is not None else None
        return seqs, targets

    def fit(self, train_data: pd.DataFrame) -> "TCNPredictor":
        feat_df = make_features(train_data, horizon=self.horizon)
        X_df, y = split_features_target(feat_df, target_col="fwd_direction")
        if len(X_df) < self.seq_len * 3 or y.nunique() < 2:
            self.is_fitted = False
            return self

        X = self._scaler.fit_transform(X_df.to_numpy())
        X_seq, y_seq = self._make_sequences(X, y.to_numpy(), self.seq_len)
        if len(X_seq) < self.seq_len:
            self.is_fitted = False
            return self

        self.model = _TCNNet(n_features=X.shape[1], hidden=self.hidden, kernel_size=self.kernel_size, dilations=self.dilations)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        X_t = torch.as_tensor(X_seq, dtype=torch.float32)
        y_t = torch.as_tensor(y_seq, dtype=torch.float32)

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted or self.model is None:
            return pd.Series(0.0, index=data.index)

        feat_df = make_features(data, horizon=self.horizon)
        cols = [c for c in FEATURE_COLUMNS if c in feat_df.columns]
        clean = feat_df.dropna(subset=cols)
        if len(clean) <= self.seq_len:
            return pd.Series(0.0, index=data.index)

        X = self._scaler.transform(clean[cols].to_numpy())
        X_seq, _ = self._make_sequences(X, None, self.seq_len)
        if len(X_seq) == 0:
            return pd.Series(0.0, index=data.index)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.as_tensor(X_seq, dtype=torch.float32))
            proba_up = torch.sigmoid(logits).numpy()

        raw_signal = (proba_up - 0.5) * self.signal_strength
        signal_index = clean.index[self.seq_len - 1 :]
        signal = pd.Series(raw_signal, index=signal_index).clip(-1.0, 1.0)
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
    algo = TCNPredictor(epochs=10)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
