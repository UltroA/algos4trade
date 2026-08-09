"""
LSTM - последовательности цен, объёмов, потока заявок.

рекуррентная сеть с LSTM-ячейками читает окно последних
`seq_len` дней инженерных признаков (моментум, волатильность, RSI, MACD, объём
из core.features.make_features) и предсказывает вероятность роста цены на
горизонте `horizon` дней. Реалистичный уровень (см. docs/Алгоритмы.md):
direction accuracy ~51-55%, часто не бьёт градиентный бустинг; при этом LSTM
долго обучается и легко переобучается на коротких финансовых рядах - здесь
это отчасти сдерживается небольшим скрытым размером и умеренным числом эпох.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import FEATURE_COLUMNS, chronological_split, make_features, split_features_target


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden) - скрытое состояние последнего слоя на последнем шаге
        return self.head(last_hidden).squeeze(-1)


class LSTMPredictor(SingleAssetAlgorithm):
    name = "LSTM Predictor"
    category = AlgorithmCategory.SEQUENCE_MODEL
    description = (
        "Рекуррентная сеть (LSTM) над окном последних технических признаков предсказывает "
        "вероятность роста цены на заданном горизонте по последовательности цен и объёмов."
    )

    def __init__(
        self,
        horizon: int = 1,
        seq_len: int = 20,
        hidden: int = 32,
        num_layers: int = 1,
        epochs: int = 25,
        lr: float = 1e-3,
        signal_strength: float = 4.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            horizon=horizon, seq_len=seq_len, hidden=hidden, num_layers=num_layers,
            epochs=epochs, lr=lr, signal_strength=signal_strength, seed=seed, **kwargs,
        )
        self.horizon = horizon
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.signal_strength = signal_strength
        torch.manual_seed(seed)
        self._scaler = StandardScaler()
        self.model: _LSTMNet | None = None

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

    def fit(self, train_data: pd.DataFrame) -> "LSTMPredictor":
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

        self.model = _LSTMNet(n_features=X.shape[1], hidden=self.hidden, num_layers=self.num_layers)
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
    algo = LSTMPredictor(epochs=10)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
