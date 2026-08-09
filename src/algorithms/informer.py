"""
Transformer (упрощённый Informer) - self-attention по полной последовательности.

Transformer(PatchTST/Informer). Внимание: это
аппроксимация Informer, а не полная реализация из оригинальной статьи -
здесь используется обычный nn.TransformerEncoder (полное O(seq_len^2)
внимание) вместо ProbSparse self-attention, ради которого Informer и был
предложен (снижение сложности внимания для длинных горизонтов). Каждый день
окна `seq_len` подаётся как отдельный токен (без патчинга, в отличие от
transformer_patchtst.py), линейно проецируется в d_model, к эмбеддингам
добавляются обучаемые позиционные эмбеддинги, дальше - TransformerEncoder
и mean pooling по временным шагам.

Реалистичный уровень: хорош на дневках с длинной историей, дорог по данным;
на коротких выборках (как здесь) склонен к переобучению не меньше, чем LSTM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import FEATURE_COLUMNS, chronological_split, make_features, split_features_target


class _InformerLiteNet(nn.Module):
    def __init__(self, n_features: int, seq_len: int, d_model: int = 32, nhead: int = 4, num_layers: int = 1):
        super().__init__()
        self.token_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) - каждый день = один токен
        tokens = self.token_proj(x) + self.pos_embedding
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)  # mean pooling по временным шагам
        return self.head(pooled).squeeze(-1)


class InformerPredictor(SingleAssetAlgorithm):
    name = "Informer-lite Transformer Predictor"
    category = AlgorithmCategory.SEQUENCE_MODEL
    description = (
        "Упрощённая (без ProbSparse-attention) аппроксимация Informer: self-attention "
        "по полной последовательности дневных признаков, предсказывает вероятность роста цены."
    )

    def __init__(
        self,
        horizon: int = 1,
        seq_len: int = 20,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        epochs: int = 25,
        lr: float = 1e-3,
        signal_strength: float = 4.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            horizon=horizon, seq_len=seq_len, d_model=d_model, nhead=nhead,
            num_layers=num_layers, epochs=epochs, lr=lr, signal_strength=signal_strength, seed=seed, **kwargs,
        )
        self.horizon = horizon
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.signal_strength = signal_strength
        torch.manual_seed(seed)
        self._scaler = StandardScaler()
        self.model: _InformerLiteNet | None = None

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

    def fit(self, train_data: pd.DataFrame) -> "InformerPredictor":
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

        self.model = _InformerLiteNet(
            n_features=X.shape[1], seq_len=self.seq_len, d_model=self.d_model,
            nhead=self.nhead, num_layers=self.num_layers,
        )
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
    algo = InformerPredictor(epochs=10)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
