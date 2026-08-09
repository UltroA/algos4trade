"""
Transformer (упрощённый PatchTST) - патчинг + self-attention по временному ряду.

вместо подачи в attention каждого дня по отдельности, окно из `seq_len` дней признаков режется на непересекающиеся
патчи длиной `patch_len` (идея PatchTST - attention работает быстрее и лучше
обобщается на патчах, а не на отдельных точках). Каждый патч линейно
проецируется в d_model, к эмбеддингам добавляются обучаемые позиционные
эмбеддинги, дальше - стандартный nn.TransformerEncoder и усреднение по патчам.

Реалистичный уровень: хорош на дневных данных с длинной историей, слаб на
тиках; для устойчивого обучения нужно заметно больше данных, чем деревьям
или линейным моделям - здесь обучающая выборка (одна акция за несколько лет)
на грани минимально достаточной.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import FEATURE_COLUMNS, chronological_split, make_features, split_features_target


class _PatchTSTNet(nn.Module):
    def __init__(self, n_features: int, seq_len: int, patch_len: int = 5, d_model: int = 32, nhead: int = 4, num_layers: int = 1):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len  # обрезаем хвост окна, если seq_len не делится нацело
        self.used_len = self.n_patches * patch_len

        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> берём последние used_len шагов, режем на патчи
        x = x[:, -self.used_len :, :]
        batch = x.shape[0]
        patches = x.reshape(batch, self.n_patches, self.patch_len * x.shape[2])
        tokens = self.patch_proj(patches) + self.pos_embedding  # (batch, n_patches, d_model)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)  # mean pooling по токенам-патчам
        return self.head(pooled).squeeze(-1)


class PatchTSTPredictor(SingleAssetAlgorithm):
    name = "PatchTST Transformer Predictor"
    category = AlgorithmCategory.SEQUENCE_MODEL
    description = (
        "Упрощённый PatchTST: временной ряд признаков режется на патчи, кодируется "
        "self-attention трансформером и предсказывает вероятность роста цены на горизонте."
    )

    def __init__(
        self,
        horizon: int = 1,
        seq_len: int = 20,
        patch_len: int = 5,
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
            horizon=horizon, seq_len=seq_len, patch_len=patch_len, d_model=d_model, nhead=nhead,
            num_layers=num_layers, epochs=epochs, lr=lr, signal_strength=signal_strength, seed=seed, **kwargs,
        )
        self.horizon = horizon
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.signal_strength = signal_strength
        torch.manual_seed(seed)
        self._scaler = StandardScaler()
        self.model: _PatchTSTNet | None = None

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

    def fit(self, train_data: pd.DataFrame) -> "PatchTSTPredictor":
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

        self.model = _PatchTSTNet(
            n_features=X.shape[1], seq_len=self.seq_len, patch_len=self.patch_len,
            d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers,
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
    algo = PatchTSTPredictor(epochs=10)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
