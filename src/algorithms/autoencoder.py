"""
Autoencoder - compresses technical features into latent factors.

Autoencoder(VAE): compresses a set of technical features
(from core.features) into a low-dimensional latent representation and denoises them.
On its own, the autoencoder does not predict price direction; per the table,
it "improves the input for other models." Here that is shown directly: on top of
the trained (unsupervised, no target) encoder, a simple downstream
classifier (LogisticRegression) is trained on the latent factors. The main weakness
is that the latent factors are not interpretable (unlike, for example, RSI/MACD in
the LightGBM model, where feature importance can be inspected directly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split, make_features, split_features_target


class _Autoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 8, latent_dim: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction


class AutoencoderFactorModel(SingleAssetAlgorithm):
    name = "Autoencoder Factor Model"
    category = AlgorithmCategory.REPRESENTATION_LEARNING
    description = (
        "Compresses technical features into latent factors with an autoencoder (unsupervised) "
        "and trains a simple linear classifier of price direction on top of them."
    )

    def __init__(
        self,
        hidden: int = 8,
        latent_dim: int = 3,
        epochs: int = 200,
        lr: float = 1e-3,
        signal_strength: float = 4.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            hidden=hidden, latent_dim=latent_dim, epochs=epochs, lr=lr,
            signal_strength=signal_strength, seed=seed, **kwargs,
        )
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.signal_strength = signal_strength
        torch.manual_seed(seed)
        self.scaler = StandardScaler()
        self.model: _Autoencoder | None = None
        self.downstream = LogisticRegression(max_iter=1000)

    def fit(self, train_data: pd.DataFrame) -> "AutoencoderFactorModel":
        feat_df = make_features(train_data)
        X, y = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty or y.nunique() < 2:
            self.is_fitted = False
            return self

        X_scaled = self.scaler.fit_transform(X.to_numpy(dtype=np.float32))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model = _Autoencoder(X.shape[1], self.hidden, self.latent_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            latent, reconstruction = self.model(X_tensor)
            loss = loss_fn(reconstruction, X_tensor)
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            latent_np = self.model(X_tensor)[0].numpy()
        self.downstream.fit(latent_np, y.to_numpy())

        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted or self.model is None:
            return pd.Series(0.0, index=data.index)

        feat_df = make_features(data)
        X, _ = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty:
            return pd.Series(0.0, index=data.index)

        X_scaled = self.scaler.transform(X.to_numpy(dtype=np.float32))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            latent_np = self.model(X_tensor)[0].numpy()

        proba_up = self.downstream.predict_proba(latent_np)[:, 1]
        raw_signal = (proba_up - 0.5) * self.signal_strength
        signal = pd.Series(np.clip(raw_signal, -1.0, 1.0), index=X.index)
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
    algo = AutoencoderFactorModel(epochs=100)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
