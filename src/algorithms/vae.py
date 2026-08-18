"""
Variational Autoencoder (VAE) - a probabilistic version of compressing features into factors.

Unlike the deterministic autoencoder.py, the VAE learns not a point in latent space but a distribution
q(z|x) = N(mu, exp(log_var)), regularized toward N(0, 1) via the KL divergence.
This should yield a smoother, less overfit latent space
(the denoising benefit noted in the table), at the cost of worse reconstruction accuracy. As with
autoencoder.py, the VAE by itself does not trade - on top of the deterministic mu
(no sampling at inference time), a downstream classifier of price
direction is trained. The latent factors remain uninterpretable (the main
weakness of row #14).
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


class _VAE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 8, latent_dim: int = 3):
        super().__init__()
        self.encoder_body = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.log_var_head = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.mu_head(h), self.log_var_head(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var


def _vae_loss(reconstruction: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(reconstruction, target)
    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_div


class VAEFactorModel(SingleAssetAlgorithm):
    name = "VAE Factor Model"
    category = AlgorithmCategory.REPRESENTATION_LEARNING
    description = (
        "A variational autoencoder compresses technical features into a probabilistic latent "
        "space N(mu, sigma); a linear classifier is trained on top of the deterministic mu."
    )

    def __init__(
        self,
        hidden: int = 8,
        latent_dim: int = 3,
        epochs: int = 200,
        lr: float = 1e-3,
        beta: float = 0.1,
        signal_strength: float = 4.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            hidden=hidden, latent_dim=latent_dim, epochs=epochs, lr=lr, beta=beta,
            signal_strength=signal_strength, seed=seed, **kwargs,
        )
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
        self.signal_strength = signal_strength
        torch.manual_seed(seed)
        self.scaler = StandardScaler()
        self.model: _VAE | None = None
        self.downstream = LogisticRegression(max_iter=1000)

    def fit(self, train_data: pd.DataFrame) -> "VAEFactorModel":
        feat_df = make_features(train_data)
        X, y = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty or y.nunique() < 2:
            self.is_fitted = False
            return self

        X_scaled = self.scaler.fit_transform(X.to_numpy(dtype=np.float32))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model = _VAE(X.shape[1], self.hidden, self.latent_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            reconstruction, mu, log_var = self.model(X_tensor)
            loss = _vae_loss(reconstruction, X_tensor, mu, log_var, self.beta)
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            mu_np, _ = self.model.encode(X_tensor)
            mu_np = mu_np.numpy()
        self.downstream.fit(mu_np, y.to_numpy())

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
            mu_np, _ = self.model.encode(X_tensor)
            mu_np = mu_np.numpy()

        proba_up = self.downstream.predict_proba(mu_np)[:, 1]
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
    algo = VAEFactorModel(epochs=100)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
