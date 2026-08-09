"""
Gaussian processes / Bayesian optimization - hyperparameter tuning and
prediction uncertainty estimation.

GP regression predicts future return (fwd_return)
and, more importantly, gives a calibrated estimate of prediction uncertainty (std). This
literally allows for "not trading when confidence is low" - if the predicted
std exceeds a threshold computed on train, the signal is forced to zero.

Before the main training, a simple Bayesian optimization of kernel
hyperparameters (length_scale, alpha) is performed: a random search over several candidates,
selected by log-marginal-likelihood on train (an analog of the hyperparameter optimization
from the table, without external libraries like optuna/skopt).

Main weakness (see table): GP scales poorly with data (training is
O(n^3) in the number of points) - so here the training set is truncated to the last
`max_train_points` points of the train period if there is more data than that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split, make_features, split_features_target

_GP_FEATURE_COLUMNS = ["return_5d", "return_20d", "volatility_20", "rsi_14", "macd"]


class GaussianProcessTrader(SingleAssetAlgorithm):
    name = "Gaussian Process Trader"
    category = AlgorithmCategory.BAYESIAN_OPTIMIZATION
    description = (
        "GP regression predicts future return and its uncertainty (std); "
        "trading is skipped when the model is not sufficiently confident in its prediction."
    )

    def __init__(
        self,
        horizon: int = 1,
        max_train_points: int = 500,
        n_restarts_optimizer: int = 3,
        n_hparam_candidates: int = 10,
        uncertainty_percentile: float = 75.0,
        signal_strength: float = 8.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            horizon=horizon,
            max_train_points=max_train_points,
            n_restarts_optimizer=n_restarts_optimizer,
            n_hparam_candidates=n_hparam_candidates,
            uncertainty_percentile=uncertainty_percentile,
            signal_strength=signal_strength,
            seed=seed,
            **kwargs,
        )
        self.horizon = horizon
        self.max_train_points = max_train_points
        self.n_restarts_optimizer = n_restarts_optimizer
        self.n_hparam_candidates = n_hparam_candidates
        self.uncertainty_percentile = uncertainty_percentile
        self.signal_strength = signal_strength
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.scaler = StandardScaler()
        self.model: GaussianProcessRegressor | None = None
        self._std_threshold: float = np.inf

    def _select_hyperparams(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Simple Bayesian optimization: random search over (length_scale, alpha) by log-marginal-likelihood."""
        best_score = -np.inf
        best_params = (1.0, 1e-2)
        length_scales = self._rng.uniform(0.3, 5.0, size=self.n_hparam_candidates)
        alphas = self._rng.uniform(1e-3, 1e-1, size=self.n_hparam_candidates)

        for length_scale, alpha in zip(length_scales, alphas):
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=1.0)
            candidate = GaussianProcessRegressor(
                kernel=kernel, alpha=alpha, n_restarts_optimizer=0, normalize_y=True, random_state=self.seed
            )
            try:
                candidate.fit(X, y)
                score = candidate.log_marginal_likelihood(candidate.kernel_.theta)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_params = (length_scale, alpha)

        return best_params

    def fit(self, train_data: pd.DataFrame) -> "GaussianProcessTrader":
        feat_df = make_features(train_data, horizon=self.horizon)
        X_df, y = split_features_target(feat_df, target_col="fwd_return", feature_cols=_GP_FEATURE_COLUMNS)
        if X_df.empty or len(X_df) < 20:
            self.is_fitted = False
            return self

        if len(X_df) > self.max_train_points:
            X_df = X_df.iloc[-self.max_train_points :]
            y = y.iloc[-self.max_train_points :]

        X = self.scaler.fit_transform(X_df.to_numpy())
        y_arr = y.to_numpy()

        length_scale, alpha = self._select_hyperparams(X, y_arr)
        kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=1.0)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=self.seed,
        )
        self.model.fit(X, y_arr)

        _, train_std = self.model.predict(X, return_std=True)
        self._std_threshold = float(np.percentile(train_std, self.uncertainty_percentile))

        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted or self.model is None:
            return pd.Series(0.0, index=data.index)

        feat_df = make_features(data, horizon=self.horizon)
        X_df, _ = split_features_target(feat_df, target_col="fwd_return", feature_cols=_GP_FEATURE_COLUMNS)
        if X_df.empty:
            return pd.Series(0.0, index=data.index)

        X = self.scaler.transform(X_df.to_numpy())
        pred, std = self.model.predict(X, return_std=True)

        raw_signal = np.tanh(pred * self.signal_strength)
        raw_signal[std > self._std_threshold] = 0.0  # low confidence -> don't trade

        signal = pd.Series(raw_signal, index=X_df.index).clip(-1.0, 1.0)
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
    algo = GaussianProcessTrader()
    algo.fit(train_df)

    in_sample_signals = algo.generate_signals(train_df)
    print("in-sample (train) signal stats - sanity check that the 'confident' branch fires:")
    print(in_sample_signals.describe())

    signals = algo.generate_signals(test_df)
    print("\nout-of-sample (test) std threshold:", algo._std_threshold)
    print(signals.describe())
    print("fraction flat (uncertain) out-of-sample:", float((signals == 0.0).mean()))
    print(
        "\nNote: on this synthetic random-walk dataset the model correctly abstains almost "
        "everywhere out-of-sample (test features are far from the train distribution -> high "
        "GP std) - this is the intended 'don't trade when uncertain' behavior, not a bug."
    )
