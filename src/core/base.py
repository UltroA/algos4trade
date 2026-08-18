"""
Unified OOP contract for all trading algorithms in the project.

Each algorithm in ``src/algorithms/`` inherits from :class:`BaseTradingAlgorithm`
(directly or via :class:`SingleAssetAlgorithm` / :class:`MultiAssetAlgorithm`)
and implements ``fit`` and ``generate_signals``. This unified interface lets
:class:`~core.backtester.Backtester` and :class:`~core.benchmark_runner.BenchmarkRunner`
run the backtest and measure performance the same way for any algorithm
from the docs/Алгоритмы.md table, regardless of what's inside it -
gradient boosting, a neural network, an RL agent, or genetic programming.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import pandas as pd


class AlgorithmCategory(str, Enum):
    """Algorithm category - as in the docs/Алгоритмы.md table."""

    SUPERVISED_RANKING = "supervised_ranking"
    LINEAR_FACTOR = "linear_factor"
    META_LABELING = "meta_labeling"
    REGIME_DETECTION = "regime_detection"
    PAIRS_STAT_ARB = "pairs_stat_arb"
    CLUSTERING = "clustering"
    SEQUENCE_MODEL = "sequence_model"
    TIME_SERIES_FORECAST = "time_series_forecast"
    PATTERN_RECOGNITION = "pattern_recognition"
    REPRESENTATION_LEARNING = "representation_learning"
    ANOMALY_DETECTION = "anomaly_detection"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CAPITAL_ALLOCATION = "capital_allocation"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    SYMBOLIC_REGRESSION = "symbolic_regression"
    COMPOSITE = "composite_pipeline"
    BASELINE = "baseline"
    NEWS_SENTIMENT = "news_sentiment"


class InputMode(str, Enum):
    """Defines the shape of the data the algorithm accepts in fit/generate_signals."""

    SINGLE_ASSET = "single_asset"   # pd.DataFrame with open/high/low/close/volume columns
    MULTI_ASSET = "multi_asset"     # dict[ticker -> pd.DataFrame]


class BaseTradingAlgorithm(ABC):
    """
    Base class for a trading algorithm.

    Class attributes (overridden in subclasses):
        name        - short human-readable name.
        category    - :class:`AlgorithmCategory`.
        input_mode  - :class:`InputMode`.
        description - 1-2 sentences on the algorithm's role in trading.

    Contract:
        * ``fit`` trains only on the data it's given (the train split),
          and must not see the test split - otherwise the backtest is invalid.
        * ``generate_signals`` returns, for each row of ``data``, a position
          in the range [-1, 1] (long/short/flat), using only information
          available AT THE TIME of that row (no look-ahead).
    """

    name: str = "BaseTradingAlgorithm"
    category: AlgorithmCategory = AlgorithmCategory.SUPERVISED_RANKING
    input_mode: InputMode = InputMode.SINGLE_ASSET
    description: str = ""

    def __init__(self, **hyperparams: Any) -> None:
        self.hyperparams = hyperparams
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, train_data: pd.DataFrame | dict[str, pd.DataFrame]) -> "BaseTradingAlgorithm":
        """Trains the model. Must return self and set is_fitted=True."""
        raise NotImplementedError

    @abstractmethod
    def generate_signals(
        self, data: pd.DataFrame | dict[str, pd.DataFrame]
    ) -> pd.Series | dict[str, pd.Series]:
        """
        Returns the target position over time.

        For SINGLE_ASSET: pd.Series of the same length/index as data, values in [-1, 1].
        For MULTI_ASSET: dict[ticker -> pd.Series] in the same format.
        """
        raise NotImplementedError

    def fit_generate(
        self,
        train_data: pd.DataFrame | dict[str, pd.DataFrame],
        eval_data: pd.DataFrame | dict[str, pd.DataFrame],
    ) -> pd.Series | dict[str, pd.Series]:
        """Convenience shortcut: fit on train, then generate signals on eval."""
        self.fit(train_data)
        return self.generate_signals(eval_data)

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<{self.__class__.__name__} name={self.name!r} fitted={self.is_fitted}>"


class SingleAssetAlgorithm(BaseTradingAlgorithm):
    """Convenience base class for algorithms with input_mode=SINGLE_ASSET."""

    input_mode = InputMode.SINGLE_ASSET

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> "SingleAssetAlgorithm":
        raise NotImplementedError

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class MultiAssetAlgorithm(BaseTradingAlgorithm):
    """Convenience base class for algorithms with input_mode=MULTI_ASSET (pairs, clustering, allocation)."""

    input_mode = InputMode.MULTI_ASSET

    @abstractmethod
    def fit(self, train_data: dict[str, pd.DataFrame]) -> "MultiAssetAlgorithm":
        raise NotImplementedError

    @abstractmethod
    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        raise NotImplementedError
