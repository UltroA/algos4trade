"""
Единый ООП-контракт для всех торговых алгоритмов проекта.

Каждый алгоритм в ``src/algorithms/`` наследует :class:`BaseTradingAlgorithm`
(напрямую или через :class:`SingleAssetAlgorithm` / :class:`MultiAssetAlgorithm`)
и реализует ``fit`` и ``generate_signals``. Единый интерфейс позволяет
:class:`~core.backtester.Backtester` и :class:`~core.benchmark_runner.BenchmarkRunner`
одинаково прогонять бэктест и замерять производительность для любого алгоритма
из таблицы docs/Алгоритмы.md, независимо от того, что у него внутри -
градиентный бустинг, нейросеть, RL-агент или генетическое программирование.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import pandas as pd


class AlgorithmCategory(str, Enum):
    """Категория алгоритма - как из таблицы docs/Алгоритмы.md."""

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


class InputMode(str, Enum):
    """Определяет форму данных, которые алгоритм принимает в fit/generate_signals."""

    SINGLE_ASSET = "single_asset"   # pd.DataFrame с колонками open/high/low/close/volume
    MULTI_ASSET = "multi_asset"     # dict[ticker -> pd.DataFrame]


class BaseTradingAlgorithm(ABC):
    """
    Базовый класс торгового алгоритма.

    Атрибуты класса (переопределяются в наследниках):
        name        - короткое человекочитаемое имя.
        category    - :class:`AlgorithmCategory`.
        input_mode  - :class:`InputMode`.
        description - 1-2 предложения о роли алгоритма в трейдинге.

    Контракт:
        * ``fit`` обучается только на данных, переданных ему (train-часть),
          и не должен видеть тестовую часть - иначе бэктест некорректен.
        * ``generate_signals`` для каждой строки ``data`` возвращает позицию
          в диапазоне [-1, 1] (лонг/шорт/флэт), используя только информацию,
          доступную НА МОМЕНТ этой строки (без заглядывания вперёд).
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
        """Обучает модель. Должен вернуть self и выставить is_fitted=True."""
        raise NotImplementedError

    @abstractmethod
    def generate_signals(
        self, data: pd.DataFrame | dict[str, pd.DataFrame]
    ) -> pd.Series | dict[str, pd.Series]:
        """
        Возвращает целевую позицию по времени.

        Для SINGLE_ASSET: pd.Series той же длины/индекса, что и data, значения в [-1, 1].
        Для MULTI_ASSET: dict[ticker -> pd.Series] в том же формате.
        """
        raise NotImplementedError

    def fit_generate(
        self,
        train_data: pd.DataFrame | dict[str, pd.DataFrame],
        eval_data: pd.DataFrame | dict[str, pd.DataFrame],
    ) -> pd.Series | dict[str, pd.Series]:
        """Удобный шорткат: fit на train, затем сигналы на eval."""
        self.fit(train_data)
        return self.generate_signals(eval_data)

    def __repr__(self) -> str:  # pragma: no cover - удобство отладки
        return f"<{self.__class__.__name__} name={self.name!r} fitted={self.is_fitted}>"


class SingleAssetAlgorithm(BaseTradingAlgorithm):
    """Удобный базовый класс для алгоритмов с input_mode=SINGLE_ASSET."""

    input_mode = InputMode.SINGLE_ASSET

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> "SingleAssetAlgorithm":
        raise NotImplementedError

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class MultiAssetAlgorithm(BaseTradingAlgorithm):
    """Удобный базовый класс для алгоритмов с input_mode=MULTI_ASSET (пары, кластеризация, аллокация)."""

    input_mode = InputMode.MULTI_ASSET

    @abstractmethod
    def fit(self, train_data: dict[str, pd.DataFrame]) -> "MultiAssetAlgorithm":
        raise NotImplementedError

    @abstractmethod
    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        raise NotImplementedError
