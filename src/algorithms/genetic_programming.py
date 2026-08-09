"""
Генетическое программирование / symbolic regression - автоматический поиск
формул-альф (в духе WorldQuant "101 Formulaic Alphas").

symbolic regression - "эволюционный алгоритм" комбинирует технические признаки
арифметическими операциями (+, -, *, /, sqrt, abs, ...) в древовидные формулы
и отбирает те, что лучше предсказывают fwd_return. Найденная формула
интерпретируема (её можно прочитать как обычное математическое выражение) -
это и есть заявленное преимущество перед "чёрными ящиками" вроде нейросетей.

Главная слабость (см. таблицу): гигантский риск data mining bias - при большой
популяции/числе поколений и множестве кандидатных признаков эволюция almost
неизбежно находит формулу, которая просто переобучилась под шум train-выборки.
Здесь эта угроза НИЧЕМ не смягчена, кроме parsimony_coefficient (штраф за
сложность дерева) и обычного chronological train/test сплита бэктестера -
поэтому out-of-sample результат этого алгоритма стоит расценивать особенно
скептически.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split, make_features, split_features_target

_GP_FEATURE_COLUMNS = ["return_5d", "return_20d", "volatility_20", "rsi_14", "macd", "volume_change_5d"]


class SymbolicRegressionAlpha(SingleAssetAlgorithm):
    name = "Symbolic Regression Alpha"
    category = AlgorithmCategory.SYMBOLIC_REGRESSION
    description = (
        "Генетическое программирование эволюционирует интерпретируемую формулу-альфу из "
        "технических признаков, предсказывающую будущую доходность. Высокий риск data mining bias."
    )

    def __init__(
        self,
        horizon: int = 1,
        population_size: int = 500,
        generations: int = 15,
        parsimony_coefficient: float = 0.001,
        signal_strength: float = 10.0,
        seed: int = 0,
        n_jobs: int = 1,
        **kwargs,
    ):
        super().__init__(
            horizon=horizon,
            population_size=population_size,
            generations=generations,
            parsimony_coefficient=parsimony_coefficient,
            signal_strength=signal_strength,
            seed=seed,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.horizon = horizon
        self.signal_strength = signal_strength
        self.model = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            function_set=("add", "sub", "mul", "div", "sqrt", "abs", "neg", "max", "min"),
            parsimony_coefficient=parsimony_coefficient,
            stopping_criteria=0.001,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=0,
        )
        self.formula_: str | None = None

    def fit(self, train_data: pd.DataFrame) -> "SymbolicRegressionAlpha":
        feat_df = make_features(train_data, horizon=self.horizon)
        X, y = split_features_target(feat_df, target_col="fwd_return", feature_cols=_GP_FEATURE_COLUMNS)
        if X.empty or len(X) < 20:
            self.is_fitted = False
            return self

        self._feature_names = list(X.columns)
        self.model.set_params(feature_names=self._feature_names)
        self.model.fit(X.to_numpy(), y.to_numpy())
        self.formula_ = str(self.model._program)
        self.is_fitted = True
        return self

    def get_formula(self) -> str:
        """Возвращает найденную эволюцией формулу-альфу в читаемом виде."""
        return self.formula_ or "<not fitted>"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(0.0, index=data.index)

        feat_df = make_features(data, horizon=self.horizon)
        X, _ = split_features_target(feat_df, target_col="fwd_return", feature_cols=_GP_FEATURE_COLUMNS)
        if X.empty:
            return pd.Series(0.0, index=data.index)

        pred = self.model.predict(X.to_numpy())
        raw_signal = np.tanh(pred * self.signal_strength)
        signal = pd.Series(raw_signal, index=X.index).clip(-1.0, 1.0)
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

    # Уменьшенные параметры эволюции - только чтобы smoke-тест отрабатывал быстро.
    algo = SymbolicRegressionAlpha(population_size=300, generations=8)
    algo.fit(train_df)
    print("Найденная формула-альфа:", algo.get_formula())

    signals = algo.generate_signals(test_df)
    print(signals.describe())
