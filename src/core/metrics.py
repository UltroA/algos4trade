"""Метрики качества торговой стратегии, вычисляемые по ряду доходностей."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestMetrics:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    hit_rate: float
    n_trades: int

    def to_dict(self) -> dict:
        return asdict(self)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def compute_hit_rate(position_lagged: pd.Series, asset_returns: pd.Series) -> float:
    """
    HR = доля шагов, на которых знак вчерашней позиции совпал со знаком
    сегодняшней доходности актива: sign(p_{t-1}) == sign(r_t^asset).

    Шаги с p_{t-1} == 0 (позиция не открыта) исключены из знаменателя -
    иначе при полностью плоской стратегии (0 сделок) sign(0) == sign(0)
    формально даёт HR = 100%, что не отражает никакого предсказания.
    Если таких шагов вообще нет (стратегия ни разу не открыла позицию),
    HR не определён -> NaN.
    """
    position_lagged = position_lagged.reindex(asset_returns.index).fillna(0.0)
    mask = position_lagged != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(position_lagged[mask]) == np.sign(asset_returns[mask])).mean())


def compute_metrics(
    strategy_returns: pd.Series,
    signals: pd.Series,
    hit_rate: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestMetrics:
    """
    strategy_returns - доходность стратегии за период (уже с учётом позиции и издержек).
    signals          - позиция [-1, 1] в каждый момент (для подсчёта числа сделок и win-rate).
    hit_rate         - посчитан вызывающей стороной через compute_hit_rate (формула
                        завязана на лаг позиции и сырую доходность актива, которых
                        внутри этой функции нет).
    """
    strategy_returns = strategy_returns.fillna(0.0)
    equity_curve = (1.0 + strategy_returns).cumprod()

    total_return = float(equity_curve.iloc[-1] - 1.0) if len(equity_curve) else 0.0
    n_periods = len(strategy_returns)
    years = max(n_periods / periods_per_year, 1e-9)
    annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
    annualized_volatility = float(strategy_returns.std() * np.sqrt(periods_per_year))
    sharpe = sharpe_ratio(strategy_returns, periods_per_year)
    mdd = max_drawdown(equity_curve)

    nonzero_returns = strategy_returns[signals.reindex(strategy_returns.index).fillna(0) != 0]
    win_rate = float((nonzero_returns > 0).mean()) if len(nonzero_returns) else 0.0

    position_changes = signals.diff().fillna(signals.iloc[0] if len(signals) else 0)
    n_trades = int((position_changes.abs() > 1e-9).sum())

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        win_rate=win_rate,
        hit_rate=hit_rate,
        n_trades=n_trades,
    )
