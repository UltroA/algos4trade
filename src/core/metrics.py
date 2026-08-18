"""Trading strategy quality metrics computed from a returns series."""

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
    starting_capital: float = 1_000_000.0
    final_capital: float = 1_000_000.0
    pnl_rub: float = 0.0

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
    HR = fraction of steps where the sign of yesterday's position matched
    the sign of today's asset return: sign(p_{t-1}) == sign(r_t^asset).

    Steps with p_{t-1} == 0 (no open position) are excluded from the
    denominator - otherwise, for a fully flat strategy (0 trades),
    sign(0) == sign(0) would formally give HR = 100%, which reflects no
    prediction at all.
    If there are no such steps at all (the strategy never opened a position),
    HR is undefined -> NaN.
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
    starting_capital: float = 1_000_000.0,
) -> BacktestMetrics:
    """
    strategy_returns - strategy return per period (already accounting for
                        position and costs).
    signals          - position [-1, 1] at each point in time (used to count
                        trades and win rate).
    hit_rate         - computed by the caller via compute_hit_rate (the formula
                        depends on the lagged position and raw asset return,
                        neither of which is available inside this function).
    starting_capital - notional account size (RUB) used to express total_return
                        as money won/lost (final_capital, pnl_rub), alongside
                        the percentage metrics above.
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

    final_capital = starting_capital * (1.0 + total_return)

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        win_rate=win_rate,
        hit_rate=hit_rate,
        n_trades=n_trades,
        starting_capital=starting_capital,
        final_capital=final_capital,
        pnl_rub=final_capital - starting_capital,
    )
