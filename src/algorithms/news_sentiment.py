"""
News Sentiment Signal - maps a precomputed LLM sentiment score to a position.

Unlike every other algorithm in this file's siblings, this one does not
compute its input from OHLCV: the `sentiment_score` column is expected to
already be present in each ticker's DataFrame (in [-1, 1], daily-aggregated),
produced offline by scripts/run_news_monitor.py from RSS news scored via
core/llm_sentiment.py (see that module and core/ticker_linker.py for how the
score is derived).

This is a MultiAssetAlgorithm, not a SingleAssetAlgorithm, specifically
because news coverage is sparse and uneven across tickers: on a given day
some tracked tickers get several news mentions and others get none. A
single-asset version of this algorithm would only ever be run against
core.market_simulator's "primary" ticker (tickers[0]) and would sit flat for
days at a time whenever that one specific ticker stayed out of the news,
even while the RSS+LLM pipeline was actively scoring other tickers in the
configured universe. Trading every ticker it is handed, each independently
off its own sentiment_score column, means it takes a position whenever
*any* covered ticker gets news, not just the one first in the list.

There is currently no historical RSS archive, so this class has nothing to
backtest against yet - it exists so that once scripts/run_news_monitor.py
has logged enough data/news_signals/*.jsonl to build a `sentiment_score`
column from, it plugs into Backtester/BenchmarkRunner exactly like any
other algorithm without further plumbing changes.
"""

from __future__ import annotations

import pandas as pd

from core.base import AlgorithmCategory, MultiAssetAlgorithm


class NewsSentimentSignal(MultiAssetAlgorithm):
    name = "News Sentiment Signal"
    category = AlgorithmCategory.NEWS_SENTIMENT
    description = (
        "Position = LLM-assessed news sentiment (direction * confidence) for the day, "
        "clipped to [-1, 1], independently per ticker. No training - the score comes from "
        "an external RSS+LLM pipeline (core/llm_sentiment.py), joined in as a "
        "'sentiment_score' column per ticker."
    )

    def __init__(self, decay_days: int = 1, **kwargs):
        # decay_days: how many rows a score stays in effect after the news day
        # (a genuinely NaN "no news" row would otherwise imply "flat" on every
        # day without news; a real 0.0 score is not decayed, since it is a
        # legitimate neutral read, not a missing one).
        super().__init__(decay_days=decay_days, **kwargs)
        self.decay_days = decay_days

    def fit(self, train_data: dict[str, pd.DataFrame]) -> "NewsSentimentSignal":
        # Nothing to train - the signal is a direct read of an externally computed column.
        self.is_fitted = True
        return self

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        return {ticker: self._signal_for(df) for ticker, df in data.items()}

    def _signal_for(self, df: pd.DataFrame) -> pd.Series:
        if "sentiment_score" not in df.columns:
            return pd.Series(0.0, index=df.index)
        score = df["sentiment_score"].clip(-1.0, 1.0)
        if self.decay_days > 1:
            score = score.ffill(limit=self.decay_days - 1)
        return score.fillna(0.0).astype(float)


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200
    index = pd.date_range("2023-01-01", periods=n, freq="B")

    def make_ticker_df(sentiment_probability: float) -> pd.DataFrame:
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        sentiment = np.where(
            rng.random(n) < sentiment_probability, rng.choice([0.6, -0.4], size=n), np.nan
        )
        return pd.DataFrame(
            {
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": rng.integers(1_000_000, 5_000_000, n),
                "sentiment_score": sentiment,
            },
            index=index,
        )

    # SBER gets frequent news coverage, GAZP gets almost none - demonstrates
    # that each ticker trades off its own coverage rather than sitting flat
    # whenever a single designated "primary" ticker has no news.
    dummy = {"SBER": make_ticker_df(0.2), "GAZP": make_ticker_df(0.02)}
    algo = NewsSentimentSignal(decay_days=2)
    algo.fit({t: df.iloc[:140] for t, df in dummy.items()})
    signals = algo.generate_signals({t: df.iloc[140:] for t, df in dummy.items()})
    for ticker, s in signals.items():
        print(f"--- {ticker} ---")
        print(s.describe())
