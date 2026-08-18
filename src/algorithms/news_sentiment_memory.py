"""
News Sentiment Signal with Position Memory - a NewsSentimentSignal subclass
that adds a rolling per-ticker memory of brief news items seen while that
ticker is held (bought), and periodically re-queries the same LM Studio
model used by core/llm_sentiment.py to reassess the position from that
accumulated context instead of only the latest single headline.

NewsSentimentSignal itself is stateless: each call to generate_signals reads
the externally computed sentiment_score column fresh, with no notion of what
happened on earlier days once decay_days has elapsed. This subclass keeps
state across calls - for every ticker whose base signal currently indicates
a held long position, it appends any new (title, event_type, direction)
briefs found in data/news_signals/YYYY-MM-DD.jsonl (the same log
scripts/run_news_monitor.py writes) into a per-ticker deque bounded to
memory_size items. Once that memory changes, it builds a synthetic digest
NewsItem out of it and scores it with core.llm_sentiment.LMStudioSentimentClient
- the same client/model the base pipeline uses to score individual
headlines - and blends the resulting direction*confidence into the day's
position. This lets a held position's signal reflect "how has the news flow
around this stock trended since I bought it", not just "what did the latest
single headline say".

Like its parent, this has no historical RSS archive to backtest against, so
it is a live/demo-only algorithm for now (see NewsSentimentSignal's
docstring); fit() and the base sentiment_score read work identically, this
class only adds the memory/re-scoring layer on top.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.base import AlgorithmCategory
from core.llm_sentiment import LMStudioSentimentClient
from core.news_feed import NewsItem

from algorithms.news_sentiment import NewsSentimentSignal

logger = logging.getLogger(__name__)

_DEFAULT_NEWS_LOG_DIR = Path("data/news_signals")


class NewsSentimentMemorySignal(NewsSentimentSignal):
    name = "News Sentiment Signal (Position Memory)"
    category = AlgorithmCategory.NEWS_SENTIMENT
    description = (
        "NewsSentimentSignal plus a rolling memory of the last memory_size news briefs "
        "seen for each currently-held ticker. Re-scores that memory digest with the same "
        "LM Studio model used for individual headlines and blends the result into the "
        "day's position, so a held position responds to how its news flow has trended, "
        "not just the latest single headline."
    )

    def __init__(
        self,
        decay_days: int = 1,
        memory_size: int = 20,
        hold_threshold: float = 0.1,
        news_log_dir: str | Path = _DEFAULT_NEWS_LOG_DIR,
        llm: LMStudioSentimentClient | None = None,
        **kwargs,
    ):
        # memory_size: how many recent news briefs to retain per held ticker.
        # hold_threshold: base signal above this counts as "bought" for memory purposes.
        # llm=None means "construct LMStudioSentimentClient() lazily on first use" -
        # deferred so a smoke test can inject a fake client instead of requiring a
        # running LM Studio server just to import/instantiate this class.
        super().__init__(decay_days=decay_days, **kwargs)
        self.memory_size = memory_size
        self.hold_threshold = hold_threshold
        self.news_log_dir = Path(news_log_dir)
        self._llm = llm
        self._news_memory: dict[str, deque[str]] = {}
        self._memory_version: dict[str, int] = {}
        self._memory_score: dict[str, float] = {}
        self._memory_score_version: dict[str, int] = {}

    def _get_llm(self) -> LMStudioSentimentClient:
        if self._llm is None:
            self._llm = LMStudioSentimentClient()
        return self._llm

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        base_signals = super().generate_signals(data)
        for ticker, series in base_signals.items():
            held = bool(len(series)) and series.iloc[-1] > self.hold_threshold
            if held:
                self._remember_news(ticker, data[ticker])
        return {ticker: self._apply_memory(ticker, series) for ticker, series in base_signals.items()}

    def _remember_news(self, ticker: str, df: pd.DataFrame) -> None:
        as_of = df.index[-1]
        if isinstance(as_of, pd.Timestamp):
            as_of = as_of.to_pydatetime()
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        path = self.news_log_dir / f"{as_of.date().isoformat()}.jsonl"
        if not path.exists():
            return

        memory = self._news_memory.setdefault(ticker, deque(maxlen=self.memory_size))
        seen = set(memory)
        added = False
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("ticker") != ticker:
                    continue
                brief = (
                    f"{rec.get('title', '').strip()} "
                    f"({rec.get('event_type', 'other')}, dir={float(rec.get('direction', 0.0)):+.2f})"
                )
                if brief in seen:
                    continue
                memory.append(brief)
                seen.add(brief)
                added = True
        except OSError:
            logger.warning("could not read news log at %s", path)
            return

        if added:
            self._memory_version[ticker] = self._memory_version.get(ticker, 0) + 1

    def _apply_memory(self, ticker: str, series: pd.Series) -> pd.Series:
        memory = self._news_memory.get(ticker)
        if not memory:
            return series
        aggregate = self._aggregate_score(ticker, memory)
        if aggregate is None:
            return series
        blended = series.copy()
        if len(blended):
            blended.iloc[-1] = max(-1.0, min(1.0, (blended.iloc[-1] + aggregate) / 2.0))
        return blended

    def _aggregate_score(self, ticker: str, memory: deque[str]) -> float | None:
        version = self._memory_version.get(ticker, 0)
        if self._memory_score_version.get(ticker) == version:
            return self._memory_score.get(ticker)

        digest = "\n".join(f"- {brief}" for brief in memory)
        digest_item = NewsItem(
            source="position_memory_digest",
            published_at=datetime.now(timezone.utc),
            title=f"Сводка последних {len(memory)} новостей по {ticker} за время удержания позиции",
            summary=digest,
            link="",
            guid=f"memory:{ticker}:{version}",
        )
        try:
            score = self._get_llm().score(digest_item, ticker)
        except Exception:
            logger.exception("memory re-scoring failed for %s", ticker)
            return None

        aggregate = max(-1.0, min(1.0, score.direction * score.confidence))
        self._memory_score[ticker] = aggregate
        self._memory_score_version[ticker] = version
        return aggregate


if __name__ == "__main__":
    import tempfile

    import numpy as np
    from pydantic import BaseModel

    class _FakeSentimentScore(BaseModel):
        direction: float
        confidence: float
        horizon: str = "days"
        event_type: str = "other"
        reasoning: str = "fake client for smoke test"

    class _FakeLLMClient:
        """Stands in for LMStudioSentimentClient so this smoke test runs without a
        live LM Studio server - always reassesses the digest as mildly positive."""

        model = "fake-smoke-test-model"

        def score(self, news_item: NewsItem, ticker: str) -> _FakeSentimentScore:
            return _FakeSentimentScore(direction=0.5, confidence=0.8)

    rng = np.random.default_rng(0)
    n = 30
    index = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    sentiment = np.zeros(n)
    sentiment[2] = 0.6  # SBER gets bought on day 2 and held for the rest of the window
    df = pd.DataFrame(
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

    with tempfile.TemporaryDirectory() as tmpdir:
        news_log_dir = Path(tmpdir)
        for i, day in enumerate(index):
            path = news_log_dir / f"{day.date().isoformat()}.jsonl"
            record = {
                "processed_at": day.isoformat(),
                "ticker": "SBER",
                "title": f"Новость дня {i} про SBER",
                "event_type": "guidance",
                "direction": 0.3,
            }
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

        # decay_days spans the whole window so the day-2 buy stays "held" through
        # every later tick - this mirrors how core.market_simulator actually drives
        # the algorithm: one generate_signals() call per tick on the growing history
        # so far, not a single call over the whole backtest window at once.
        algo = NewsSentimentMemorySignal(
            decay_days=n, memory_size=20, news_log_dir=news_log_dir, llm=_FakeLLMClient()
        )
        algo.fit({"SBER": df.iloc[:2]})
        for i in range(2, n):
            tick_signals = algo.generate_signals({"SBER": df.iloc[: i + 1]})

        print(tick_signals["SBER"].tail(10))
        print(f"remembered {len(algo._news_memory['SBER'])}/20 briefs for SBER (from {n - 2} held days)")
