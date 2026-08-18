"""
Live news-sentiment monitor: RSS -> ticker match -> LLM score -> logged signal.

This is a forward-only, live process (see docs-level plan for why: RSS feeds
don't expose historical archives, so there's nothing to backtest against
yet). It polls the configured RSS feeds on an interval, and for every new
item that mentions a tracked MOEX ticker (core/ticker_linker.py), asks the
local LM Studio model (core/llm_sentiment.py) how holders/traders are
likely to react, turns that into position = direction * confidence, and
appends one JSON line per (news item, ticker) to
data/news_signals/YYYY-MM-DD.jsonl.

Requires LM Studio running locally with a model loaded and its local server
started (LM Studio -> Developer tab), and LMSTUDIO_MODEL set in .env to
match the loaded model's identifier.

Usage:
    PYTHONPATH=src python scripts/run_news_monitor.py [--interval SECONDS] [--once]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from core.llm_sentiment import LMStudioSentimentClient  # noqa: E402
from core.news_feed import NewsItem, RssNewsPoller  # noqa: E402
from core.ticker_linker import TickerLinker  # noqa: E402

_LOG_DIR = Path("data/news_signals")


def _log_path(when: datetime) -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR / f"{when.date().isoformat()}.jsonl"


def process_item(item: NewsItem, tickers: list[str], llm: LMStudioSentimentClient) -> list[dict]:
    records = []
    for ticker in tickers:
        try:
            score = llm.score(item, ticker)
        except Exception as exc:
            print(f"  ! LLM scoring failed for {ticker}: {type(exc).__name__}: {exc}")
            continue
        position = max(-1.0, min(1.0, score.direction * score.confidence))
        record = {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "source": item.source,
            "published_at": item.published_at.isoformat(),
            "title": item.title,
            "link": item.link,
            "ticker": ticker,
            "direction": score.direction,
            "confidence": score.confidence,
            "horizon": score.horizon,
            "event_type": score.event_type,
            "reasoning": score.reasoning,
            "position": position,
        }
        records.append(record)
        arrow = "BUY-ish" if position > 0.1 else "SELL-ish" if position < -0.1 else "flat"
        print(f"  [{ticker}] {arrow} pos={position:+.2f} ({score.event_type}, {score.horizon}) - {item.title}")
    return records


def run(interval_seconds: int, once: bool) -> None:
    poller = RssNewsPoller()
    linker = TickerLinker()
    llm = LMStudioSentimentClient()
    print(f"Watching {len(poller.feeds)} feeds, model={llm.model} @ {llm.base_url}")

    while True:
        # Picks up on/off feed toggles saved from the web dashboard
        # (configs/news_feeds.json) without needing to restart this process.
        poller.refresh_feeds()
        items = poller.poll_once()
        if items:
            print(f"{datetime.now().isoformat(timespec='seconds')}: {len(items)} new items")
        for item in items:
            tickers = linker.match(item.text)
            if not tickers:
                continue
            records = process_item(item, tickers, llm)
            if not records:
                continue
            # Keyed by processed_at (now), not item.published_at: RSS items can
            # be scored well after they were published (backlog catch-up, poll
            # lag), and core.market_simulator._latest_sentiment only ever looks
            # at "today's" (as_of.date()) file when a live session asks for the
            # latest score - a record filed under its news date instead of its
            # processing date would land in a file live sessions never read.
            path = _log_path(datetime.now(timezone.utc))
            with path.open("a", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        if once:
            break
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=120, help="Seconds between polls (default: 120)")
    parser.add_argument("--once", action="store_true", help="Poll a single time and exit (for testing)")
    args = parser.parse_args()
    run(interval_seconds=args.interval, once=args.once)
