"""
RSS polling for Russian business-media news, feeding the news-sentiment
pipeline (see core/ticker_linker.py, core/llm_sentiment.py,
algorithms/news_sentiment.py).

This is a live/forward-only data source (RSS feeds only expose current
news, not a historical archive), so it is deliberately kept separate from
the OHLCV pipeline (data_loader.py) - it does not go through disk-cached
parquet, it just tracks which entries have already been seen across polls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import mktime

import feedparser

logger = logging.getLogger(__name__)

# source label -> RSS feed URL. See docs/research or the project's news-sentiment
# plan for why each source was picked; some URLs (Finmarket, PRIME) should be
# re-verified periodically since specialized-media RSS paths change without notice.
DEFAULT_FEEDS: dict[str, str] = {
    "interfax": "https://www.interfax.ru/rss.asp",
    "finam_company_news": "https://www.finam.ru/analysis/conews/rsspoint/",
    "finam_news_comments": "https://www.finam.ru/analysis/nslent/rsspoint/",
    "vedomosti_markets": "https://www.vedomosti.ru/rss/rubric/finance/markets.xml",
    "vedomosti_business": "https://www.vedomosti.ru/rss/rubric/business.xml",
    "kommersant": "https://www.kommersant.ru/RSS/news.xml",
    "tass": "https://tass.ru/rss/v2.xml",
    "investing_ru_russia": "https://ru.investing.com/rss/news_12.rss",
    "investing_ru_stocks": "https://ru.investing.com/rss/news_25.rss",
}

# Per-source on/off toggle, written by scripts/web_configurator.py's news-feed
# panel. A flat {source: enabled} map rather than a copy of DEFAULT_FEEDS,
# so that a URL fix to DEFAULT_FEEDS always takes effect even for a user who
# already has this file on disk - only the enabled bit is persisted here, the
# URL is always looked up fresh from DEFAULT_FEEDS.
FEED_TOGGLE_CONFIG_PATH = Path("configs/news_feeds.json")


def _load_feed_toggles(path: str | Path = FEED_TOGGLE_CONFIG_PATH) -> dict[str, bool]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("could not read feed toggle config at %s, enabling all feeds", path)
        return {}
    return {source: bool(enabled) for source, enabled in raw.items()}


def feed_status(path: str | Path = FEED_TOGGLE_CONFIG_PATH) -> list[dict]:
    """Every DEFAULT_FEEDS source plus its current enabled/disabled state.

    A source with no entry in the toggle file is treated as enabled, so a
    fresh checkout polls every known feed until someone explicitly disables
    one from the web dashboard.
    """
    toggles = _load_feed_toggles(path)
    return [
        {"source": source, "url": url, "enabled": toggles.get(source, True)}
        for source, url in DEFAULT_FEEDS.items()
    ]


def enabled_feeds(path: str | Path = FEED_TOGGLE_CONFIG_PATH) -> dict[str, str]:
    """source -> url, restricted to feeds not disabled in the toggle config."""
    return {row["source"]: row["url"] for row in feed_status(path) if row["enabled"]}


def set_feed_enabled(source: str, enabled: bool, path: str | Path = FEED_TOGGLE_CONFIG_PATH) -> list[dict]:
    """Persists one source's on/off toggle and returns the resulting feed_status()."""
    if source not in DEFAULT_FEEDS:
        raise ValueError(f"unknown feed source {source!r}")
    path = Path(path)
    toggles = _load_feed_toggles(path)
    toggles[source] = bool(enabled)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(toggles, indent=2, ensure_ascii=False), encoding="utf-8")
    return feed_status(path)


@dataclass(frozen=True)
class NewsItem:
    source: str
    published_at: datetime
    title: str
    summary: str
    link: str
    guid: str

    @property
    def text(self) -> str:
        """Title + summary, the text handed to the ticker linker and the LLM."""
        return f"{self.title}\n{self.summary}".strip()


def _entry_guid(entry) -> str:
    return str(getattr(entry, "id", None) or getattr(entry, "link", None) or entry.get("title", ""))


def _entry_published(entry) -> datetime:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed is None:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)


class RssNewsPoller:
    """Polls a fixed set of RSS feeds and yields NewsItems not seen before.

    Seen entry ids are persisted to disk (cache_path) so a restarted process
    doesn't re-emit (and re-score with the LLM) news from a previous run.
    """

    def __init__(
        self,
        feeds: dict[str, str] | None = None,
        cache_path: str | Path = "data/news_cache/seen_ids.json",
        max_seen: int = 20_000,
    ):
        # feeds=None means "follow the toggle config", tracked via
        # _follow_toggle_config so refresh_feeds() knows whether to keep
        # re-reading it or leave an explicitly-passed feeds dict alone.
        self._follow_toggle_config = feeds is None
        self.feeds = feeds if feeds is not None else enabled_feeds()
        self._cache_path = Path(cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_seen = max_seen
        self._seen_ids: set[str] = self._load_seen()

    def refresh_feeds(self) -> None:
        """Re-reads which feeds are enabled from FEED_TOGGLE_CONFIG_PATH.

        Lets a long-running poll loop (scripts/run_news_monitor.py) pick up
        on/off toggles made from the web dashboard without restarting the
        monitor process. No-op if this poller was constructed with an
        explicit feeds dict.
        """
        if self._follow_toggle_config:
            self.feeds = enabled_feeds()

    def _load_seen(self) -> set[str]:
        if self._cache_path.exists():
            try:
                return set(json.loads(self._cache_path.read_text()))
            except (json.JSONDecodeError, OSError):
                logger.warning("could not read seen-ids cache at %s, starting fresh", self._cache_path)
        return set()

    def _save_seen(self) -> None:
        # Keep only the most recently added ids so the cache file doesn't grow forever.
        trimmed = list(self._seen_ids)[-self._max_seen :]
        self._seen_ids = set(trimmed)
        self._cache_path.write_text(json.dumps(trimmed))

    def poll_once(self) -> list[NewsItem]:
        """Fetches every configured feed once and returns new items only."""
        fresh: list[NewsItem] = []
        for source, url in self.feeds.items():
            try:
                parsed = feedparser.parse(url)
            except Exception:
                logger.exception("failed to fetch feed %s (%s)", source, url)
                continue
            if getattr(parsed, "bozo", False) and not parsed.entries:
                logger.warning("feed %s (%s) returned no usable entries: %s", source, url, getattr(parsed, "bozo_exception", ""))
                continue
            for entry in parsed.entries:
                guid = _entry_guid(entry)
                if not guid or guid in self._seen_ids:
                    continue
                self._seen_ids.add(guid)
                fresh.append(
                    NewsItem(
                        source=source,
                        published_at=_entry_published(entry),
                        title=getattr(entry, "title", "").strip(),
                        summary=getattr(entry, "summary", "").strip(),
                        link=getattr(entry, "link", ""),
                        guid=guid,
                    )
                )
        if fresh:
            self._save_seen()
        return fresh


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    poller = RssNewsPoller()
    items = poller.poll_once()
    print(f"{len(items)} new items across {len(poller.feeds)} feeds")
    for item in items[:5]:
        print(f"- [{item.source}] {item.title}")
