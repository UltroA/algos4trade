"""
LLM-based news impact scoring via a local LM Studio server.

LM Studio exposes an OpenAI-compatible REST API (default
http://localhost:1234/v1), so this uses the plain `openai` client pointed
at localhost - no cloud calls, no API key. The model itself (T-Pro-32B,
T-Lite-8B, Qwen3-32B, ...) is whatever is currently loaded in LM Studio;
switching it is a config change (LMSTUDIO_MODEL), not a code change.

The score is intentionally NOT "buy/sell" - the prompt asks the model to
reason about how holders/traders are likely to react to the news for a
specific, already-identified ticker (see core/ticker_linker.py), and the
resulting direction/confidence is turned into a position by a plain
arithmetic rule in algorithms/news_sentiment.py, not by the LLM itself.
"""

from __future__ import annotations

import os

from openai import OpenAI
from pydantic import BaseModel, Field

from .news_feed import NewsItem

_EVENT_TYPES = (
    "dividends",
    "earnings",
    "sanctions",
    "m_and_a",
    "management",
    "regulatory",
    "guidance",
    "other",
)
_HORIZONS = ("intraday", "days", "weeks")


class SentimentScore(BaseModel):
    direction: float = Field(ge=-1.0, le=1.0, description="Expected price reaction: -1 strongly negative, +1 strongly positive")
    confidence: float = Field(ge=0.0, le=1.0, description="Model's confidence in this assessment")
    horizon: str = Field(description=f"Expected reaction horizon, one of {_HORIZONS}")
    event_type: str = Field(description=f"News category, one of {_EVENT_TYPES}")
    reasoning: str = Field(description="One or two sentences on why holders/traders would react this way")


_SYSTEM_PROMPT = (
    "You are analyzing Russian business-media news for a specific MOEX-listed company. "
    "You are NOT giving financial advice - you are estimating how current holders and "
    "active traders of this specific stock are likely to react to this specific news, "
    "based only on the text given. Consider: is this new information or already priced "
    "in, is it about this company specifically or the sector/macro, and how large a "
    "reaction this type of news typically causes. "
    "Respond only with the requested JSON, no extra commentary."
)

_JSON_SCHEMA = {
    "name": "sentiment_score",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "direction": {"type": "number", "minimum": -1, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "horizon": {"type": "string", "enum": list(_HORIZONS)},
            "event_type": {"type": "string", "enum": list(_EVENT_TYPES)},
            "reasoning": {"type": "string"},
        },
        "required": ["direction", "confidence", "horizon", "event_type", "reasoning"],
        "additionalProperties": False,
    },
}


class LMStudioSentimentClient:
    """Scores a NewsItem's likely market impact on a given ticker via a local LLM."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        self.model = model or os.environ.get("LMSTUDIO_MODEL")
        if not self.model:
            raise ValueError(
                "No model configured - set LMSTUDIO_MODEL in .env to whatever is "
                "loaded in LM Studio (e.g. a T-Pro-32B or T-Lite-8B GGUF)."
            )
        self._client = OpenAI(base_url=self.base_url, api_key="lm-studio")

    def score(self, news_item: NewsItem, ticker: str) -> SentimentScore:
        user_prompt = (
            f"Ticker: {ticker}\n"
            f"Source: {news_item.source}\n"
            f"Published: {news_item.published_at.isoformat()}\n"
            f"Text:\n{news_item.text}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": _JSON_SCHEMA},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return SentimentScore.model_validate_json(content)


if __name__ == "__main__":
    from datetime import datetime, timezone

    dummy_item = NewsItem(
        source="test",
        published_at=datetime.now(timezone.utc),
        title="Сбербанк увеличил дивиденды на 20% по итогам года",
        summary="Наблюдательный совет Сбербанка рекомендовал рекордные дивиденды акционерам.",
        link="https://example.com",
        guid="test-1",
    )
    client = LMStudioSentimentClient()
    print(client.score(dummy_item, "SBER"))
