"""
Deterministic entity linking: Russian company name/brand mentions -> MOEX ticker.

Deliberately NOT delegated to the LLM - asking a model to recall which ticker
a company trades under risks hallucinated/stale mappings, and a plain alias
match is cheap enough to run on every RSS item before any LLM call, which
also cuts the number of (paid-in-latency) LLM calls by filtering out the
large majority of news that doesn't mention a tracked issuer at all.

The alias table below is a manually curated starter set covering the most
liquid / frequently newsworthy names from scripts/moex_universe.py's
TICKERS_100 universe. It is not exhaustive - extend ALIASES as gaps are found
(e.g. by checking scripts/moex_universe.py output against missed mentions).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from moex_universe import TICKERS_100  # noqa: E402

# ticker -> Russian name variants as they actually appear in news prose
# (lowercase, no punctuation needed - matching is done case-insensitively).
ALIASES: dict[str, list[str]] = {
    "AFKS": ["афк система", "система"],
    "AFLT": ["аэрофлот"],
    "AKRN": ["акрон"],
    "ALRS": ["алроса"],
    "APTK": ["аптеки 36,6", "аптечная сеть 36.6", "36,6"],
    "AQUA": ["инарктика", "русская аквакультура"],
    "ASTR": ["группа астра", "астра"],
    "BANEP": ["башнефть"],
    "BELU": ["белуга групп", "белуга", "novabev", "новабев"],
    "BSPB": ["банк санкт-петербург", "бспб"],
    "CBOM": ["мкб", "московский кредитный банк"],
    "CHMF": ["северсталь"],
    "DOMRF": ["дом.рф"],
    "FEES": ["фск россети", "фск еэс", "россети"],
    "FESH": ["двмп", "дальневосточное морское пароходство", "fesco", "феско"],
    "FIXR": ["фикс прайс", "fix price"],
    "FLOT": ["совкомфлот"],
    "GAZP": ["газпром"],
    "GEMC": ["united medical group", "юмг", "european medical center", "emc"],
    "GMKN": ["норникель", "норильский никель"],
    "HEAD": ["хэдхантер", "headhunter", "hh.ru"],
    "HYDR": ["русгидро"],
    "IRAO": ["интер рао"],
    "LENT": ["лента"],
    "LKOH": ["лукойл"],
    "LSRG": ["группа лср", "лср"],
    "MAGN": ["ммк", "магнитогорский металлургический комбинат"],
    "MBNK": ["мтс банк", "мтс-банк"],
    "MDMG": ["мать и дитя", "md medical"],
    "MGNT": ["магнит"],
    "MOEX": ["московская биржа"],
    "MRKC": ["россети центр"],
    "MRKP": ["россети центр и приволжье"],
    "MRKU": ["россети урал"],
    "MRKV": ["россети волга"],
    "MSNG": ["мосэнерго"],
    "MSRS": ["россети московский регион", "моэск"],
    "MTLR": ["мечел"],
    "MTLRP": ["мечел-п", "мечел ап", "привилегированные акции мечела"],
    "MTSS": ["мтс", "мобильные телесистемы"],
    "NKNCP": ["нижнекамскнефтехим"],
    "NLMK": ["нлмк", "новолипецкий металлургический комбинат"],
    "NMTP": ["нмтп", "новороссийский морской торговый порт"],
    "NVTK": ["новатэк"],
    "OZON": ["озон", "ozon"],
    "OZPH": ["озон фармацевтика"],
    "PHOR": ["фосагро"],
    "PIKK": ["пик", "группа пик"],
    "PLZL": ["полюс золото", "полюс"],
    "POSI": ["позитив текнолоджиз", "positive technologies", "позитив"],
    "PRMD": ["промомед"],
    "RASP": ["распадская"],
    "RENI": ["ренессанс страхование"],
    "RNFT": ["русснефть"],
    "ROSN": ["роснефть"],
    "RTKM": ["ростелеком"],
    "RTKMP": ["ростелеком-п", "ростелеком ап", "привилегированные акции ростелекома"],
    "RUAL": ["русал"],
    # "сбер" is only 4 letters, under _word_pattern's >5 stemming threshold, so
    # it matches ONLY the bare nominative "Сбер" - not "Сбера"/"Сберу"/"Сбером"/
    # "Сбере", the case forms that make up most real mentions ("акции Сбера
    # выросли", "аналитики Сбера"). "сбербанк" (8 letters) stems fine on its
    # own; declined forms of the short name are listed explicitly instead of
    # lowering the global stemming threshold (which would reintroduce false
    # positives on unrelated short words - see _word_pattern's docstring).
    "SBER": ["сбербанк", "сбер", "сбера", "сберу", "сбером", "сбере"],
    "SBERP": ["сбербанк-п", "сбербанк ап", "сбер-п", "привилегированные акции сбербанка"],
    "SELG": ["селигдар"],
    "SFIN": ["эсэфай", "sfi"],
    "SGZH": ["сегежа", "segezha"],
    "SMLT": ["самолет", "группа самолет"],
    "SNGS": ["сургутнефтегаз"],
    "SNGSP": ["сургутнефтегаз-п", "сургутнефтегаз ап", "привилегированные акции сургутнефтегаза"],
    "SOFL": ["софтлайн", "softline"],
    "SPBE": ["спб биржа"],
    "SVAV": ["соллерс", "sollers"],
    "SVCB": ["совкомбанк"],
    "T": ["т-технологии", "т-банк", "тинькофф"],
    "TATN": ["татнефть"],
    "TATNP": ["татнефть-п", "татнефть ап", "привилегированные акции татнефти"],
    "TGKA": ["тгк-1"],
    "TRMK": ["тмк", "трубная металлургическая компания"],
    "TRNFP": ["транснефть"],
    "VKCO": ["вконтакте", " vk ", "вк групп"],
    "VSMO": ["всмпо-ависма", "ависма"],
    "VTBR": ["втб"],
    "WUSH": ["whoosh", "вуш"],
    "X5": ["x5", "x5 group", "пятёрочка", "пятерочка", "перекрёсток", "перекресток"],
    "YDEX": ["яндекс", "yandex"],
}


_CYRILLIC_WORD = re.compile(r"^[а-яё]+$", re.IGNORECASE)

# Single-word aliases that are also common Russian words, unrelated to the
# company most of the time ("самолет" = airplane, "позитив" = a good mood,
# "система" = generic "system", "магнит" = a magnet, "лента" = a ribbon/tape
# and, notably, "news feed", "пик" = a mountain peak, "полюс" = a geographic
# pole) - matching these bare produces heavy false positives (e.g. any
# drone/aviation story matches SMLT, an earthquake-magnitude story matches
# MGNT). Russian business media almost always put a brand name like this in
# quotes ("Самолет" отчитался...), so for these specific words a quoted
# mention is required instead of a bare word-boundary match.
_AMBIGUOUS_BARE_WORDS = {"самолет", "позитив", "система", "магнит", "лента", "пик", "полюс"}
_QUOTE = r"[«»\"'“”‘’]"


def _word_pattern(word: str) -> str:
    """Builds a regex fragment for one word that also matches Russian case endings.

    Plain substring/exact matching would miss declined forms ("Норникеля",
    "Татнефти") since Russian case endings change the last letter (nominative
    "россети" vs genitive "россетей" etc.). For Cyrillic words longer than 5
    characters, the pattern matches on the stem (word minus its last letter)
    followed by \\w* - only 1 letter is dropped, not more, because a shorter
    stem starts matching unrelated words entirely (e.g. dropping 2 letters off
    "россети" left a "россе" stem that also matched "Россельхознадзор").
    Short abbreviations (e.g. "вк", "пик") and non-Cyrillic brand names (e.g.
    "ozon") are matched exactly instead, since stemming those would be even
    more prone to false positives.
    """
    if len(word) > 5 and _CYRILLIC_WORD.match(word):
        return re.escape(word[:-1]) + r"\w*"
    return re.escape(word)


def _alias_pattern(alias: str) -> str:
    words = alias.strip().split()
    core = r"\s+".join(_word_pattern(w) for w in words)
    if len(words) == 1 and words[0].lower() in _AMBIGUOUS_BARE_WORDS:
        return f"{_QUOTE}{core}{_QUOTE}"
    return r"\b" + core + r"\b"


class TickerLinker:
    """Matches free-text news to MOEX tickers via a curated alias table."""

    def __init__(self, aliases: dict[str, list[str]] | None = None, universe: list[str] | None = None):
        aliases = aliases or ALIASES
        universe = universe or TICKERS_100
        # Only keep aliases for tickers actually in the tracked universe, and
        # compile one case-insensitive regex per ticker for fast matching.
        # Each alternative already carries its own boundary (word-boundary or
        # quote-anchored - see _alias_pattern), so no extra wrapping here.
        self._patterns: dict[str, re.Pattern] = {}
        for ticker, names in aliases.items():
            if ticker not in universe:
                continue
            patterns = [_alias_pattern(n) for n in names if n.strip()]
            if not patterns:
                continue
            self._patterns[ticker] = re.compile("(?:%s)" % "|".join(patterns), re.IGNORECASE)

    def match(self, text: str) -> list[str]:
        """Returns the tickers whose aliases appear in `text`, sorted for determinism."""
        hits = [ticker for ticker, pattern in self._patterns.items() if pattern.search(text)]
        return sorted(hits)


if __name__ == "__main__":
    linker = TickerLinker()
    samples = [
        "Сбербанк отчитался о рекордной прибыли по итогам квартала",
        "Акции Норникеля упали на фоне снижения цен на никель",
        "ФосАгро увеличила экспорт удобрений, а Яндекс запустил новый сервис",
        "Совет директоров X5 рекомендовал дивиденды",
        "Погода в Москве: снег и гололедица",
    ]
    for text in samples:
        print(f"{linker.match(text)}  <-  {text}")
