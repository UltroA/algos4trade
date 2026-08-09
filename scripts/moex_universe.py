"""
Универсум из 100 ликвидных инструментов MOEX для кросс-секционной валидации
бенчмарка (см. scripts/run_benchmarks_wide.py).

Список получен из публичного MOEX ISS API (не требует токена) - тикеры,
входящие в индекс МосБиржи широкого рынка (MOEXBMI) на дату генерации:

GET https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/MOEXBMI/tickers.json

Взяты тикеры последнего (максимального) периода действия ("till") в ответе
API - то есть текущий состав индекса на 2026-08-08. MOEXBMI выбран потому,
что явно недостаточная представленность инструментов (10 акций) была
задокументированным ограничением исходного бенчмарка: HDBSCAN не находил
кластеров, а обобщение single-asset результатов требовало "гораздо более
широкого универсума (сотни бумаг)".

Обновить список: запустить `python scripts/moex_universe.py`.
"""

TICKERS_100 = [
    "AFKS", "AFLT", "AKRN", "ALRS", "APTK", "AQUA", "ASTR", "BANEP", "BAZA", "BELU",
    "BSPB", "CBOM", "CHMF", "CNRU", "DATA", "DIAS", "DOMRF", "ELFV", "ENPG", "ETLN",
    "EUTR", "FEES", "FESH", "FIXR", "FLOT", "GAZP", "GEMC", "GLRX", "GMKN", "HEAD",
    "HNFG", "HYDR", "IRAO", "LENT", "LKOH", "LSNGP", "LSRG", "MAGN", "MBNK", "MDMG",
    "MGNT", "MOEX", "MRKC", "MRKP", "MRKU", "MRKV", "MSNG", "MSRS", "MTLR", "MTLRP",
    "MTSS", "NKHP", "NKNCP", "NLMK", "NMTP", "NVTK", "OGKB", "OZON", "OZPH", "PHOR",
    "PIKK", "PLZL", "POSI", "PRMD", "RAGR", "RASP", "RENI", "RNFT", "ROSN", "RTKM",
    "RTKMP", "RUAL", "SBER", "SBERP", "SELG", "SFIN", "SGZH", "SMLT", "SNGS", "SNGSP",
    "SOFL", "SPBE", "SVAV", "SVCB", "T", "TATN", "TATNP", "TGKA", "TRMK", "TRNFP",
    "UGLD", "UPRO", "UWGN", "VKCO", "VSEH", "VSMO", "VTBR", "WUSH", "X5", "YDEX",
]

assert len(TICKERS_100) == 100, len(TICKERS_100)


def fetch_current_moexbmi_tickers() -> list[str]:
    """Перегенерировать список из MOEX ISS API (публичный, без токена)."""
    import requests

    url = "https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/MOEXBMI/tickers.json"
    data = requests.get(url, timeout=15).json()["tickers"]["data"]
    max_till = max(row[2] for row in data)
    return sorted({row[0] for row in data if row[2] == max_till})


if __name__ == "__main__":
    fresh = fetch_current_moexbmi_tickers()
    print(f"{len(fresh)} tickers")
    print(fresh)
    if set(fresh) != set(TICKERS_100):
        print("DIFFERS from TICKERS_100 baked into this file - consider updating it.")
