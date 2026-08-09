"""
Universe of 100 liquid MOEX instruments for cross-sectional validation
of the benchmark (see scripts/run_benchmarks_wide.py).

The list is obtained from the public MOEX ISS API (no token required) -
tickers included in the MOEX Broad Market Index (MOEXBMI) as of the
generation date:

GET https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/MOEXBMI/tickers.json

Tickers are taken from the latest (maximum) validity period ("till") in the
API response - i.e. the current index composition as of 2026-08-08. MOEXBMI
was chosen because insufficient instrument coverage (10 stocks) was a
documented limitation of the original benchmark: HDBSCAN could not find any
clusters, and generalizing single-asset results required "a much wider
universe (hundreds of securities)".

To update the list: run `python scripts/moex_universe.py`.
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
    """Regenerate the list from the MOEX ISS API (public, no token required)."""
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
