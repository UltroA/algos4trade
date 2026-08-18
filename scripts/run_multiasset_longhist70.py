"""
Calendar-aligned control run for the three multi-asset algorithms (HDBSCAN
clustering, correlation clustering, Thompson sampling allocator) on the
70-ticker subset of MOEXBMI with listing history back to the start of 2019
(the same LONGHIST_70 set used by run_composite_benchmarks.py's
Thompson-with-strong-arms comparison).

Why this script exists: run_benchmarks_wide.py's naive multi-asset pass
intersects the per-ticker chronological train splits across all 97
MOEXBMI tickers, which is empty when the universe mixes long-listed and
recently-IPO'd names (see docs/research/3-wide-validation.typ, "calendar
heterogeneity" pitfall) - both clustering algorithms correctly return
is_fitted = False and a zero signal in that case. This control run
restricts the universe to tickers whose training window actually overlaps
in calendar time, isolating the effect of universe width from the effect
of calendar mismatch.

Run: source .venv/bin/activate && python scripts/run_multiasset_longhist70.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from core.backtester import Backtester
from core.providers.tinvest import TInvestDataLoader

from algorithms.hdbscan_clustering import HDBSCANPairsClustering
from algorithms.correlation_clustering import CorrelationClustering
from algorithms.thompson_bandits import ThompsonSamplingAllocator

START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)
MIN_ROWS = 300

LONGHIST_70 = set("""AFKS AFLT AKRN ALRS APTK AQUA BANEP BELU BSPB CBOM CHMF ELFV ENPG ETLN FEES FESH
GAZP GMKN HYDR IRAO LKOH LSNGP LSRG MAGN MGNT MOEX MRKC MRKP MRKU MRKV MSNG MSRS MTLR MTLRP MTSS
NKHP NKNCP NLMK NMTP NVTK OGKB PHOR PIKK PLZL RAGR RASP RNFT ROSN RTKM RTKMP RUAL SBER SBERP SELG
SFIN SNGS SNGSP SVAV T TATN TATNP TGKA TRMK TRNFP UPRO UWGN VSMO VTBR X5 YDEX""".split())

ALGORITHMS = {
    "hdbscan_clustering": HDBSCANPairsClustering,
    "correlation_clustering": CorrelationClustering,
    "thompson_bandits": ThompsonSamplingAllocator,
}


def main() -> None:
    loader = TInvestDataLoader()
    data = {}
    for ticker in sorted(LONGHIST_70):
        try:
            df = loader.load_candles(ticker, START, END, interval="day", use_cache=True)
        except Exception as exc:
            print(f"  [longhist70] {ticker}: load failed - {type(exc).__name__}: {exc}")
            continue
        if len(df) >= MIN_ROWS:
            data[ticker] = df
    print(f"Universe ready: {len(data)}/{len(LONGHIST_70)} tickers usable\n")

    backtester = Backtester(transaction_cost_bps=5.0)
    for spec_name, cls in ALGORITHMS.items():
        result = backtester.run(cls(), data)
        status = (
            "ERROR: " + result.error
            if result.error
            else f"sharpe={result.metrics.sharpe_ratio:.3f}, n_trades={result.metrics.n_trades}"
        )
        print(f"[longhist70] {spec_name} on {len(data)} tickers -> {status}")


if __name__ == "__main__":
    main()
