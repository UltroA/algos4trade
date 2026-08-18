"""
Sanity checks on core.algorithm_discovery.discover_algorithms() itself -
the machinery tests/test_algorithm_smoke.py and every "run everything"
script (run_all_benchmarks.py, run_market_simulation.py, web_configurator.py)
depend on.
"""

from core.algorithm_discovery import discover_algorithms
from core.base import BaseTradingAlgorithm


def test_discovery_finds_the_documented_algorithm_count():
    found, skipped, failed = discover_algorithms()
    # README.md's catalogue table (section 6) lists 31 algorithms; discovery
    # additionally finds the reference baselines (SMA crossover, buy-and-hold,
    # random +-1) and both news-sentiment variants - see CLAUDE.md's "as of
    # 2026-08 this discovers 36 classes". A hard equality would make this
    # test brittle against every new algorithm file, so it's a floor instead.
    assert len(found) >= 34, f"expected at least 34 auto-discoverable algorithms, found {len(found)}: {found}"
    # The 4 composite.py wrapper classes need a base-algorithm factory argument
    # and are correctly excluded from `found` - see core/algorithms/composite.py.
    assert len(skipped) >= 4, f"expected at least 4 skipped wrapper classes, found {len(skipped)}: {skipped}"


def test_every_found_class_is_a_concrete_base_trading_algorithm_subclass():
    found, _, _ = discover_algorithms()
    import importlib

    for modname, clsname in found:
        module = importlib.import_module(f"algorithms.{modname}")
        cls = getattr(module, clsname)
        assert issubclass(cls, BaseTradingAlgorithm)
        instance = cls()  # must not require arguments - that's the discovery rule itself
        assert isinstance(instance, BaseTradingAlgorithm)


def test_failed_imports_are_reported_not_raised():
    # discover_algorithms() must never let one unimportable module (e.g. a
    # missing optional "news" extra dependency) crash the whole scan - see
    # its docstring and algorithms/news_sentiment_memory.py's deferred import.
    found, skipped, failed = discover_algorithms()
    assert isinstance(failed, list)
    for modname, error in failed:
        assert isinstance(modname, str) and isinstance(error, str)
