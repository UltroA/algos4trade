"""
Auto-discovery of benchmarkable algorithm classes in src/algorithms/.

Shared by scripts/run_all_benchmarks.py and scripts/run_market_simulation.py
so both "run everything in the folder" entry points use the exact same
instantiability rule instead of duplicating the reflection logic.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path


def _is_instantiable(cls: type) -> bool:
    """True if cls() can be called with no arguments (all __init__ parameters
    other than self/*args/**kwargs have default values)."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is param.empty:
            return False
    return True


def discover_algorithms() -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Scans src/algorithms/*.py and returns (found, skipped, failed) - lists
    of (module_name, class_name) for found/skipped, sorted by module name.

    Classes that need another algorithm to be constructed (wrapper classes
    in src/algorithms/composite.py, whose first required argument is a
    *_factory) are reported in `skipped` rather than instantiated - they
    cannot be meaningfully run "by default".

    A module that fails to import at all is reported in `failed` as
    (module_name, error_message) instead of raising - the only expected
    cause is a missing optional dependency (e.g. algorithms/news_sentiment_
    memory.py needs the "news" extra: openai/pydantic via core.llm_sentiment,
    feedparser via core.news_feed's NewsItem). A base install (`pip install
    -e .`, no `[news]` extra) must still be able to discover and run every
    other algorithm - one unimportable module should not take down the
    whole scan.
    """
    import algorithms as algorithms_pkg
    from core.base import BaseTradingAlgorithm

    found, skipped, failed = [], [], []
    algo_dir = Path(algorithms_pkg.__file__).parent
    module_names = sorted(m.name for m in pkgutil.iter_modules([str(algo_dir)]))
    for modname in module_names:
        try:
            module = importlib.import_module(f"algorithms.{modname}")
        except ImportError as exc:
            failed.append((modname, str(exc)))
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue  # only classes defined directly in this file
            if not issubclass(cls, BaseTradingAlgorithm) or cls is BaseTradingAlgorithm:
                continue
            if inspect.isabstract(cls):
                continue
            (found if _is_instantiable(cls) else skipped).append((modname, cls.__name__))
    return found, skipped, failed
