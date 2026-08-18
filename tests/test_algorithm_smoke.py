"""
Wraps every algorithm file's own `if __name__ == "__main__":` smoke test
(synthetic data, fit() + generate_signals(), prints the resulting signal
distribution) in pytest, parametrized over every class
core.algorithm_discovery.discover_algorithms() finds - so "does every
algorithm still fit and produce a valid signal" is checked on every push via
CI (.github/workflows/ci.yml), not just by whoever happens to run a file by
hand.

Each smoke test runs as its own subprocess (`python src/algorithms/<file>.py`),
matching how run_all_benchmarks.py and the live simulator already isolate
each algorithm - mixing lightgbm/xgboost/catboost/torch/hmmlearn/hdbscan's
native runtimes in one interpreter reliably segfaults once enough distinct
libraries have been used (see CLAUDE.md's native-library pitfalls); running
32 of them in a single pytest process would hit exactly that.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.algorithm_discovery import discover_algorithms

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ALGORITHMS_DIR = _REPO_ROOT / "src" / "algorithms"
_TIMEOUT_SECONDS = 300

_found, _skipped, _failed = discover_algorithms()
_MODULE_NAMES = sorted({modname for modname, _ in _found})


@pytest.mark.parametrize("modname", _MODULE_NAMES)
def test_algorithm_smoke_test_runs(modname):
    script = _ALGORITHMS_DIR / f"{modname}.py"
    assert script.exists(), f"discover_algorithms() found {modname!r} but {script} does not exist"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, (
        f"{modname}.py smoke test exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
    assert result.stdout.strip(), f"{modname}.py smoke test produced no output"


def test_failed_import_modules_are_reported_for_visibility(capsys):
    # Not a failure by itself (see test_algorithm_discovery.py) - this just
    # makes a missing optional dependency visible in CI output rather than
    # silent, since it means fewer algorithms actually got smoke-tested above.
    if _failed:
        print(f"\nNote: {len(_failed)} module(s) failed to import and were not smoke-tested:")
        for modname, error in _failed:
            print(f"  - {modname}: {error}")
