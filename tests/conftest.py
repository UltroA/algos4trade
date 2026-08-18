"""
Makes `core`/`algorithms` importable under pytest regardless of install
method: a fallback for `pip install -r requirements.txt` (no editable
install, no `src/` on sys.path) alongside `pip install -e ".[dev]"` (already
on sys.path via the package's own .pth file, where this is a harmless no-op).
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
