import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
_src = _pkg_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Ensure the base deepeval evaluator package is importable for shared helpers.
_base = (
    Path(__file__).resolve().parents[2]
    / "opentelemetry-util-genai-evals-deepeval"
    / "src"
)
if _base.exists() and str(_base) not in sys.path:
    sys.path.insert(0, str(_base))
