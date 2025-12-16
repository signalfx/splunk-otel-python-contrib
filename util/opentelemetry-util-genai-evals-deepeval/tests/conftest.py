# Ensure the local src/ path for opentelemetry.util.genai development version is importable
import sys
from pathlib import Path

import pytest  # type: ignore[import]

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


@pytest.fixture(autouse=True)
def _patch_entry_points(monkeypatch):
    # Avoid enumerating full environment entry points during tests, which can be slow.
    class _EmptyEntryPoints(list):
        def select(self, **kwargs):  # type: ignore[override]
            return []

    empty_eps = _EmptyEntryPoints()

    monkeypatch.setattr(
        "opentelemetry.util.evaluator.deepeval_models.metadata.entry_points",
        lambda: empty_eps,
    )
    yield
