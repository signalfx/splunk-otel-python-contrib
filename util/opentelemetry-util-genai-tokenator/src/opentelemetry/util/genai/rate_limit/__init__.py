"""Tokenator - Predictive rate limit monitoring for GenAI agentic applications.

"I'll be back... before you hit your rate limit."
"""

from opentelemetry.util.genai.rate_limit.emitter import (
    RateLimitPredictorEmitter,
    load_emitters,
)
from opentelemetry.util.genai.rate_limit.predictor import (
    RateLimitPrediction,
    RateLimitPredictor,
    WorkflowPrediction,
)
from opentelemetry.util.genai.rate_limit.providers.base import (
    ModelLimits,
    RateLimitProvider,
)
from opentelemetry.util.genai.rate_limit.providers.mockopenai import (
    MockOpenAIRateLimitProvider,
)
from opentelemetry.util.genai.rate_limit.providers.openai import (
    OpenAIRateLimitProvider,
)
from opentelemetry.util.genai.rate_limit.tracker import TokenTracker
from opentelemetry.util.genai.rate_limit.version import __version__

__all__ = [
    "__version__",
    "RateLimitPredictorEmitter",
    "load_emitters",
    "TokenTracker",
    "RateLimitPredictor",
    "RateLimitPrediction",
    "WorkflowPrediction",
    "RateLimitProvider",
    "ModelLimits",
    "OpenAIRateLimitProvider",
    "MockOpenAIRateLimitProvider",
]
