# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT

Enable capture of GenAI message content. When set to a truthy value (``true``,
``1``, ``yes``, ``on``) the emitter pipeline records prompt and completion
content according to :envvar:`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE`.
Unset or falsey values disable content capture entirely.
"""

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE

Controls where captured message content is emitted. Accepted case-insensitive values:

``SPAN_ONLY`` – emit content on spans only.
``EVENT_ONLY`` – emit content as log events only.
``SPAN_AND_EVENT`` (default) – emit content on spans and as log events.
``NONE`` – disable content emission (equivalent to disabling capture entirely).
"""


OTEL_INSTRUMENTATION_GENAI_UPLOAD_HOOK = (
    "OTEL_INSTRUMENTATION_GENAI_UPLOAD_HOOK"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_UPLOAD_HOOK
"""

OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH = (
    "OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH

An :func:`fsspec.open` compatible URI/path for uploading prompts and responses. Can be a local
path like ``/path/to/prompts`` or a cloud storage URI such as ``gs://my_bucket``. For more
information, see

* `Instantiate a file-system
  <https://filesystem-spec.readthedocs.io/en/latest/usage.html#instantiate-a-file-system>`_ for supported values and how to
  install support for additional backend implementations.
* `Configuration
  <https://filesystem-spec.readthedocs.io/en/latest/features.html#configuration>`_ for
  configuring a backend with environment variables.
*  `URL Chaining
   <https://filesystem-spec.readthedocs.io/en/latest/features.html#url-chaining>`_ for advanced
   use cases.
"""

# ---- Evaluation configuration ----
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS = (
    "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS

Comma-separated list describing evaluator configuration. Each entry selects an evaluator
registered under the ``opentelemetry_util_genai_evaluators`` entry-point group. Optional
per-type overrides may be supplied using the syntax::

    EvaluatorName(TypeName(metric,metric2(config=value)))

Examples::

    Deepeval
    Deepeval,NLTK
    Deepeval(LLMInvocation(bias,toxicity))
    Deepeval(LLMInvocation(bias(threshold=1),toxicity))

If no configuration is provided, each evaluator defaults to its declared metric set per
GenAI invocation type.
"""

OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION = (
    "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION

When set to ``true``/``1``/``yes`` aggregate results from all evaluators for a sampled
invocation into a single list before forwarding to the handler. Otherwise, results are
forwarded per-evaluator.
"""

OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL = (
    "OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL

Polling interval (seconds) for the evaluation worker loop. Defaults to ``5.0`` seconds.
"""

OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS = (
    "OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS

Comma-separated list of completion callback entry point names to enable. When unset,
every available callback exposed via the ``opentelemetry_util_genai_completion_callbacks`` group
is loaded. Names are compared case-insensitively.
"""

OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS = (
    "OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS

Disable automatic loading of default completion callbacks (evaluators, persistence hooks, etc.).
Truthiness is evaluated using the standard ``1/true/yes/on`` parsing. When set, no completion callbacks
are registered automatically.
"""

OTEL_INSTRUMENTATION_GENAI_EMITTERS = "OTEL_INSTRUMENTATION_GENAI_EMITTERS"
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_EMITTERS

Comma-separated list of generators names to run (e.g. ``span,traceloop_compat``).

Select telemetry flavor (composed emitters). Accepted baseline values (case-insensitive):

* ``span`` (default) - spans only
* ``span_metric`` - spans + metrics
* ``span_metric_event`` - spans + metrics + content events

Additional extender emitters:
* ``traceloop_compat`` - adds a Traceloop-compatible LLM span (requires installing ``opentelemetry-util-genai-emitters-traceloop``). If specified *alone*, only the compat span is emitted. If combined (e.g. ``span,traceloop_compat``) both semconv and compat spans are produced.

Invalid or unset values fallback to ``span``.
"""

OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN = (
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN"
)
OTEL_INSTRUMENTATION_GENAI_EMITTERS_METRICS = (
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS_METRICS"
)
OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS = (
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS"
)
OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION = (
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_EMITTERS_<CATEGORY>

Optional category-specific overrides applied after builtin and entry-point emitters
are registered. Accepts comma-separated emitter names with optional directives such
as ``replace:`` (replace entire category) or ``append:``/``prepend:`` (explicit
positioning). Categories: ``SPAN``, ``METRICS``, ``CONTENT_EVENTS``, ``EVALUATION``.
"""

OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE = (
    "OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE"
)
OTEL_GENAI_EVALUATION_EVENT_LEGACY = "OTEL_GENAI_EVALUATION_EVENT_LEGACY"

__all__ = [
    # existing
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE",
    "OTEL_INSTRUMENTATION_GENAI_UPLOAD_HOOK",
    "OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH",
    # evaluation
    "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS",
    "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION",
    "OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL",
    # generator selection
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS",
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN",
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS_METRICS",
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS",
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION",
    "OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE",
    "OTEL_GENAI_EVALUATION_EVENT_LEGACY",
    "OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS",
    "OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS",
]
