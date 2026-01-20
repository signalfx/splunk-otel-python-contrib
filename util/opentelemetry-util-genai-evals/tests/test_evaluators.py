import asyncio
import importlib
import os
import unittest
from unittest.mock import patch

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS,
    OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION,
)
from opentelemetry.util.genai.evals.base import Evaluator
from opentelemetry.util.genai.evals.bootstrap import (
    EvaluatorCompletionCallback,
)
from opentelemetry.util.genai.evals.manager import Manager
from opentelemetry.util.genai.evals.registry import (
    clear_registry,
    register_evaluator,
)
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    EvaluationResult,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)


def _reload_builtin_evaluators() -> None:
    from opentelemetry.util.genai.evals import builtins as builtin_module

    importlib.reload(builtin_module)


def _mock_load_callbacks(_selected):
    callback = EvaluatorCompletionCallback()
    return [("evaluations", callback)], {"evaluations"}


class _RecordingHandler:
    def __init__(self) -> None:
        self.observations: list[list[EvaluationResult]] = []

    def evaluation_results(
        self, invocation: LLMInvocation, results: list[EvaluationResult]
    ) -> None:
        self.observations.append(list(results))


class _StaticEvaluator(Evaluator):
    def __init__(
        self,
        metrics=None,
        *,
        invocation_type: str | None = None,
        options=None,
    ) -> None:
        super().__init__(
            metrics, invocation_type=invocation_type, options=options
        )

    def default_metrics(self):  # pragma: no cover - trivial
        return ("static_metric",)

    def evaluate_llm(
        self, invocation: LLMInvocation
    ) -> list[EvaluationResult]:  # pragma: no cover - trivial
        results: list[EvaluationResult] = []
        for metric in self.metrics:
            opts = self.options.get(metric, {})
            results.append(
                EvaluationResult(
                    metric_name=metric,
                    score=1.0,
                    label="ok",
                    explanation="static evaluator result",
                    attributes={"options": opts},
                )
            )
        return results


class TestManagerConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        if hasattr(get_telemetry_handler, "_default_handler"):
            delattr(get_telemetry_handler, "_default_handler")
        clear_registry()
        _reload_builtin_evaluators()
        register_evaluator(
            "Static",
            lambda metrics=None,
            invocation_type=None,
            options=None: _StaticEvaluator(
                metrics,
                invocation_type=invocation_type,
                options=options,
            ),
            default_metrics=lambda: {"LLMInvocation": ("static_metric",)},
        )

    def tearDown(self) -> None:  # pragma: no cover - defensive
        clear_registry()
        _reload_builtin_evaluators()

    def _build_invocation(self) -> LLMInvocation:
        invocation = LLMInvocation(request_model="m1")
        invocation.input_messages.append(
            InputMessage(role="user", parts=[Text(content="hi")])
        )
        invocation.output_messages.append(
            OutputMessage(
                role="assistant",
                parts=[Text(content="hello")],
                finish_reason="stop",
            )
        )
        return invocation

    @patch.dict(
        os.environ,
        {
            OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS: "Static",
            OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION: "true",
        },
        clear=True,
    )
    def test_manager_runs_default_metrics(self) -> None:
        handler = _RecordingHandler()
        manager = Manager(handler)
        invocation = self._build_invocation()
        results = manager.evaluate_now(invocation)
        manager.shutdown()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metric_name, "static_metric")
        self.assertEqual(len(handler.observations), 1)
        self.assertEqual(
            handler.observations[0][0].metric_name, "static_metric"
        )

    @patch.dict(
        os.environ,
        {
            OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS: (
                "Static(LLMInvocation(metric_one(threshold=0.5),metric_two))"
            )
        },
        clear=True,
    )
    def test_manager_parses_metric_options(self) -> None:
        handler = _RecordingHandler()
        manager = Manager(handler)
        invocation = self._build_invocation()
        results = manager.evaluate_now(invocation)
        manager.shutdown()
        metric_names = {result.metric_name for result in results}
        self.assertEqual(metric_names, {"metric_one", "metric_two"})
        options = {
            result.metric_name: result.attributes.get("options")
            for result in results
        }
        self.assertEqual(options["metric_one"].get("threshold"), "0.5")
        self.assertFalse(options["metric_two"])

    @patch.dict(os.environ, {}, clear=True)
    def test_manager_auto_discovers_defaults(self) -> None:
        with (
            patch(
                "opentelemetry.util.genai.evals.manager.list_evaluators",
                return_value=["Static"],
            ),
            patch(
                "opentelemetry.util.genai.evals.manager.get_default_metrics",
                return_value={"LLMInvocation": ("static_metric",)},
            ),
        ):
            handler = _RecordingHandler()
            manager = Manager(handler)
            try:
                self.assertTrue(manager.has_evaluators)
            finally:
                manager.shutdown()

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS: "none"},
        clear=True,
    )
    def test_manager_respects_none(self) -> None:
        handler = _RecordingHandler()
        manager = Manager(handler)
        try:
            self.assertFalse(manager.has_evaluators)
        finally:
            manager.shutdown()


class TestHandlerIntegration(unittest.TestCase):
    def setUp(self) -> None:
        if hasattr(get_telemetry_handler, "_default_handler"):
            delattr(get_telemetry_handler, "_default_handler")
        clear_registry()
        _reload_builtin_evaluators()
        register_evaluator(
            "Static",
            lambda metrics=None,
            invocation_type=None,
            options=None: _StaticEvaluator(
                metrics,
                invocation_type=invocation_type,
                options=options,
            ),
            default_metrics=lambda: {"LLMInvocation": ("static_metric",)},
        )

    def tearDown(self) -> None:  # pragma: no cover - defensive
        clear_registry()
        _reload_builtin_evaluators()

    def _build_invocation(self) -> LLMInvocation:
        invocation = LLMInvocation(request_model="m2")
        invocation.input_messages.append(
            InputMessage(role="user", parts=[Text(content="hi")])
        )
        invocation.output_messages.append(
            OutputMessage(
                role="assistant",
                parts=[Text(content="hello")],
                finish_reason="stop",
            )
        )
        return invocation

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS: "Static"},
        clear=True,
    )
    def test_handler_registers_manager(self) -> None:
        with patch(
            "opentelemetry.util.genai.utils.load_completion_callbacks",
            side_effect=_mock_load_callbacks,
        ):
            handler = get_telemetry_handler()
            invocation = self._build_invocation()
            handler.start_llm(invocation)
            invocation.output_messages = invocation.output_messages
            handler.stop_llm(invocation)
            handler.wait_for_evaluations(2.0)
            manager = getattr(handler, "_evaluation_manager", None)
            self.assertIsNotNone(manager)
            self.assertTrue(
                invocation.attributes.get("gen_ai.evaluation.executed")
            )
            manager.shutdown()

    @patch.dict(
        os.environ,
        {
            OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS: "Static",
            OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION: "false",
        },
        clear=True,
    )
    def test_handler_evaluate_llm_returns_results(self) -> None:
        with patch(
            "opentelemetry.util.genai.utils.load_completion_callbacks",
            side_effect=_mock_load_callbacks,
        ):
            handler = get_telemetry_handler()
            invocation = self._build_invocation()
            results = handler.evaluate_llm(invocation)
            manager = getattr(handler, "_evaluation_manager", None)
            if manager is not None:
                manager.shutdown()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metric_name, "static_metric")

    @patch.dict(os.environ, {}, clear=True)
    def test_handler_auto_enables_when_env_missing(self) -> None:
        with (
            patch(
                "opentelemetry.util.genai.evals.manager.list_evaluators",
                return_value=["Static"],
            ),
            patch(
                "opentelemetry.util.genai.evals.manager.get_default_metrics",
                return_value={"LLMInvocation": ("static_metric",)},
            ),
            patch(
                "opentelemetry.util.genai.utils.load_completion_callbacks",
                side_effect=_mock_load_callbacks,
            ),
        ):
            handler = get_telemetry_handler()
            manager = getattr(handler, "_evaluation_manager", None)
            self.assertIsNotNone(manager)
            self.assertTrue(manager.has_evaluators)  # type: ignore[union-attr]
            if manager is not None:
                manager.shutdown()

    @patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS: "none"},
        clear=True,
    )
    def test_handler_disables_when_none(self) -> None:
        with patch(
            "opentelemetry.util.genai.utils.load_completion_callbacks",
            side_effect=_mock_load_callbacks,
        ):
            handler = get_telemetry_handler()
            manager = getattr(handler, "_evaluation_manager", None)
            if manager is not None:
                manager.shutdown()
            self.assertIsNone(manager)


# =============================================================================
# Async Evaluator Tests
# =============================================================================


class _AsyncStaticEvaluator(Evaluator):
    """Static evaluator with native async support for testing."""

    def __init__(
        self,
        metrics=None,
        *,
        invocation_type: str | None = None,
        options=None,
    ) -> None:
        super().__init__(
            metrics, invocation_type=invocation_type, options=options
        )

    def default_metrics(self):
        return ("async_metric",)

    def evaluate_llm(
        self, invocation: LLMInvocation
    ) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                metric_name="async_metric",
                score=1.0,
                label="sync",
                explanation="sync evaluation",
            )
        ]

    async def evaluate_llm_async(
        self, invocation: LLMInvocation
    ) -> list[EvaluationResult]:
        # Simulate async work
        await asyncio.sleep(0.01)
        return [
            EvaluationResult(
                metric_name="async_metric",
                score=1.0,
                label="async",
                explanation="async evaluation",
            )
        ]


class TestAsyncEvaluator(unittest.TestCase):
    """Tests for async evaluation support in base Evaluator."""

    def _build_invocation(self) -> LLMInvocation:
        invocation = LLMInvocation(request_model="async-test")
        invocation.input_messages.append(
            InputMessage(role="user", parts=[Text(content="hello")])
        )
        invocation.output_messages.append(
            OutputMessage(
                role="assistant",
                parts=[Text(content="hi")],
                finish_reason="stop",
            )
        )
        return invocation

    def _build_agent_invocation(self) -> AgentInvocation:
        invocation = AgentInvocation(name="test-agent")
        invocation.operation = "invoke_agent"
        return invocation

    def test_base_evaluator_has_async_methods(self) -> None:
        """Base Evaluator class has async evaluation methods."""
        evaluator = _StaticEvaluator()
        self.assertTrue(hasattr(evaluator, "evaluate_async"))
        self.assertTrue(hasattr(evaluator, "evaluate_llm_async"))
        self.assertTrue(hasattr(evaluator, "evaluate_agent_async"))
        self.assertTrue(asyncio.iscoroutinefunction(evaluator.evaluate_async))

    def test_evaluate_async_delegates_to_sync_by_default(self) -> None:
        """Default evaluate_async runs sync evaluate in thread pool."""
        evaluator = _StaticEvaluator()
        invocation = self._build_invocation()

        async def run_test():
            results = await evaluator.evaluate_async(invocation)
            return results

        results = asyncio.run(run_test())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metric_name, "static_metric")

    def test_async_evaluator_uses_native_async(self) -> None:
        """Evaluator with native async uses async methods."""
        evaluator = _AsyncStaticEvaluator()
        invocation = self._build_invocation()

        async def run_test():
            results = await evaluator.evaluate_async(invocation)
            return results

        results = asyncio.run(run_test())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].label, "async")
        self.assertEqual(results[0].explanation, "async evaluation")

    def test_sync_and_async_produce_same_metric(self) -> None:
        """Sync and async evaluation produce same metric name."""
        evaluator = _AsyncStaticEvaluator()
        invocation = self._build_invocation()

        sync_results = evaluator.evaluate(invocation)
        async_results = asyncio.run(evaluator.evaluate_async(invocation))

        self.assertEqual(
            sync_results[0].metric_name, async_results[0].metric_name
        )

    def test_evaluate_async_handles_agent_invocation(self) -> None:
        """Async evaluation works for agent invocations."""
        evaluator = _StaticEvaluator()
        invocation = self._build_agent_invocation()

        async def run_test():
            # StaticEvaluator only implements evaluate_llm, so agent returns empty
            results = await evaluator.evaluate_async(invocation)
            return results

        results = asyncio.run(run_test())
        self.assertEqual(results, [])

    def test_evaluate_async_returns_empty_for_unsupported_type(self) -> None:
        """Async evaluation returns empty for unsupported invocation types."""
        evaluator = _StaticEvaluator()

        async def run_test():
            # Pass an arbitrary object
            results = await evaluator.evaluate_async(object())  # type: ignore
            return results

        results = asyncio.run(run_test())
        self.assertEqual(results, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
