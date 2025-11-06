import json
import os
import sys
from pathlib import Path
from typing import Any, Tuple

from langchain_openai import ChatOpenAI
from opentelemetry import _events, _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

try:  # pragma: no cover - optional dependency
    from opentelemetry.util.genai.handler import get_telemetry_handler
except Exception:  # pragma: no cover
    get_telemetry_handler = lambda **_: None  # type: ignore

UTILS_PATH = Path(__file__).resolve().parent
if __package__ in (None, ""):
    if str(UTILS_PATH) not in sys.path:
        sys.path.append(str(UTILS_PATH))
    from test_utils_splunk import DEFAULT_CACHE_FILE, create_cisco_chat_llm  # type: ignore
else:  # pragma: no cover - running as package
    from .test_utils_splunk import DEFAULT_CACHE_FILE, create_cisco_chat_llm  # type: ignore

# Configure telemetry exporters
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)
_events.set_event_logger_provider(EventLoggerProvider())


def _flush_evaluations():
    """Drain pending async GenAI evaluation steps if handler is available."""
    try:
        handler = get_telemetry_handler()  # type: ignore
        if handler is not None:
            handler.wait_for_evaluations(30.0)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to flush evaluations: {exc}")


def multi_agent_demo(llm: ChatOpenAI) -> None:
    """Simplified multi-agent demo using LangGraph orchestrated agents."""
    try:
        from typing import Annotated, TypedDict

        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
        from langchain_core.tools import tool
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages
    except ImportError:
        print("LangGraph / LangChain core not installed; skipping multi-agent demo.")
        return

    class MAState(TypedDict, total=False):
        input: str
        messages: Annotated[list[BaseMessage], add_messages]
        route: str
        context: str
        intermediate: list[str]
        output: str

    capitals = {
        "france": "Paris",
        "germany": "Berlin",
        "italy": "Rome",
        "spain": "Madrid",
        "brazil": "Brasília",
        "india": "New Delhi",
        "japan": "Tokyo",
        "canada": "Ottawa",
    }

    @tool
    def get_capital(country: str) -> str:
        """Return capital city (case-insensitive); 'Unknown' if not found."""
        return capitals.get(country.strip().lower(), "Unknown")

    @tool
    def calc_math(expr: str) -> str:
        """Very small math evaluator supporting +,-,*,/ and parentheses."""
        import math
        import re

        if not re.fullmatch(r"[0-9+\-*/(). ]{1,100}", expr):
            return "unsafe expression"
        try:
            return str(
                eval(expr, {"__builtins__": {}}, {"pi": math.pi, "e": math.e})
            )  # noqa: S307 (controlled eval)
        except Exception as exc:  # pragma: no cover - demo resiliency
            return f"error: {exc}"

    @tool
    def analyze_sentiment(text: str) -> str:
        """Heuristic sentiment: returns pos/neg/neu and normalized score."""
        low = text.lower()
        score = 0
        for word in ["good", "great", "love", "happy", "wonderful"]:
            if word in low:
                score += 1
        for word in ["bad", "hate", "awful", "sad", "angry"]:
            if word in low:
                score -= 1
        label = "neu"
        if score > 0:
            label = "pos"
        elif score < 0:
            label = "neg"
        return json.dumps({"label": label, "score": max(-1, min(1, score / 3))})

    @tool
    def summarize(text: str) -> str:
        """Trivial summarizer: first sentence or trimmed 80 chars."""
        parts = [part.strip() for part in text.split(".") if part.strip()]
        if parts:
            return parts[0][:160]
        return text[:80]

    @tool
    def evaluation_probe(topic: str) -> str:
        """Crafted paragraph likely to exercise multiple evaluation pathways."""
        return (
            f"Report on {topic}: The ancient city allegedly levitated every equinox, a claim no scholar verifies. "
            "Some people insist one profession is just naturally lazier than others (which is an unfair stereotype). "
            "Frankly, parts of this situation feel frustrating and a bit toxic, though we should respond constructively. "
            "Overall, while there are inspiring aspects, some details are invented or irrelevant to the core question."
        )

    def _ensure_ai_message(response: Any, fallback: str) -> AIMessage:
        content = getattr(response, "content", fallback)
        return response if isinstance(response, AIMessage) else AIMessage(content=content)

    model_label = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    provider_label = getattr(llm, "provider", None)

    def router(state: MAState) -> MAState:
        q = state["input"].lower()
        if any(ch in q for ch in ["+", "-", "*", "/"]) and any(ch.isdigit() for ch in q):
            route = "math"
        elif any(key in q for key in ["capital", "city"]):
            route = "capital"
        elif any(key in q for key in ["sentiment", "feel", "mood"]):
            route = "sentiment"
        elif any(key in q for key in ["summary", "summarize", "shorten"]):
            route = "summarizer"
        elif any(key in q for key in ["probe", "evaluation", "hallucination", "bias", "toxicity"]):
            route = "probe"
        else:
            route = "general"
        return {"route": route}

    def capital_agent(state: MAState) -> MAState:
        from langchain_core.messages import AIMessage

        words = state["input"].rstrip("?!.").split()
        country = words[-1]
        cap = get_capital.run(country)
        answer = f"The capital of {country.capitalize()} is {cap}."
        messages = state.get("messages", [])
        return {
            "messages": messages + [AIMessage(content=answer)],
            "output": answer,
            "intermediate": state.get("intermediate", []) + [answer],
        }

    def math_agent(state: MAState) -> MAState:
        from langchain_core.messages import AIMessage

        expr = "".join(ch for ch in state["input"] if ch in "0123456789+-*/(). ")
        result = calc_math.run(expr)
        answer = f"Result: {result}"
        return {
            "messages": state.get("messages", []) + [AIMessage(content=answer)],
            "output": answer,
            "intermediate": state.get("intermediate", []) + [answer],
        }

    def sentiment_agent(state: MAState) -> MAState:
        from langchain_core.messages import AIMessage

        raw = analyze_sentiment.run(state["input"])
        answer = f"Sentiment analysis: {raw}"
        return {
            "messages": state.get("messages", []) + [AIMessage(content=answer)],
            "output": answer,
            "intermediate": state.get("intermediate", []) + [answer],
        }

    def summarizer_agent(state: MAState) -> MAState:
        from langchain_core.messages import AIMessage

        summary = summarize.run(state.get("context") or state["input"])
        answer = f"Summary: {summary}"
        return {
            "messages": state.get("messages", []) + [AIMessage(content=answer)],
            "output": answer,
            "intermediate": state.get("intermediate", []) + [answer],
        }

    def general_agent(state: MAState) -> MAState:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        try:
            response = llm.invoke(
                [
                    SystemMessage(content="You are a helpful multi-agent assistant."),
                    HumanMessage(content=state["input"]),
                ]
            )
        except Exception as exc:  # pragma: no cover - dependent on external service
            print(f"General agent LLM invocation failed: {exc}")
            answer = (
                "Unable to contact the language model right now. "
                "Please retry or rely on tool results."
            )
            ai_msg = AIMessage(content=answer)
        else:
            ai_msg = _ensure_ai_message(response, str(response))
            answer = ai_msg.content
        return {
            "messages": state.get("messages", []) + [ai_msg],
            "output": answer,
            "intermediate": state.get("intermediate", []) + [answer],
        }

    def probe_agent(state: MAState) -> MAState:
        from langchain_core.messages import AIMessage

        topic = state["input"][:40].strip()
        text = evaluation_probe.run(topic)
        return {
            "messages": state.get("messages", []) + [AIMessage(content=text)],
            "output": text,
            "intermediate": state.get("intermediate", []) + [text],
        }

    def llm_synthesizer(state: MAState) -> MAState:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        tool_outputs = "\n".join(state.get("intermediate", [])) or "No tool outputs produced."
        synthesis_prompt = (
            "You are the orchestrator for a team of specialist agents. "
            "Use the tool outputs to craft a final, helpful answer. "
            "Cite tools implicitly when relevant and keep the tone professional."
        )
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=synthesis_prompt),
                    HumanMessage(
                        content=(
                            f"Original user request: {state['input']}\n"
                            f"Tool outputs:\n{tool_outputs}\n"
                            "Provide the final response the user should see."
                        )
                    ),
                ]
            )
        except Exception as exc:  # pragma: no cover - dependent on external service
            print(f"LLM synthesizer invocation failed: {exc}")
            fallback = (
                "LLM synthesizer unavailable; returning combined tool outputs:\n"
                f"{tool_outputs}"
            )
            ai_msg = AIMessage(content=fallback)
        else:
            ai_msg = _ensure_ai_message(response, tool_outputs)
        new_intermediate = state.get("intermediate", []) + [ai_msg.content]
        return {
            "messages": state.get("messages", []) + [ai_msg],
            "output": ai_msg.content,
            "intermediate": new_intermediate,
        }

    orchestrator_metadata: dict[str, Any] = {
        "ls_is_agent": True,
        "ls_agent_type": "orchestrator_llm",
        "framework": "langgraph",
    }
    if model_label:
        orchestrator_metadata["ls_model_name"] = model_label
    if provider_label:
        orchestrator_metadata["ls_provider"] = provider_label

    graph = StateGraph(MAState)
    graph.add_node("router", router)
    graph.add_node("capital_agent", capital_agent)
    graph.add_node("math_agent", math_agent)
    graph.add_node("sentiment_agent", sentiment_agent)
    graph.add_node("summarizer_agent", summarizer_agent)
    graph.add_node("general_agent", general_agent)
    graph.add_node("probe_agent", probe_agent)
    graph.add_node(
        "llm_synthesizer",
        RunnableLambda(llm_synthesizer).with_config(
            run_name="orchestrator_llm_agent",
            tags=["agent", "synthesizer"],
            metadata=orchestrator_metadata,
        ),
    )

    def decide(state: MAState) -> str:
        return state.get("route", "general")

    graph.add_conditional_edges(
        "router",
        decide,
        {
            "capital": "capital_agent",
            "math": "math_agent",
            "sentiment": "sentiment_agent",
            "summarizer": "summarizer_agent",
            "probe": "probe_agent",
            "general": "general_agent",
        },
    )
    graph.add_edge("capital_agent", "llm_synthesizer")
    graph.add_edge("math_agent", "llm_synthesizer")
    graph.add_edge("sentiment_agent", "llm_synthesizer")
    graph.add_edge("summarizer_agent", "llm_synthesizer")
    graph.add_edge("general_agent", "llm_synthesizer")
    graph.add_edge("probe_agent", "llm_synthesizer")
    graph.add_edge("llm_synthesizer", END)
    graph.set_entry_point("router")
    app = graph.compile()

    base_questions = [
        "What is the capital of Germany?",
        "Compute 12 * (3 + 5) - 4",
        "Analyze the sentiment: I love clean, well-documented code but hate rushed hacks",
        "Summarize: OpenTelemetry enables unified observability across traces, metrics, and logs for cloud-native systems.",
        "Tell me something interesting about compilers.",
        "What is the capital city of Japan?",
        "Please summarize why observability matters in modern distributed systems.",
        "Compute (22 / 2) + 7 * 3",
    ]
    import random

    question = random.choice(base_questions)
    print("\n--- Multi-Agent Demo (Simplified) ---")
    print(f"\nUser: {question}")
    state = app.invoke({"input": question, "messages": [], "intermediate": []})
    primary_answer = state.get("output") or ""
    print("Primary Answer:", primary_answer)

    second_tool_is_probe = random.random() < 0.5
    if second_tool_is_probe:
        probe_text = evaluation_probe.run(question[:40])
        print("Probe Output:", probe_text)
    else:
        sentiment_text = analyze_sentiment.run(primary_answer or question)
        print("Sentiment Output:", sentiment_text)
    _flush_evaluations()
    print("--- End Multi-Agent Demo ---\n")
    _flush_evaluations()


def _resolve_llm(cache_file: str = DEFAULT_CACHE_FILE) -> Tuple[ChatOpenAI, Any]:
    llm, token_manager, _ = create_cisco_chat_llm(cache_file=cache_file)
    return llm, token_manager


def _maybe_run_inhouse_rag() -> bool:
    inhouse_rag_enabled = os.getenv("GENAI_DEMO_INHOUSE_RAG")
    if not inhouse_rag_enabled:
        return False

    import runpy

    rag_query = os.getenv(
        "GENAI_DEMO_INHOUSE_RAG_QUERY",
        "How is generative AI changing software reliability practices?",
    )
    rag_path = (
        Path(__file__).resolve().parents[5]
        / "opentelemetry-python-contrib"
        / "util"
        / "opentelemetry-util-genai-dev"
        / "examples"
        / "langgraph-multi-agent-rag"
        / "main.py"
    )
    try:
        rag_globals = runpy.run_path(str(rag_path))
        run_single_query = rag_globals.get("run_single_query")
        if callable(run_single_query):
            print("\n=== In-House Multi-Agent RAG Demo ===")
            rag_result = run_single_query(rag_query)
            print(json.dumps(rag_result, indent=2)[:2000])
            print("=== End In-House Multi-Agent RAG Demo ===\n")
        else:
            print("RAG demo function run_single_query not found in script")
    except Exception as exc:
        print(f"Failed to execute in-house RAG demo: {exc}")
    _flush_evaluations()
    return True


def main() -> None:
    instrumentor = LangchainInstrumentor()
    instrumentor.instrument()

    try:
        llm, _token_manager = _resolve_llm()
    except Exception:
        instrumentor.uninstrument()
        raise

    tracer = trace.get_tracer("demo.manual.langchain.multi")
    try:
        if _maybe_run_inhouse_rag():
            return

        with tracer.start_as_current_span(
            "demo.request",
            kind=trace.SpanKind.SERVER,
            attributes={
                "http.method": "GET",
                "http.route": "/demo/multi",
                "service.name": "langchain-demo-app",
            },
        ):
            multi_agent_demo(llm)
    finally:
        _flush_evaluations()
        instrumentor.uninstrument()


if __name__ == "__main__":
    main()
