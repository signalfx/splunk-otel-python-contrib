"""
Centralized constants for GenAI telemetry attribute names.
This module replaces inline string literals for span & event attributes.
"""

# Semantic attribute names for core GenAI spans/events
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_COMPLETION_PREFIX = "gen_ai.completion"

# Additional semantic attribute constants
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_EVALUATION_NAME = "gen_ai.evaluation.name"
GEN_AI_EVALUATION_SCORE_VALUE = "gen_ai.evaluation.score.value"
GEN_AI_EVALUATION_SCORE_LABEL = "gen_ai.evaluation.score.label"
GEN_AI_EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"
GEN_AI_EVALUATION_ATTRIBUTES_PREFIX = "gen_ai.evaluation.attributes."

# Agent attributes (from semantic conventions)
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
GEN_AI_AGENT_TOOLS = "gen_ai.agent.tools"
GEN_AI_AGENT_TYPE = "gen_ai.agent.type"
GEN_AI_AGENT_SYSTEM_INSTRUCTIONS = "gen_ai.agent.system_instructions"

# Workflow attributes (not in semantic conventions)
GEN_AI_WORKFLOW_NAME = "gen_ai.workflow.name"
GEN_AI_WORKFLOW_TYPE = "gen_ai.workflow.type"
GEN_AI_WORKFLOW_DESCRIPTION = "gen_ai.workflow.description"

# Step attributes (not in semantic conventions)
GEN_AI_STEP_NAME = "gen_ai.step.name"
GEN_AI_STEP_TYPE = "gen_ai.step.type"
GEN_AI_STEP_OBJECTIVE = "gen_ai.step.objective"
GEN_AI_STEP_SOURCE = "gen_ai.step.source"
GEN_AI_STEP_ASSIGNED_AGENT = "gen_ai.step.assigned_agent"
GEN_AI_STEP_STATUS = "gen_ai.step.status"

# Embedding attributes
GEN_AI_EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"
GEN_AI_EMBEDDINGS_INPUT_TEXTS = "gen_ai.embeddings.input.texts"
GEN_AI_REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"

# Retrieval attributes
GEN_AI_RETRIEVAL_TYPE = "gen_ai.retrieval.type"
GEN_AI_RETRIEVAL_QUERY_TEXT = "gen_ai.retrieval.query.text"
GEN_AI_RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"
GEN_AI_RETRIEVAL_DOCUMENTS_RETRIEVED = "gen_ai.retrieval.documents_retrieved"
GEN_AI_RETRIEVAL_DOCUMENTS = "gen_ai.retrieval.documents"

# Tool attributes (from semantic conventions execute_tool span)
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

# Server attributes (from semantic conventions)
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"

# Security attributes (Cisco AI Defense)
GEN_AI_SECURITY_EVENT_ID = "gen_ai.security.event_id"

# Context key for suppressing instrumentation to avoid duplicate telemetry
# when multiple instrumentations (e.g., LangChain + OpenAI) are active
SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY = (
    "suppress_language_model_instrumentation"
)
