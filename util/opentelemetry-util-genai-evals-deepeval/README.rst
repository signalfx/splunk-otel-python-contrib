OpenTelemetry GenAI Utilities Evals for Deepeval (opentelemetry-util-genai-evals-deepeval)
==========================================================================================

This package plugs the `deepeval <https://github.com/confident-ai/deepeval>`_ metrics
suite into the OpenTelemetry GenAI evaluation pipeline. When it is installed a
``Deepeval`` evaluator is registered automatically and, unless explicitly disabled,
is executed for every LLM/agent invocation alongside the builtin metrics.

Installation
------------

Install the evaluator (and its runtime dependencies) from PyPI:

.. code-block:: bash

    pip install opentelemetry-util-genai-evals-deepeval

The command pulls in ``opentelemetry-util-genai``, ``deepeval`` and ``openai``
automatically so the evaluator is ready to use right after installation.

Requirements
------------

* ``opentelemetry-util-genai`` together with ``deepeval`` and ``openai`` –
  these are installed automatically when you install this package.
* An LLM provider supported by Deepeval. By default the evaluator uses OpenAI's
  ``gpt-4o-mini`` model because it offers the best balance of latency and cost
  for judge workloads right now, so make sure ``OPENAI_API_KEY`` is available.
  To override the model, set ``DEEPEVAL_EVALUATION_MODEL`` (or ``DEEPEVAL_MODEL`` /
  ``OPENAI_MODEL``) to a different deployment along with the corresponding
  provider credentials.
* (Optional) ``DEEPEVAL_API_KEY`` if your Deepeval account requires it.

Configuration
-------------

Use ``OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS`` to select the metrics that
should run. Leaving the variable unset enables every registered evaluator with its
default metric set. Examples:

* ``OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=Deepeval`` – run the default
  Deepeval bundle (Bias, Toxicity, Answer Relevancy, Faithfulness).
* ``OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=Deepeval(LLMInvocation(bias(threshold=0.75)))`` –
  override the Bias threshold for LLM invocations and skip the remaining metrics.
* ``OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=none`` – disable the evaluator entirely.

Results are emitted through the standard GenAI evaluation emitters (events,
metrics, spans). Each metric includes helper attributes such as
``deepeval.success``, ``deepeval.threshold`` and any evaluation model metadata
returned by Deepeval. Metrics that cannot run because required inputs are missing
(for example Faithfulness without a ``retrieval_context``) are marked as
``label="skipped"`` and carry a ``deepeval.error`` attribute so you can wire the
necessary data or disable that metric explicitly.

Default OpenAI Usage
--------------------

If you're using OpenAI directly, no additional configuration is needed. Simply set
``OPENAI_API_KEY`` and the evaluator will work out of the box:

.. code-block:: bash

    export OPENAI_API_KEY="sk-your-openai-api-key"

The custom ``DEEPEVAL_LLM_*`` environment variables described below are **only needed
when using a custom LLM provider** (e.g., Azure OpenAI, private deployments, or API
gateways). They do not affect the default OpenAI behavior.

Custom LLM Provider Configuration
---------------------------------

Use these environment variables to configure a custom LLM endpoint (e.g., Azure OpenAI,
private deployments, or LLM gateways). These settings are **optional** and only apply
when ``DEEPEVAL_LLM_BASE_URL`` is set:

**Basic Configuration:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Environment Variable
     - Description
   * - ``DEEPEVAL_LLM_BASE_URL``
     - Custom LLM endpoint URL (required for custom providers)
   * - ``DEEPEVAL_LLM_MODEL``
     - Model name (default: ``gpt-4o-mini``)
   * - ``DEEPEVAL_LLM_PROVIDER``
     - Provider identifier for model prefix (default: ``openai``)
   * - ``DEEPEVAL_LLM_API_KEY``
     - Static API key (use this OR OAuth2, not both)
   * - ``DEEPEVAL_LLM_AUTH_HEADER``
     - Auth header name (default: ``api-key``)
   * - ``DEEPEVAL_LLM_EXTRA_HEADERS``
     - JSON string of additional HTTP headers (see examples below)
   * - ``DEEPEVAL_LLM_CLIENT_APP_NAME``
     - App key/name passed in request body for some providers

OAuth2 Authentication
---------------------

For providers requiring OAuth2 token-based authentication:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Environment Variable
     - Description
   * - ``DEEPEVAL_LLM_TOKEN_URL``
     - OAuth2 token endpoint (enables OAuth2 mode)
   * - ``DEEPEVAL_LLM_CLIENT_ID``
     - OAuth2 client ID
   * - ``DEEPEVAL_LLM_CLIENT_SECRET``
     - OAuth2 client secret
   * - ``DEEPEVAL_LLM_GRANT_TYPE``
     - OAuth2 grant type (default: ``client_credentials``)
   * - ``DEEPEVAL_LLM_SCOPE``
     - OAuth2 scope (optional)
   * - ``DEEPEVAL_LLM_AUTH_METHOD``
     - Token auth method: ``basic`` (default) or ``post``

Examples
--------

**Static API Key (Azure OpenAI):**

.. code-block:: bash

    export DEEPEVAL_LLM_BASE_URL="https://your-resource.openai.azure.com/openai/deployments"
    export DEEPEVAL_LLM_MODEL="gpt-4o"
    export DEEPEVAL_LLM_PROVIDER="azure"
    export DEEPEVAL_LLM_API_KEY="your-api-key"

**OAuth2 with Basic Auth (Okta-style):**

.. code-block:: bash

    export DEEPEVAL_LLM_BASE_URL="https://llm-gateway.example.com/openai/deployments/gpt-4o-mini"
    export DEEPEVAL_LLM_MODEL="gpt-4o-mini"
    export DEEPEVAL_LLM_CLIENT_ID="your-client-id"
    export DEEPEVAL_LLM_CLIENT_SECRET="your-client-secret"
    export DEEPEVAL_LLM_TOKEN_URL="https://identity.example.com/oauth2/default/v1/token"
    export DEEPEVAL_LLM_CLIENT_APP_NAME="your-app-key"

**OAuth2 with Azure Active Directory:**

.. code-block:: bash

    export DEEPEVAL_LLM_BASE_URL="https://your-api.example.com/v1"
    export DEEPEVAL_LLM_MODEL="gpt-4o"
    export DEEPEVAL_LLM_PROVIDER="openai"
    export DEEPEVAL_LLM_CLIENT_ID="azure-client-id"
    export DEEPEVAL_LLM_CLIENT_SECRET="azure-client-secret"
    export DEEPEVAL_LLM_TOKEN_URL="https://login.microsoftonline.com/tenant-id/oauth2/v2.0/token"
    export DEEPEVAL_LLM_SCOPE="api://resource/.default"
    export DEEPEVAL_LLM_AUTH_METHOD="post"

**Custom Headers (for API gateways requiring additional headers):**

.. code-block:: bash

    # Add custom headers as JSON
    export DEEPEVAL_LLM_EXTRA_HEADERS='{"system-code": "APP-123", "x-tenant-id": "tenant-abc"}'

    # Combined with other settings
    export DEEPEVAL_LLM_BASE_URL="https://gateway.example.com/openai/deployments"
    export DEEPEVAL_LLM_MODEL="gpt-4o"
    export DEEPEVAL_LLM_API_KEY="your-api-key"
    export DEEPEVAL_LLM_EXTRA_HEADERS='{"system-code": "MYAPP-001"}'

The ``DEEPEVAL_LLM_EXTRA_HEADERS`` environment variable accepts a JSON-formatted string
containing key-value pairs that will be added as HTTP headers to all LLM API requests.
This is useful for API gateways that require custom headers for authentication or tracking.

**Note:** LiteLLM does not natively support setting ``extra_headers`` via environment
variables (it must be passed programmatically). We provide ``DEEPEVAL_LLM_EXTRA_HEADERS``
to bridge this gap for DeepEval users who need custom headers without code changes.
See `LiteLLM SDK Header Support <https://docs.litellm.ai/docs/sdk/headers>`_ for more details
on how headers work in LiteLLM.
