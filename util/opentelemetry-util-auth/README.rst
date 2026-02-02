OpenTelemetry Util Auth
=======================

OAuth2 authentication utilities for OpenTelemetry instrumented LLM applications.

Installation
------------

**From source (for development):**

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/signalfx/splunk-otel-python-contrib.git
    cd splunk-otel-python-contrib
    
    # Install in editable mode
    pip install -e ./util/opentelemetry-util-auth

Usage
-----

The ``OAuth2TokenManager`` manages OAuth2 access tokens for LLM endpoints using the client credentials flow.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from opentelemetry.util.auth import OAuth2TokenManager

    # Using environment variables
    token_manager = OAuth2TokenManager()
    token = token_manager.get_token()

    # Use token with your LLM client (OpenAI SDK)
    from openai import OpenAI
    client = OpenAI(
        api_key=token,  # Token used as Bearer token
        base_url=OAuth2TokenManager.get_llm_base_url("gpt-4o-mini")
    )

    # Or with custom header (for gateways requiring specific header names)
    client = OpenAI(
        api_key="placeholder",  # SDK requires this
        base_url=OAuth2TokenManager.get_llm_base_url("gpt-4o-mini"),
        default_headers={"api-key": token}  # Custom header for your gateway
    )

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Required:

- ``LLM_CLIENT_ID``: OAuth2 client ID
- ``LLM_CLIENT_SECRET``: OAuth2 client secret  
- ``LLM_TOKEN_URL``: OAuth2 token endpoint URL

Additional:

- ``LLM_BASE_URL``: LLM endpoint base URL
- ``LLM_AUTH_METHOD``: Authentication method - ``basic`` (default) or ``post``
- ``LLM_SCOPE``: OAuth2 scope

Authentication Methods
~~~~~~~~~~~~~~~~~~~~~~

**Basic Authentication (Default)**

Most OAuth2 providers (Okta, Cisco Identity):

.. code-block:: python

    import os
    os.environ["LLM_AUTH_METHOD"] = "basic"
    token_manager = OAuth2TokenManager()

**POST Body Authentication**

Azure AD and some enterprise IdPs:

.. code-block:: python

    import os
    os.environ["LLM_AUTH_METHOD"] = "post"
    os.environ["LLM_SCOPE"] = "api://your-resource/.default"
    token_manager = OAuth2TokenManager()

Development
-----------

**Running Tests:**

.. code-block:: bash

    # Install test dependencies
    pip install -e "./util/opentelemetry-util-auth[test]"
    
    # Run tests
    pytest util/opentelemetry-util-auth/tests/ -v
    
    # Run tests with coverage
    pytest util/opentelemetry-util-auth/tests/ -v --cov=opentelemetry.util.auth

License
-------

Apache License 2.0

