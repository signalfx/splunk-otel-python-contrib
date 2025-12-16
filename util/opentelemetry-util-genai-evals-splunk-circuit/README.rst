splunk-otel-util-genai-evals-splunk-circuit
===========================================

This package adds a Cisco CircuIT evaluation model that can be used with
``splunk-otel-genai-evals-deepeval``. It registers a custom Deepeval model under
both ``splunk-circuit`` and ``circuit`` so it can be selected by setting the
``DEEPEVAL_MODEL`` environment variable.

Quick start
-----------

1. Install the package alongside the base Deepeval integration::

    pip install splunk-otel-genai-evals-deepeval splunk-otel-util-genai-evals-splunk-circuit

2. Configure credentials using the same environment variables as the local
   CircuIT shim::

    export CISCO_CLIENT_ID=...
    export CISCO_CLIENT_SECRET=...
    export CISCO_APP_KEY=...
    export DEEPEVAL_MODEL=splunk-circuit

   Optional overrides:

   * ``CIRCUIT_UPSTREAM_BASE`` - base URL for the CircuIT API
   * ``CISCO_TOKEN_URL`` - OAuth token endpoint (default
     ``https://id.cisco.com/oauth2/default/v1/token``)
   * ``CIRCUIT_TOKEN_CACHE`` - path for cached access tokens
   * ``CIRCUIT_DEFAULT_DEPLOYMENT`` - default deployment/model name used for
     evaluations
   * ``CISCO_CIRCUIT_TOKEN`` - supply a pre-minted token instead of client
     credentials

3. Run evaluations as usual. The Deepeval integration will automatically create
   the CircuIT evaluation model and use it to score metrics.

