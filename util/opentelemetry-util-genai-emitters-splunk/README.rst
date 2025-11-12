OpenTelemetry GenAI Utilities Splunk Compatible Emitter (opentelemetry-util-genai-emitters-splunk)
==================================================================================================

Splunk Emitter for OpenTelemetry GenAI Utilities
------------------------------------------------

This package provides emitters for Splunk schema for Evaluation Results logs to optimize storage and filtering in Splunk Platform.

This log includes:
* Evaluated GenAI Type metadata (model, provider, request/response details, etc)
* Trace and Span ids references
* Evaluation results

Further Documentation
---------------------
This package extends ``opentelemetry-util-genai``. For architecture, design rationale, and broader usage patterns please consult:
* `Core concepts, high-level usage and setup <https://github.com/signalfx/splunk-otel-python-contrib/>`_
* ``README.packages.architecture.md`` – extensibility architecture & emitter pipeline design.

Those documents cover configuration (environment variables, content capture modes, evaluation emission, extensibility via entry points) and release/stability notes.

Support & Stability
-------------------
GenAI semantic conventions are incubating.

License
-------
Apache 2.0 (see ``LICENSE``).
