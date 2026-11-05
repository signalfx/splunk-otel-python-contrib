# Repository Deprecation

This repository and all packages maintained in `instrumentation-genai/` and
`util/` are deprecated. Support and maintenance have ended, including bug fixes,
new features, and security updates. Avoid adopting these packages for new projects.

Use [upstream OpenTelemetry Python Contrib](https://github.com/open-telemetry/opentelemetry-python-contrib)
for instrumentation going forward. Direct instrumentation issues and contributions
to that project, following its contribution and security policies.

This includes framework instrumentations, the GenAI utility library, evaluators,
emitters, and telemetry translators. Existing documentation and examples remain
available as reference material for existing users.

## Existing Users

- Review where your applications depend on these packages and plan to replace
  those dependencies.
- Keep a record of your installed versions and configuration while evaluating
  alternatives.
- Validate instrumentation coverage, span relationships, attributes, metrics,
  content capture, and evaluation behavior before switching implementations.

Check upstream for instrumentation availability and supported framework versions.
Upstream packages may differ in APIs, configuration, and telemetry schemas; this
notice does not promise a drop-in replacement for every package or Splunk-specific
utility, emitter, translator, or evaluator. No removal date is announced.

For security disclosures affecting this repository, the reporting channel in
[SECURITY.md](SECURITY.md) remains available; fixes are no longer provided here.

## Scope Of This Change

This notice documents repository deprecation. It does not change package runtime
behavior or versions, publish new distributions, remove existing distributions,
or archive the repository on GitHub. Notices in package READMEs become visible
on package registries only if updated distributions are published.
