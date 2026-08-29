# AI Underwriting Pipeline on Bedrock AgentCore

A multi-agent property-insurance underwriting assistant, instrumented with the Splunk
OpenTelemetry GenAI stack and deployable to AWS Bedrock AgentCore. Built for the
**O11y for AI – GenAI Lens** demo, on the same shape as
[`multi_agent_travel_planner/agentcore-splunk`](../../multi_agent_travel_planner/agentcore-splunk).

It exists to produce the three metrics the GenAI Lens consolidates from Agent
Observability, and to demonstrate the same workload in two telemetry postures.

## The three Lens metrics

| Lens metric | Where it comes from |
|---|---|
| **Action completeness** — evaluated metrics / total traces | `underwriting.action_completeness` on the root span: of the six actions a complete pass must perform, how many it actually performed. `underwriting.actions_missing` names the gaps |
| **Successful / failed traces** | Root span status. `OK` when every required action completed, `ERROR` with the missing list when not |
| **Responses prevented by agent control** | `agent_control.response_prevented` on the root, plus one `agent_control.evaluate` span per screening with `agent_control.blocked`, `.rule` and `.reason` |

A blocked response is deliberately **not** counted as a failed trace. The control firing is
the system behaving correctly; conflating the two would make the Lens's failure rate climb
every time the guardrails did their job.

## Two telemetry postures, one codebase

Selected by `UNDERWRITING_EXPORT_MODE`. Same application, same spans — only the transport
differs, which is the entire point.

| Mode | Behaviour | Demonstrates |
|---|---|---|
| `otlp` | OTLP export to Agent Observability | The **properly instrumented** case. The workload appears in AO's own inventory and AO can evaluate it |
| `console` | Spans serialised to **stdout**, which AgentCore captures into CloudWatch Logs | An app that genuinely runs OpenTelemetry and exports it to a **log sink**. This is the input for recovering spans from logs and forwarding them to AO |
| `both` | Both at once | Side-by-side comparison in a single run |

`console` mode uses `SimpleSpanProcessor`, not `BatchSpanProcessor`, on purpose: one
complete span per log record. A batch processor emits a JSON array spanning many lines, and
CloudWatch line-fragments multiline output — the array would arrive as unjoinable pieces.

### Why `console` mode matters

AWS Transaction Search already writes OTLP-shaped spans into the `aws/spans` CloudWatch
group, but those are **AWS platform spans**: they prove an invocation happened and how long
it took, and carry no `gen_ai.*` attributes at all. They are Tier A.

This app's own spans, in `console` mode, carry the full set — measured locally:

```
gen_ai.agent.name          gen_ai.provider.name        gen_ai.usage.input_tokens
gen_ai.framework           gen_ai.request.model        gen_ai.usage.output_tokens
gen_ai.operation.name      gen_ai.response.model       gen_ai.tool.name
gen_ai.step.name           gen_ai.request.temperature  gen_ai.response.finish_reasons
```

So recovering *these* from logs and forwarding them to AO delivers **model identity and
token counts** — Tier B — not merely trace structure. That is the difference between the
two span sources, and the reason this variant exists.

## The pipeline

```
intake → credit → claims → property → guidelines → pricing → decision
```

Each stage is a LangGraph node with its own span. Tools (`underwriting_tools.py`) are
mocked but **deterministic per applicant id** — the same id always yields the same credit
score, claims and property risk. An evaluation score that moved because the data moved
would be evidence of nothing, so the only intended source of variation is the quality-noise
injection.

`_derive_tier` deliberately penalises *missing* facts rather than treating them as benign:
a pass that skipped the claims check does not get to call the risk preferred. A skipped step
therefore shows up in the outcome, not only in the metric.

### Injected defects

`UNDERWRITING_NOISE_RATE` (default `0.35`) is the fraction of traces that deliberately
exhibit a defect. Without this every trace is perfect, the completeness metric is a flat
line at 1.0 and the agent-control gate never fires — which proves nothing about the
observability of imperfection.

| Defect | Effect |
|---|---|
| `skip_claims`, `skip_property` | Action completeness drops; tier is penalised |
| `protected_rationale` | Model is steered to cite national origin → **blocked** |
| `unconditional_promise` | Model is steered to guarantee coverage → **blocked** |
| `leak_other_applicant` | Model is steered to name another applicant → **blocked** |

The steer goes into the prompt rather than post-editing the output, so the agent-control
gate is screening a genuine model response.

Measured locally at `UNDERWRITING_NOISE_RATE=1.0` over 6 traces: 2/6 responses prevented,
mean action completeness 0.889, 3/6 traces fully complete, tiers spread across
standard / substandard / declined.

## Agent control

`agent_control.py` screens every outbound decision. Four rules, chosen because in
underwriting the things you must not say are concrete and regulated:

| Rule | Action |
|---|---|
| `cross_applicant_data_leak` | **block** — response referenced a different applicant |
| `protected_characteristic_in_rationale` | **block** — unlawful basis for a decision |
| `unconditional_coverage_commitment` | **block** — the assistant may recommend, not bind |
| `pii_redaction` | **allow, redacted** — the decision is legitimate; strip the identifier |

Deliberately a simple deterministic gate. Real Agent Control does far more — policy sets,
SLM classifiers, streaming interception. What matters here is that a block is *observable in
telemetry with a reason*, so AO can aggregate it. Swapping the rule engine later does not
change the telemetry contract.

## Running locally

```shell
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # add your OPENAI_API_KEY

export UNDERWRITING_EXPORT_MODE=console
export UNDERWRITING_NOISE_RATE=1.0
.venv/bin/python main.py 6        # 6 underwriting passes
```

### Corporate TLS interception

On a laptop behind a TLS-intercepting proxy, the OpenAI call fails with
`CERTIFICATE_VERIFY_FAILED … unable to get local issuer certificate`, which reads like a
network outage rather than a trust problem. The corporate roots are in the macOS System
keychain but not in Python's `certifi` bundle. Combine them:

```shell
CERTIFI=$(.venv/bin/python -c "import certifi;print(certifi.where())")
{ cat "$CERTIFI"
  security find-certificate -a -p /Library/Keychains/System.keychain
  security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain
} > /tmp/combined-ca.pem
export SSL_CERT_FILE=/tmp/combined-ca.pem REQUESTS_CA_BUNDLE=/tmp/combined-ca.pem
```

Not needed inside AgentCore — there is no interception there.

## Deploying to AgentCore

`./deploy.sh` wraps `agentcore configure` + `agentcore launch` for both variants:

```shell
./deploy.sh ao          # OTLP → Agent Observability
./deploy.sh spans-logs  # spans → stdout → CloudWatch Logs
./deploy.sh both        # deploy both agents
```

Requires `GALILEO_API_KEY`, `GALILEO_PROJECT` and `GALILEO_OTEL_TRACES_ENDPOINT` for the
`ao` variant. See `.env.example`.

### Dependency layer order — do not reorder

`aws-opentelemetry-distro` pins the whole OTel stack with `==`. Installing it **after** the
application requirements silently downgrades `opentelemetry-semantic-conventions` beneath
what the GenAI instrumentation needs; the instrumentation then fails to import at runtime
while the process carries on serving traffic normally, and no span is ever emitted to
contradict it. `requirements.txt` installs app deps first and the Splunk GenAI packages
second, and does not include the AWS distro at all.

## Files

| File | Purpose |
|---|---|
| `main.py` | LangGraph pipeline, AgentCore entrypoint, root-span Lens metrics |
| `telemetry.py` | Provider setup and export-mode selection |
| `agent_control.py` | Outbound response screening; emits the prevented-response signal |
| `underwriting_tools.py` | Deterministic mock data sources + `REQUIRED_ACTIONS` |
| `deploy.sh` | AgentCore deployment for both variants |
