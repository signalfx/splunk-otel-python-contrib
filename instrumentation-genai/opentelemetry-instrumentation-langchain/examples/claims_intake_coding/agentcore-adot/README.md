# Claims intake coding agent (ADOT)

A minimal Bedrock AgentCore agent that extracts structured intake fields from a free-text
claim description. It exists to demonstrate the AWS-standardized instrumentation path --
`aws-opentelemetry-distro` (ADOT) via `opentelemetry-instrument`, not the Splunk OTel GenAI
stack the sibling `ai_underwriting_pipeline/agentcore-splunk` example uses -- reporting into
the same Agent Observability project so the GenAI Lens onboarding wizard can discover it
alongside the other two demo agents.

See `main.py` and `patch_dockerfile.py` for why this combination (ADOT + the community
`opentelemetry-instrumentation-openai` package) avoids the semconv-version conflict that made
`--disable-otel` necessary elsewhere: the starter toolkit's Dockerfile template hardcodes an
old ADOT release (`0.12.2`, pinning `opentelemetry-sdk==1.33.1`); `patch_dockerfile.py` bumps
it to `0.19.0` (`opentelemetry-sdk==1.44.0`), which is new enough to carry
`GEN_AI_CONVERSATION_ID`. This agent does not depend on `splunk-otel-util-genai` /
`splunk-otel-genai-emitters-splunk`, which pin `opentelemetry-instrumentation~=0.57b0` and
cannot install alongside *any* current ADOT release regardless of version bump -- that
incompatibility is separate from, and not fixed by, the version-bump this example applies.

## Deploy

```bash
set -a; . ~/.cr/.cr.aws.o11y-for-ir; set +a   # AWS credentials
set -a; . ~/.cr/.cr.galileo.delian-league; set +a   # GALILEO_* / AO endpoint
set -a; . ~/.cr/.cr.openai.galileo; set +a    # OPENAI_API_KEY
./deploy.sh
```

## Invoke

```bash
agentcore invoke --agent claims_intake_coding '{"claim_text":"..."}'
```
