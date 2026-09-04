# Fraud-detection agent (ADOT)

A minimal Bedrock AgentCore agent that scores a claim's fraud risk from structured facts. Same
ADOT-only instrumentation rationale as `claims_intake_coding/agentcore-adot` -- see that
example's README and `main.py`/`patch_dockerfile.py` for the detail. Reports into the same
Agent Observability project/log-stream as `ai_underwriting_pipeline` and
`claims_intake_coding`, distinguished by `service.name` and `genai_lens.feature_id`.

## Deploy

```bash
set -a; . ~/.cr/.cr.aws.o11y-for-ir; set +a   # AWS credentials
set -a; . ~/.cr/.cr.galileo.delian-league; set +a   # GALILEO_* / AO endpoint
set -a; . ~/.cr/.cr.openai.galileo; set +a    # OPENAI_API_KEY
./deploy.sh
```

## Invoke

```bash
agentcore invoke --agent fraud_detection '{"claim_facts":{"claim_id":"CLM-1"}}'
```
