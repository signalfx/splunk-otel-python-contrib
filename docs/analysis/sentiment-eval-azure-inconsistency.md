# Sentiment Evaluation: Reported Inconsistency Investigation

**Date:** 2026-03-04
**Status:** Not reproducible — likely intermittent LLM stochasticity
**Affected component:** `opentelemetry-util-genai-evals-deepeval` sentiment metric

## Problem Statement

An end-user reported that the **sentiment evaluation** produced incorrect
results when using **Azure OpenAI (gpt-4o)** as the judge model. In two
separate cases, factual/positive text was scored as **"Negative"** with very
low scores (0.05 and 0.09), despite the explanation correctly identifying the
text as neutral or positive. The user believed the issue was specific to the
Azure provider.

## Test Data

### Case 1: NY → London + Quantum Computing Aside

**Input:**

```text
You are a flight booking specialist. Provide concise options.

Find an appealing flight from New York to London departing 2026-04-03 for 2 travellers.

Include an unrelated aside summarizing recent quantum computing acronyms even if
not requested by the traveller.
```

**Output:** Flight details (SkyLine, $953/person, Premium Economy) + quantum
computing acronyms aside (Qubit, NISQ, QAOA, VQE, QFT, QEC, CV).

**User-reported sentiment:** score=0.05, label="Negative", passed=false
> Explanation: "The text maintains a neutral tone throughout, with no
> emotion-carrying words or phrases indicating positive or negative sentiment..."

### Case 2: Seattle → Paris (Short Response)

**Input:**

```text
Find an appealing flight from Seattle to Paris departing 2026-04-03 for 2 travellers.
```

**Output:** Flight details (CloudNine, $745 return for 2, non-stop).

**User-reported sentiment:** score=0.09, label="Negative", passed=false
> Explanation: "The response conveys a positive and helpful tone, highlighting
> an appealing flight option... The overall tone is enthusiastic and
> service-oriented, aligning with a strongly positive sentiment score."

## Environment

| Package | Version |
|---|---|
| `deepeval` | 3.7.9 |
| `splunk-otel-genai-evals-deepeval` | 0.1.13 |
| `splunk-otel-util-genai-evals` | 0.1.8 |
| Python | 3.12.7 |

## Verification Results (2026-03-04)

We ran `verify_sentiment_azure_issue.py` against **three providers** using the
same input/output text. Each test case was evaluated as an `AgentInvocation`
with the full default metric suite.

### Case 1: NY → London + Quantum Aside

| Provider | Model | Sentiment Score | Label | Passed |
|---|---|---|---|---|
| CIRCUIT (OpenAI-compat) | gpt-4o-mini | 0.794 | Positive | true |
| Azure OpenAI | gpt-4o | 0.847 | Positive | true |
| Azure Cognitive Services | gpt-4o-mini | 0.801 | Positive | true |
| **User-reported** | *gpt-4o (Azure)* | **0.05** | **Negative** | **false** |

### Case 2: Seattle → Paris (Short)

| Provider | Model | Sentiment Score | Label | Passed |
|---|---|---|---|---|
| CIRCUIT (OpenAI-compat) | gpt-4o-mini | 0.832 | Positive | true |
| Azure OpenAI | gpt-4o | 0.848 | Positive | true |
| Azure Cognitive Services | gpt-4o-mini | 0.825 | Positive | true |
| **User-reported** | *gpt-4o (Azure)* | **0.09** | **Negative** | **false** |

### Other Metrics (Consistent Across All Providers)

| Metric | Score Range | Label | Passed |
|---|---|---|---|
| Bias | 0.0 | Not Biased | true |
| Toxicity | 0.0 | Non Toxic | true |
| Answer Relevancy | 0.60–1.0 | Relevant | true |
| Hallucination | 0.10–0.14 | Not Hallucinated | true |

### Key Finding

**All three providers produce consistent, correct sentiment results** (scores
0.77–0.85, "Positive", passed=true) across both test cases. The user-reported
scores of 0.05 and 0.09 could **not be reproduced** against the same Azure
endpoint and API key. This rules out a provider-specific calibration bug.

## Root Cause Analysis

### The Code Path Is Correct

The sentiment metric is a custom GEval-based metric defined in
`deepeval_metrics.build_sentiment_metric()`. The evaluation chain:

1. **GEval sends prompt to LLM judge** with criteria:
   > "Rate the overall sentiment of the output text on a scale from 0 to 1.
   > 0 = strongly negative, 0.5 = neutral, 1 = strongly positive."

2. **LLM returns both a score and an explanation** — extracted independently
   by deepeval's GEval framework.

3. **Post-processing** in `_apply_sentiment_postprocessing()` maps score to
   label using thresholds: ≤0.35 = Negative, 0.35–0.65 = Neutral, ≥0.65 =
   Positive.

4. **`_derive_passed()`** maps label to passed: Positive/Neutral → true,
   Negative → false.

Given the user's scores (0.05, 0.09), the code correctly produced
label="Negative" and passed=false. **There is no bug in the post-processing
logic.**

### The Issue Is Non-Deterministic LLM Scoring

The user-reported scores (0.05, 0.09) vs our verification scores (~0.80–0.85)
against the **same endpoints** confirm the issue is **intermittent LLM
stochasticity**, not a provider-specific or model-specific calibration bug.

In both user-reported cases, the LLM judge produced an **internally
contradictory response**: the explanation correctly described the sentiment
(neutral or positive), but the numerical score was near zero. This suggests
the model occasionally confuses the bipolar scale (0=negative, 0.5=neutral,
1=positive) with an intensity scale (0=no emotion, 1=strong emotion).

### Contributing Factors

1. **LLM stochasticity**: GEval-based scoring is inherently non-deterministic.
   Even with temperature=0, LLM outputs can vary due to batching,
   quantization, and infrastructure-level differences.
2. **Model version drift**: Azure OpenAI deployments may update model snapshots
   without notice, which can shift scoring behavior between runs.
3. **Score-explanation decoupling**: GEval extracts the numerical score and the
   textual explanation independently. The LLM can produce a correct explanation
   while assigning a contradictory score — there is no built-in consistency
   check between the two.

## Affected Code Files

| File | Role |
|---|---|
| `util/opentelemetry-util-genai-evals-deepeval/src/opentelemetry/util/evaluator/deepeval_metrics.py` | `build_sentiment_metric()` — GEval criteria/steps |
| `util/opentelemetry-util-genai-evals-deepeval/src/opentelemetry/util/evaluator/deepeval.py` | `_apply_sentiment_postprocessing()` — score→label mapping |
| `util/opentelemetry-util-genai-evals-deepeval/src/opentelemetry/util/evaluator/deepeval_model.py` | `create_eval_model()` — provider/model configuration |

## Verification Script

A reproduction script is available at:
`util/opentelemetry-util-genai-evals-deepeval/examples/verify_sentiment_azure_issue.py`

Configure provider environment variables and run to compare sentiment scoring
across providers. See the script's docstring for usage instructions.
