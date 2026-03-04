# Sentiment Evaluation: Inconsistency with Small Judge Models

**Date:** 2026-03-04
**Status:** Reproduced — `gpt-4.1-nano` cannot reliably serve as a GEval judge
**Affected component:** `opentelemetry-util-genai-evals-deepeval` sentiment metric

## Problem Statement

An end-user reported that the **sentiment evaluation** produced incorrect
results when using an Azure OpenAI model as the judge. In two separate cases,
factual/positive text was scored as **"Negative"** with very low scores (0.05
and 0.09), despite the explanation correctly identifying the text as neutral
or positive.

## Environment

| Package | Version |
|---|---|
| `deepeval` | 3.7.9 |
| `splunk-otel-genai-evals-deepeval` | 0.1.13 |
| `splunk-otel-util-genai-evals` | 0.1.8 |
| Python | 3.12.7 |

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

### Case 2: Seattle → Paris (Short Response)

**Input:**

```text
Find an appealing flight from Seattle to Paris departing 2026-04-03 for 2 travellers.
```

**Output:** Flight details (CloudNine, $745 return for 2, non-stop).

### User-Reported Results

| Case | Score | Label | Passed | Explanation excerpt |
|---|---|---|---|---|
| Case 1 | **0.05** | Negative | false | "The text maintains a **neutral** tone throughout..." |
| Case 2 | **0.09** | Negative | false | "The response conveys a **positive and helpful** tone..." |

In both cases, the explanation correctly described the sentiment but the
numerical score was near zero — a clear score-explanation contradiction.

## Verification Results (2026-03-04)

We ran `verify_sentiment_azure_issue.py` against **four provider/model
configurations** using the same input/output text. Each test case was
evaluated as an `AgentInvocation` with the full default metric suite.

### Sentiment Results

| # | Provider | Model | Score | Label | Passed |
|---|---|---|---|---|---|
| 1 | CIRCUIT (OpenAI-compat) | gpt-4o-mini | 0.777 | Positive | true |
| 2 | Azure OpenAI | gpt-4o | 0.848 | Positive | true |
| 3 | Azure Cognitive Services | gpt-4o-mini | 0.802 | Positive | true |
| 4 | Azure Cognitive Services | **gpt-4.1-nano** | **0.08** | **Negative** | **false** |

### Other Metrics (Consistent Across All Four Providers)

| Metric | Score Range | Label | Passed |
|---|---|---|---|
| Bias | 0.0 | Not Biased | true |
| Toxicity | 0.0 | Non Toxic | true |
| Answer Relevancy | 0.56–1.0 | Relevant | true |
| Hallucination | 0.10–0.14 | Not Hallucinated | true |

### Key Finding

**The issue reproduces consistently with `gpt-4.1-nano` and only with
`gpt-4.1-nano`.** All three larger models (gpt-4o-mini via CIRCUIT, gpt-4o
via Azure OpenAI, gpt-4o-mini via Azure Cognitive Services) produce correct,
consistent sentiment scores (0.77–0.85, "Positive", passed=true). The
`gpt-4.1-nano` model produces a score of 0.08 ("Negative", failed) every
time, while its own explanation says the text is "positive and neutral" —
exactly matching the user's original report.

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

Given the scores (0.05, 0.08, 0.09), the code correctly produces
label="Negative" and passed=false. **There is no bug in the post-processing
logic.**

### Root Cause: `gpt-4.1-nano` Cannot Serve as a Reliable GEval Judge

The `gpt-4.1-nano` model produces **internally contradictory responses**: the
textual explanation correctly identifies the sentiment, but the numerical
score does not match. Example from run #4:

> **Explanation:** "The response maintains a positive and neutral tone,
> providing detailed information about the flight... indicating a generally
> positive and helpful attitude. The sentiment is mildly positive..."
>
> **Score:** 0.08 (maps to "strongly negative")

The model understands the text's sentiment and can articulate it correctly,
but it **cannot calibrate numerical scores** to the 0-1 bipolar scale
specified in the GEval criteria. It appears to interpret the scale as:
- 0 = no emotion / emotionally flat
- 1 = strong emotion

Instead of the intended:
- 0 = strongly negative
- 0.5 = neutral / no sentiment
- 1 = strongly positive

This is a **model capability limitation** — smaller models lack the
instruction-following precision needed for structured numerical scoring tasks.
All other metrics (bias, toxicity, relevancy, hallucination) work correctly
with `gpt-4.1-nano` because they use simpler binary or threshold-based scoring
that is less sensitive to scale interpretation.

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
