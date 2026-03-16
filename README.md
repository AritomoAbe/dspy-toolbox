# dspy-toolbox

Reasoning and debugging utilities for DSPy prompts. Provides a structured pipeline for auditing, optimizing, and interpreting compiled DSPy programs — from raw training data all the way to weight-level attribution.

## Pipeline Overview

```
Training Data  →  Bootstrap Tuning  →  Output Auditing  →  Attribution  →  LoRA Fine-tuning
   (Stage 1)         (Stage 2)            (Stage 2)         (Stage 3)        (Stage 4)
```

Each stage answers a different question:

| Stage | Question |
|---|---|
| 1. Score the Training Set | Is my data good enough to optimize against? |
| 2. Score Actual Outputs | Is my compiled prompt stable and accurate? |
| 3. Score How the LLM Uses the Prompt | Is the prompt working for the right reasons? (`SimplePromptAttributionAuditor` / `LIGAttributionAuditor`) |
| 4. LoRA Fine-tuning + Re-scoring | Did fine-tuning internalize the task or overfit? |

---

## Stage 1 — Score the Training Set

Before touching the model, understand what you are working with. Dataset pathologies contaminate DSPy's optimization signal: teleprompters are only as good as the examples they bootstrap from.

**Auditors:**

| Auditor | What it checks |
|---|---|
| `AnalyzeExpectedFields` | Label distribution — class imbalance, missing fields, value frequency |
| `AnalyzeInputFields` | Input distribution — length, vocabulary, token counts |
| `AnalyzeNearDuplicates` | Near-duplicate detection via TF-IDF cosine similarity |
| `AnalyzeCoOccurrence` | Co-occurrence patterns between output fields |
| `AnalyzeSignalStrength` | Whether inputs are linearly separable per output label (TF-IDF + logistic regression) |

**Demo entry point:**
```
src/proc/demos/meeting_invite/tuning/1_training_data_audit/audit_training_samples.py
```

**Datasets:**
```
tuning/dataset/trainset_emails_50.jsonl          # training set (50 examples)
tuning/dataset/testset_emails_20.jsonl           # held-out test set
tuning/dataset/testset_edge_cases_emails_7.jsonl # edge cases
tuning/dataset/testset_distribution_shift_emails_7.jsonl  # shift probe
```

---

## Stage 2 — Bootstrap Tuning + Output Auditing

### 2a. Compile with BootstrapFewShot

Run DSPy's `BootstrapFewShot` teleprompter to compile the prompt from the training set. The optimized program is saved to `optimized_extractor.json` for use in subsequent audits.

```
src/proc/demos/meeting_invite/tuning/2_bootstrap/1_run_bootstrap_tuning.py
```

### 2b. Score Actual Outputs

This is about measuring the quality and stability of your compiled prompt. A single accuracy number is not enough — you need to understand variance across inputs and runs.

**Stress auditors** (`2_stress_audit_bootstrap_tuning.py`) — run at non-zero temperature to expose instability:

| Auditor | What it measures |
|---|---|
| `SignalToNoiseRatio` | SNR = mean score / (√variance + ε) across repeated runs. High SNR means the metric tracks real behavior, not noise. If SNR < 2, your metric or prompt is unreliable. |
| `MonteCarloEstimation` | Pass probability per example via repeated sampling. Classifies examples as easy (>80%), medium (30–80%), or hard (<30%). Gives a distribution, not a point estimate. |

**Accuracy auditors** (`3_audit_bootstrap_tuning.py`) — run at temperature=0 for deterministic evaluation:

| Auditor | What it measures |
|---|---|
| `AccuracyAuditor` | Per-field and overall accuracy. Flags fields where accuracy < 0.7 (needs attention) or > 0.9 (healthy). Overall accuracy requires all fields correct simultaneously. |
| `PromptSensitivityAuditor` | Consistency across shuffled few-shot demo orderings. At temperature=0, demo order is the only source of variance — a brittle prompt changes its answer when you reorder examples. avg_consistency < 0.7 means the prompt is over-fitted to demo order. |
| `FailureClusterAuditor` | Groups failures by wrong-field pattern (e.g. `"urgency+flexibility"` or `"sender_iana_timezone"`). Surfaces systemic failure modes rather than random errors — if one cluster dominates, fix that slice of the prompt. |

```
src/proc/demos/meeting_invite/tuning/2_bootstrap/2_stress_audit_bootstrap_tuning.py
src/proc/demos/meeting_invite/tuning/2_bootstrap/3_audit_bootstrap_tuning.py
```

---

## Stage 3 — Score How the LLM Uses the Prompt

Two complementary auditors answer the same question from different angles.

### 3a. Leave-one-out Perturbation (`SimplePromptAttributionAuditor`)

Masks each logical prompt segment (INSTRUCTION, each DEMO) one at a time and re-runs the model. Attribution score = `score_full − score_ablated`. Works with any API-based LLM — no model weights required.

| Metric | Threshold |
|---|---|
| `avg_instruction_attribution` | > 0.1 instruction actively used; < 0.05 model ignoring it |
| `avg_demo_attribution` | > 0.1 demos doing real work; < 0.02 dead weight |
| `per_demo_avg_attribution` | per-demo breakdown — identifies which few-shot examples to prune |

When ablating a segment causes unparseable output, `score_ablated` is treated as `0.0` (the segment was critical).

### 3b. Captum LayerIntegratedGradients (`LIGAttributionAuditor`)

Gradient-based token attribution against a local PyTorch model. Produces a three-panel analysis:

| Panel | What it shows |
|---|---|
| A — Token Saliency | Per-token L2 norm of integrated gradients |
| B — Logit Lens | Target-token rank per decoder layer (forward hooks) — shows where the model "commits" to its answer |
| C — Segment Attribution | Token saliency aggregated per DSPy prompt segment (INSTRUCTION, DEMO, INPUT) |

Requires a local HuggingFace model (`pip install torch transformers captum`). Default attribution model: `Qwen/Qwen2.5-1.5B-Instruct`.

**Demo entry points:**
```
src/proc/demos/meeting_invite/tuning/3_prompt_attribution/1_simple_prompt_attribution.py
    → LIGAttributionAuditor against Qwen/Qwen2.5-1.5B-Instruct

src/proc/demos/meeting_invite/tuning/3_prompt_attribution/2_simple_prompt_attribution_abe_gpt.py
    → LIGAttributionAuditor wired to abe-gpt (custom GPT trained from scratch, served locally)
```

---

## Stage 4 — LoRA Fine-tuning + Re-scoring with Captum

> **Planned**

Once prompt optimization has been pushed to its limits, move to lightweight fine-tuning with LoRA. Then reapply Captum attribution to the fine-tuned weights:

- Check whether the new weights genuinely internalized the task structure
- Detect overfitting to surface patterns in the training set
- Compare token attribution before and after fine-tuning to understand what changed

This closes the loop between prompt-level optimization (Stages 1–3) and weight-level optimization.

---

## Demo: Meeting Invite Extraction

The `meeting_invite` demo extracts structured scheduling data from email bodies using a compiled DSPy chain-of-thought prompt.

**Task:** Given an email (`email_from`, `email_to`, `email_body`, `current_date`), extract:

```json
{
  "sender_iana_timezone": "America/Los_Angeles",
  "duration_minutes": 60,
  "urgency": "this_week",
  "flexibility": "specific",
  "preferred_windows": [
    { "day_of_week": "Tuesday", "time_of_day": "afternoon", "iana_timezone": null }
  ],
  "meeting_topic": "Q3 planning sync"
}
```

**Tracked fields:** `sender_iana_timezone`, `duration_minutes`, `urgency`, `flexibility`, `preferred_windows`, `meeting_topic`

**Scoring:** `MeetingInviteScoreExtractor` produces a weighted score (0–1) per example based on per-field comparison with configurable tolerances (e.g. ±10 min on duration, partial credit for time-of-day overlap).

---

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Development

```bash
# Tests (coverage must stay ≥ 95%)
pytest

# Lint
python -m flake8 src/

# Type check
python -m mypy src/
```

All three checks must pass before committing.