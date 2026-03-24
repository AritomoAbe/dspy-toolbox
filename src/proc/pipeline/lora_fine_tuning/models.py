from enum import StrEnum

from pydantic import BaseModel, Field

# ── StrEnum constants ──────────────────────────────────────────────────────────

class TrainingPhase(StrEnum):
    BEFORE = "before"
    AFTER  = "after"


class AdapterSaveStatus(StrEnum):
    SAVED   = "saved"
    SKIPPED = "skipped"
    FAILED  = "failed"


# ── Module-level constants ─────────────────────────────────────────────────────

_DEFAULT_N_EPOCHS: int = 3
_DEFAULT_LEARNING_RATE: float = 2e-4
_DEFAULT_LORA_R: int = 16
_DEFAULT_LORA_ALPHA: int = 32
_DEFAULT_LORA_DROPOUT: float = 0.05
_DEFAULT_MAX_NEW_TOKENS: int = 64
_DEFAULT_ATTRIBUTION_IG_STEPS: int = 10

# Qwen-2.5 attention projection names — the modules LoRA wraps.
# Extend this list for other architectures as needed.
_DEFAULT_TARGET_MODULES: list[str] = ["q_proj", "v_proj"]

# Score threshold below which an example is considered a failure.
FAILURE_SCORE_THRESHOLD: float = 0.5

# Log prefix used to group step-level timing lines.
STEP_LOG_PREFIX: str = "lora_step"

class LoRAHyperParams(BaseModel):
    """LoRA adapter configuration passed to PEFT ``LoraConfig``."""
    r: int             = Field(default=_DEFAULT_LORA_R,       ge=1)
    lora_alpha: int    = Field(default=_DEFAULT_LORA_ALPHA,    ge=1)
    lora_dropout: float = Field(default=_DEFAULT_LORA_DROPOUT, ge=0.0, le=1.0)
    target_modules: list[str] = Field(default_factory=lambda: list(_DEFAULT_TARGET_MODULES))
    bias: str          = Field(default="none")


class TrainingHyperParams(BaseModel):
    """Optimiser and schedule parameters."""
    n_epochs: int          = Field(default=_DEFAULT_N_EPOCHS,       ge=1)
    learning_rate: float   = Field(default=_DEFAULT_LEARNING_RATE,  gt=0.0)
    max_grad_norm: float   = Field(default=1.0,                     gt=0.0)
    warmup_steps: int      = Field(default=0,                       ge=0)
    weight_decay: float    = Field(default=0.0,                     ge=0.0)
    max_new_tokens: int    = Field(default=_DEFAULT_MAX_NEW_TOKENS, ge=1)


class StepLog(BaseModel):
    """Loss recorded at a single gradient step."""
    epoch: int
    step: int
    loss: float


class EpochMetrics(BaseModel):
    """Scorer evaluation recorded after each training epoch."""
    epoch: int
    n_examples: int
    n_valid: int
    n_invalid: int
    pass_rate: float


class AttributionComparisonConfig(BaseModel):
    """
    Configuration for the optional before/after attribution comparison.

    Set ``enabled=False`` to skip the attribution comparison entirely.
    ``probe_example_index`` selects which dataset example to attribute
    (defaults to 0).
    """
    enabled: bool              = Field(default=True)
    probe_example_index: int   = Field(default=0, ge=0)
    ig_steps: int              = Field(default=_DEFAULT_ATTRIBUTION_IG_STEPS, ge=1)
    save_html: bool            = Field(default=True)
    save_plots: bool           = Field(default=True)


class LoRARunSummary(BaseModel):
    """Run-level result written to ``summary.json``."""
    hf_model_name: str
    output_dir: str
    lora: LoRAHyperParams
    training: TrainingHyperParams
    epoch_metrics: list[EpochMetrics]
    step_logs: list[StepLog]
    adapter_save_status: AdapterSaveStatus
    attribution_comparison_enabled: bool
    initial_pass_rate: float
    final_pass_rate: float
    pass_rate_delta: float
