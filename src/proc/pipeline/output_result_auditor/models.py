from typing import Any

from pydantic import BaseModel

_ZERO_FLOAT: float = 0.0  # noqa: WPS358


class SNRResult(BaseModel):
    snr: float
    avg_variance: float


class MonteCarloResult(BaseModel):
    easy: int
    medium: int
    hard: int
    probs: list[float]


class PromptSensitivityResult(BaseModel):
    """
    avg_consistency: mean fraction of runs that agree on the same output per example.
                     1.0 = perfectly stable, 0.0 = never agrees.
    avg_flip_rate:   mean fraction of runs that differ from the mode output per example.
    n_shuffles:      number of few-shot orderings used per example.
    """
    avg_consistency: float
    avg_flip_rate: float
    n_shuffles: int


# --- AccuracyAuditor ---

class FieldAccuracy(BaseModel):
    """Accuracy for a single output field."""
    field_name: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else _ZERO_FLOAT


class AccuracyResult(BaseModel):
    """
    overall_accuracy:  fraction of examples where ALL fields are correct.
    per_field:         per-field accuracy, keyed by field name.
    total_examples:    total number of examples evaluated.
    """
    overall_accuracy: float
    per_field: dict[str, FieldAccuracy]
    total_examples: int


# --- FailureClusterAuditor ---

class FailureExample(BaseModel):
    """A single failed prediction with its error cluster label."""
    index: int
    email_body: str
    expected: dict[str, Any]
    predicted: dict[str, Any]
    wrong_fields: list[str]
    cluster: str


class FailureCluster(BaseModel):
    """A group of failures sharing the same wrong-field pattern."""
    cluster: str
    count: int
    examples: list[FailureExample]

    @property
    def wrong_fields(self) -> list[str]:
        return self.cluster.split("+")


class FailureClusterResult(BaseModel):
    """
    total_failures:  number of examples with at least one wrong field.
    total_examples:  total number of examples evaluated.
    clusters:        failure clusters sorted by count descending.
    """
    total_failures: int
    total_examples: int
    clusters: list[FailureCluster]

    @property
    def failure_rate(self) -> float:
        return self.total_failures / self.total_examples if self.total_examples > 0 else _ZERO_FLOAT
