from typing import Any

from pydantic import BaseModel

from proc.pipeline.training_set_auditor.enums import (
    CorrelationRisk,
    FieldType,
    LearnabilityLevel,
    SeparabilityLevel,
)


class CategoryEntry(BaseModel):
    n: int
    pct: float


class NumericStats(BaseModel):
    count: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    unique_values: list[float]
    unique_count: int
    balance_score: float
    majority_pct: float


class CategoricalStats(BaseModel):
    count: int
    unique_count: int
    distribution: dict[str, CategoryEntry]
    balance_score: float
    majority_pct: float


class ListStats(BaseModel):
    count: int
    empty_pct: float
    non_empty_samples: list[Any]
    balance_score: float
    majority_pct: float


class FreeTextStats(BaseModel):
    count: int
    unique_count: int
    unique_ratio: float
    avg_length: float
    samples: list[str]
    balance_score: float
    majority_pct: float


FieldStats = NumericStats | CategoricalStats | ListStats | FreeTextStats


class FieldAnalysis(BaseModel):
    type: FieldType
    stats: FieldStats


class CoOccurrenceEntry(BaseModel):
    field_a: str
    field_b: str
    dominant_pair: tuple[str, str]
    dominant_pct: float
    risk: CorrelationRisk


class NearDuplicatePair(BaseModel):
    index_a: int
    index_b: int
    score: float


class FieldDuplicateStats(BaseModel):
    total: int
    near_duplicate_pairs: int
    near_duplicate_rate_pct: float
    examples: list[NearDuplicatePair]


class ClassifierError(BaseModel):
    error: str


class ClassifierStats(BaseModel):
    accuracy: float
    std: float
    chance_baseline: float
    lift_over_chance: float
    n_folds: int
    n_classes: int
    learnability: LearnabilityLevel


ClassifierResult = ClassifierError | ClassifierStats


class ClassPairSeparability(BaseModel):
    pair: tuple[str, str]
    centroid_distance: float
    separability: SeparabilityLevel


class SeparabilityStats(BaseModel):
    pairwise: list[ClassPairSeparability]
    cohesion_per_class: dict[str, float]
    mean_inter_class_distance: float


class KeywordScore(BaseModel):
    keyword: str
    score: float


class FieldSignalStats(BaseModel):
    proxy_classifier: ClassifierResult
    separability: SeparabilityStats
    discriminating_keywords: dict[str, list[KeywordScore]]


class SkippedField(BaseModel):
    skipped: str


FieldSignalResult = FieldSignalStats | SkippedField
