import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import describe as scipy_describe
from scipy.stats import entropy as scipy_entropy

from proc.pipeline.training_set_auditor.enums import FieldType
from proc.pipeline.training_set_auditor.models import (
    CategoricalStats,
    CategoryEntry,
    NumericStats,
)

_FREE_TEXT_RATIO_THRESHOLD: float = 0.6
_UNIQUE_DISPLAY_CAP: int = 20


class FieldStatsUtils:

    @classmethod
    def balance_score(cls, values: list[Any]) -> float:
        """1.0 = perfectly balanced across classes, 0.0 = all same value."""
        n_unique = len(set(values))
        if n_unique <= 1:
            return 0
        counts: list[int] = list(Counter(values).values())
        entropy_val = float(scipy_entropy(counts, base=2))
        return round(entropy_val / math.log2(n_unique), 3)

    @classmethod
    def majority_pct(cls, values: list[Any]) -> float:
        if not values:
            return 0
        mode_count = Counter(values).most_common(1)[0][1]
        return round(mode_count / len(values) * 100, 1)

    @classmethod
    def infer_field_type(cls, values: list[Any]) -> FieldType:
        non_none: list[Any] = [v for v in values if not cls._is_effectively_empty(v)]
        if not non_none:
            return FieldType.empty
        if all(isinstance(v, (int, float)) for v in non_none):
            return FieldType.numeric
        if all(isinstance(v, list) for v in non_none):
            return FieldType.list
        unique_str_count = len(set(str(v) for v in non_none))
        unique_ratio = unique_str_count / len(non_none)
        if unique_ratio > _FREE_TEXT_RATIO_THRESHOLD:
            return FieldType.free_text
        return FieldType.categorical

    @classmethod
    def numeric_stats(cls, values: list[int | float]) -> NumericStats:
        arr = np.array(values, dtype=float)
        desc = scipy_describe(arr)
        unique = np.unique(arr)
        return NumericStats(
            count=int(desc.nobs),
            mean=round(float(desc.mean), 2),
            std=round(float(np.sqrt(desc.variance)), 2),
            median=float(np.median(arr)),
            min=float(desc.minmax[0]),
            max=float(desc.minmax[1]),
            unique_values=unique[:_UNIQUE_DISPLAY_CAP].tolist(),  # cap display
            unique_count=int(len(unique)),
            balance_score=cls.balance_score([str(v) for v in values]),
            majority_pct=cls.majority_pct([str(v) for v in values]),
        )

    @classmethod
    def categorical_stats(cls, values: list[str]) -> CategoricalStats:
        counts = pd.Series(values).value_counts()
        total = len(values)
        return CategoricalStats(
            count=total,
            unique_count=int(len(counts)),
            distribution={
                str(k): cls._make_category_entry(int(v), total)
                for k, v in counts.items()
            },
            balance_score=cls.balance_score(values),
            majority_pct=cls.majority_pct(values),
        )

    @classmethod
    def _is_effectively_empty(cls, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return len(value) == 0
        if isinstance(value, list):
            return len(value) == 0
        return False

    @classmethod
    def _make_category_entry(cls, count: int, total: int) -> CategoryEntry:
        return CategoryEntry(n=count, pct=round(count / total * 100, 1))
