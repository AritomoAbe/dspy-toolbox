from collections import Counter
from itertools import combinations
from operator import attrgetter
from typing import Any

from returns.result import Result, Success

from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.pipeline.dataset.training_dataset import TrainingSetDataset
from proc.pipeline.training_set_auditor.contexts import CoOccurrenceContext
from proc.pipeline.training_set_auditor.enums import CorrelationRisk, FieldType
from proc.pipeline.training_set_auditor.models import CoOccurrenceEntry
from proc.pipeline.training_set_auditor.utils import FieldStatsUtils

_CORRELATION_THRESHOLD: float = 0.7
_HIGH_RISK_THRESHOLD: float = 0.85
_EXPECTED_KEY: str = "expected"


class AnalyzeCoOccurrence(ProcNode):
    _SCORE: float = 1.0

    def __init__(self, dataset: TrainingSetDataset) -> None:
        self._dataset = dataset

    def invoke(self) -> Result[ProcScore, ProcError]:
        examples = self._dataset.load()
        records = [{k: v for k, v in ex.items()} for ex in examples]
        result = self._analyze_co_occurrence(records)
        return Success(ProcScore(value=self._score(), context=CoOccurrenceContext(result)))

    def _score(self) -> float:
        return self._SCORE

    def _collect_field_values(
        self, records: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        field_values: dict[str, list[Any]] = {}
        for rec in records:
            expected = rec.get(_EXPECTED_KEY, {})
            if not isinstance(expected, dict):
                continue
            for field, value in expected.items():
                field_values.setdefault(field, []).append(str(value))
        return field_values

    def _find_categorical_fields(
        self, field_values: dict[str, list[Any]],
    ) -> list[str]:
        result: list[str] = []
        for field, vals in field_values.items():
            is_categorical = FieldStatsUtils.infer_field_type(vals) == FieldType.categorical
            has_variety = len(set(vals)) > 1
            if is_categorical and has_variety:
                result.append(field)
        return result

    def _build_entry(
        self,
        f1: str,
        f2: str,
        values1: list[Any],
        values2: list[Any],
    ) -> CoOccurrenceEntry | None:
        pairs: list[tuple[Any, Any]] = list(zip(values1, values2))
        pair_counts: Counter[tuple[Any, Any]] = Counter(pairs)
        total = len(pairs)
        top_pair, top_count = pair_counts.most_common(1)[0]
        most_common_pct = top_count / total
        if most_common_pct <= _CORRELATION_THRESHOLD:
            return None
        if most_common_pct > _HIGH_RISK_THRESHOLD:
            risk = CorrelationRisk.high
        else:
            risk = CorrelationRisk.moderate
        return CoOccurrenceEntry(
            field_a=f1,
            field_b=f2,
            dominant_pair=(str(top_pair[0]), str(top_pair[1])),
            dominant_pct=round(most_common_pct * 100, 1),
            risk=risk,
        )

    def _analyze_co_occurrence(
        self, records: list[dict[str, Any]],
    ) -> list[CoOccurrenceEntry]:
        field_values = self._collect_field_values(records)
        categorical_fields = self._find_categorical_fields(field_values)
        correlations: list[CoOccurrenceEntry] = []
        for f1, f2 in combinations(categorical_fields, 2):
            entry = self._build_entry(f1, f2, field_values[f1], field_values[f2])
            if entry is not None:
                correlations.append(entry)
        return sorted(correlations, key=attrgetter('dominant_pct'), reverse=True)
