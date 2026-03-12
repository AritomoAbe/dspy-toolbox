from typing import Any

from returns.result import Result, Success

from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.pipeline.dataset.training_dataset import TrainingSetDataset
from proc.pipeline.training_set_auditor.contexts import InputFieldsContext
from proc.pipeline.training_set_auditor.enums import FieldType, ListPresence
from proc.pipeline.training_set_auditor.models import (
    FieldAnalysis,
    FieldStats,
    FreeTextStats,
    ListStats,
)
from proc.pipeline.training_set_auditor.utils import FieldStatsUtils

_EXPECTED_KEY: str = "expected"


class AnalyzeInputFields(ProcNode):
    _SCORE: float = 1.0

    def __init__(self, dataset: TrainingSetDataset) -> None:
        self._dataset = dataset

    def invoke(self) -> Result[ProcScore, ProcError]:
        examples = self._dataset.load()
        records = [{k: v for k, v in ex.items()} for ex in examples]
        result = self._analyze_input_fields(records)
        return Success(ProcScore(value=self._score(), context=InputFieldsContext(result)))

    def _score(self) -> float:
        return self._SCORE

    def _build_stats_for_field(self, ftype: FieldType, values: list[Any]) -> FieldStats:
        if ftype == FieldType.numeric:
            return FieldStatsUtils.numeric_stats(values)
        if ftype == FieldType.list:
            n = len(values)
            empty_count = sum(1 for v in values if not v)
            presence: list[ListPresence] = [
                ListPresence.non_empty if v else ListPresence.empty
                for v in values
            ]
            return ListStats(
                count=n,
                empty_pct=round(empty_count / n * 100, 1),
                non_empty_samples=[v for v in values if v][:5],
                balance_score=FieldStatsUtils.balance_score(presence),
                majority_pct=FieldStatsUtils.majority_pct(presence),
            )
        if ftype == FieldType.free_text:
            n = len(values)
            str_values: list[str] = [str(v) for v in values]
            unique_strs: set[str] = set(str_values)
            return FreeTextStats(
                count=n,
                unique_count=len(unique_strs),
                unique_ratio=round(len(unique_strs) / n, 3),
                avg_length=round(sum(len(s) for s in str_values) / n, 1),
                samples=list(unique_strs)[:8],
                balance_score=1.0,  # free text — treat as diverse
                majority_pct=FieldStatsUtils.majority_pct(str_values),
            )
        return FieldStatsUtils.categorical_stats([str(v) for v in values])

    def _analyze_input_fields(
        self, records: list[dict[str, Any]],
    ) -> dict[str, FieldAnalysis]:
        field_values: dict[str, list[Any]] = {}

        for rec in records:
            for key, value in rec.items():
                if key == _EXPECTED_KEY:
                    continue
                field_values.setdefault(key, []).append(value)

        results: dict[str, FieldAnalysis] = {}
        for field, values in field_values.items():
            ftype = FieldStatsUtils.infer_field_type(values)
            results[field] = FieldAnalysis(
                type=ftype,
                stats=self._build_stats_for_field(ftype, values),
            )
        return results
