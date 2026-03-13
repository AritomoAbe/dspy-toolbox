import logging
from typing import Any

from returns.result import Result, Success, Failure

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.output_result_auditor.contexts import AccuracyContext
from proc.pipeline.output_result_auditor.models import AccuracyResult, FieldAccuracy
from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor, INVALID_SCORE

_HEALTHY_THRESHOLD: float = 0.9
_NEEDS_ATTENTION_THRESHOLD: float = 0.7

_HEALTHY_MARKER: str = "✓"
_BORDERLINE_MARKER: str = "~"
_FAILING_MARKER: str = "✗"


class AccuracyAuditor(ProcNode):
    """
    Measures per-field and overall accuracy of the compiled prompt
    on a dataset at temperature=0 (single deterministic pass).

    Overall accuracy counts an example as correct only if ALL fields match.
    Per-field accuracy shows which fields are the weakest individually,
    which is far more actionable than a single aggregate score.

    Thresholds (per field):
        accuracy > 0.9  →  healthy
        accuracy < 0.7  →  needs attention
    """

    _SCORE: float = 1.0

    def __init__(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
        tracked_fields: list[str],
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._dataset = dataset
        self._llm = llm
        self._scorer = scorer
        self._tracked_fields = tracked_fields

    def invoke(self) -> Result[ProcScore, ProcError]:
        total = 0
        overall_correct = 0
        field_correct: dict[str, int] = {f: 0 for f in self._tracked_fields}
        field_total: dict[str, int] = {f: 0 for f in self._tracked_fields}

        for index, example in enumerate(self._dataset.load()):
            self._logger.info(f"Processing example {index}")
            pred = self._llm(**example.inputs())
            score = self._scorer.extraction_metric(example, pred)
            self._logger.info(f"Processed example {index} with score {score}")

            if score == INVALID_SCORE:
                return Failure(ProcError(f"Cannot score example {index}"))

            total += 1
            if self._score_fields(index, example, pred, field_correct, field_total):
                overall_correct += 1

        if total == 0:
            return Failure(ProcError("Dataset is empty"))

        overall_accuracy = overall_correct / total
        per_field = self._build_per_field(field_correct, field_total)
        self._log_results(overall_accuracy, overall_correct, total, per_field)

        result = AccuracyResult(
            overall_accuracy=overall_accuracy,
            per_field=per_field,
            total_examples=total,
        )
        return Success(ProcScore(value=self._score(), context=AccuracyContext(result)))

    def _score_fields(
        self,
        index: int,
        example: Any,
        pred: Any,
        field_correct: dict[str, int],
        field_total: dict[str, int],
    ) -> bool:
        all_correct = True
        for field in self._tracked_fields:
            expected_val = example.get(field)
            predicted_val = getattr(pred, field, None)

            if expected_val is None and predicted_val is None:
                continue

            field_total[field] += 1
            if self._scorer.field_metric(field, expected_val, predicted_val):
                field_correct[field] += 1
            else:
                all_correct = False
                self._logger.info(
                    f"Example {index} — WRONG field '{field}': "
                    f"expected={expected_val!r}, predicted={predicted_val!r}"
                )
        return all_correct

    def _build_per_field(
        self,
        field_correct: dict[str, int],
        field_total: dict[str, int],
    ) -> dict[str, FieldAccuracy]:
        return {
            f: FieldAccuracy(field_name=f, correct=field_correct[f], total=field_total[f])
            for f in self._tracked_fields
            if field_total[f] > 0
        }

    def _log_results(
        self,
        overall_accuracy: float,
        overall_correct: int,
        total: int,
        per_field: dict[str, FieldAccuracy],
    ) -> None:
        self._logger.info(f"Overall accuracy: {overall_accuracy:.4f}  ({overall_correct}/{total})")
        for f, fa in per_field.items():
            marker = self._field_marker(fa.accuracy)
            acc_str = f"{fa.accuracy:.4f} ({fa.correct}/{fa.total})"
            self._logger.info(f"  [{marker}] {f}: {acc_str}")

    def _field_marker(self, accuracy: float) -> str:
        if accuracy >= _HEALTHY_THRESHOLD:
            return _HEALTHY_MARKER
        if accuracy >= _NEEDS_ATTENTION_THRESHOLD:
            return _BORDERLINE_MARKER
        return _FAILING_MARKER

    def _score(self) -> float:
        return self._SCORE
