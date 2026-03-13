import logging
from collections import defaultdict
from typing import Any

from returns.result import Result, Success, Failure

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.output_result_auditor.contexts import FailureClusterContext
from proc.pipeline.output_result_auditor.models import (
    FailureClusterResult,
    FailureCluster,
    FailureExample,
)
from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor, INVALID_SCORE


class FailureClusterAuditor(ProcNode):
    """
    Collects all examples where at least one field is wrong, then groups
    them by their wrong-field pattern (the "cluster").

    Cluster key is the sorted, '+'-joined list of wrong field names.
    Examples:
        "urgency"                          → only urgency was wrong
        "sender_iana_timezone"             → only timezone was wrong
        "urgency+flexibility"              → both were wrong together
        "sender_iana_timezone+urgency"     → timezone and urgency wrong

    This reveals systemic failure modes rather than random errors.
    For example, if "sender_iana_timezone" dominates the clusters, the
    prompt's timezone extraction logic needs rework — not the whole prompt.
    """

    _SCORE: float = 1.0

    def __init__(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
        tracked_fields: list[str]
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._dataset = dataset
        self._llm = llm
        self._scorer = scorer
        self._tracked_fields = tracked_fields

    def invoke(self) -> Result[ProcScore, ProcError]:
        total = 0
        clusters: dict[str, list[FailureExample]] = defaultdict(list)

        for index, example in enumerate(self._dataset.load()):
            self._logger.info(f"Processing example {index}")
            pred = self._llm(**example.inputs())
            score = self._scorer.extraction_metric(example, pred)
            self._logger.info(f"Processed example {index} with score {score}")

            if score == INVALID_SCORE:
                return Failure(ProcError(f"Cannot score example {index}"))

            total += 1
            wrong_fields = self._get_wrong_fields(example, pred)
            if wrong_fields:
                self._add_failure(index, example, pred, wrong_fields, clusters)

        if total == 0:
            return Failure(ProcError("Dataset is empty"))

        return self._build_result(clusters, total)

    def _get_wrong_fields(self, example: Any, pred: Any) -> list[str]:
        wrong: list[str] = []
        for field in self._tracked_fields:
            expected_val = example.get(field)
            predicted_val = getattr(pred, field, None)
            if expected_val is None and predicted_val is None:
                continue
            if not self._scorer.field_metric(field, expected_val, predicted_val):
                wrong.append(field)
        return wrong

    def _add_failure(
        self,
        index: int,
        example: Any,
        pred: Any,
        wrong_fields: list[str],
        clusters: dict[str, list[FailureExample]],
    ) -> None:
        cluster_key = "+".join(sorted(wrong_fields))
        expected_fields: dict[str, Any] = {
            f: val for f in self._tracked_fields if (val := example.get(f)) is not None
        }
        failure = FailureExample(
            index=index,
            email_body=example.get("email_body", ""),
            expected=expected_fields,
            predicted={f: getattr(pred, f, None) for f in self._tracked_fields},
            wrong_fields=wrong_fields,
            cluster=cluster_key,
        )
        clusters[cluster_key].append(failure)
        self._logger.info(
            f"Example {index} — cluster='{cluster_key}', wrong_fields={wrong_fields}"
        )

    def _build_result(
        self,
        clusters: dict[str, list[FailureExample]],
        total: int,
    ) -> Result[ProcScore, ProcError]:
        total_failures = sum(len(v) for v in clusters.values())
        sorted_clusters = sorted(
            [
                FailureCluster(cluster=key, count=len(examples), examples=examples)
                for key, examples in clusters.items()
            ],
            key=lambda c: c.count,
            reverse=True,
        )
        self._logger.info(
            f"Failure rate: {total_failures}/{total} ({100 * total_failures / total:.1f}%)"
        )
        for cluster in sorted_clusters:
            self._logger.info(f"  [{cluster.count:>3}x] '{cluster.cluster}'")

        result = FailureClusterResult(
            total_failures=total_failures,
            total_examples=total,
            clusters=sorted_clusters,
        )
        return Success(ProcScore(value=self._score(), context=FailureClusterContext(result)))

    def _score(self) -> float:
        return self._SCORE
