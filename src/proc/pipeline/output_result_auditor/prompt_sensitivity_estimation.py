import logging
import random
from collections import Counter
from typing import Any

import dspy
import numpy as np
from returns.result import Result, Success, Failure

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.output_result_auditor.contexts import PromptSensitivityContext
from proc.pipeline.output_result_auditor.models import PromptSensitivityResult
from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor, INVALID_SCORE

_DEFAULT_SHUFFLES: int = 5


class PromptSensitivityAuditor(ProcNode):
    """
    Measures how sensitive the compiled prompt is to the ordering of
    few-shot demonstrations. At temperature=0 the model is deterministic,
    so the only source of output variance is the ordering of demos in the
    prompt. A robust compiled prompt should produce the same answer
    regardless of which order the bootstrapped examples appear in.

    Metric — consistency per example:
        consistency = mode_count / n_shuffles
        e.g. if 4/5 shuffles produce the same output → consistency = 0.8

    Thresholds:
        avg_consistency > 0.9  →  healthy   (prompt is stable)
        avg_consistency < 0.7  →  brittle   (prompt is over-fitted to demo order)
    """

    _SCORE: float = 1.0

    def __init__(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
        n_shuffles: int = _DEFAULT_SHUFFLES,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._dataset = dataset
        self._llm = llm
        self._scorer = scorer
        self._n_shuffles = n_shuffles

    def invoke(self) -> Result[ProcScore, ProcError]:
        predictors = self._llm.predictors()

        if not predictors:
            return Failure(ProcError("No predictors found in LLM — cannot access few-shot demos"))

        original_demos = {p: list(p.demos) for p in predictors}

        if all(len(d) == 0 for d in original_demos.values()):
            return Failure(ProcError(
                "Predictors have no demos — run BootstrapFewShot before PromptSensitivityAuditor"
            ))

        per_example_consistency: list[float] = []
        per_example_flip_rate: list[float] = []
        proc_error: ProcError | None = None

        for index, example in enumerate(self._dataset.load()):
            self._logger.info(f"Processing example {index}")
            run_scores, error = self._run_shuffles(index, example, predictors, original_demos)
            if error is not None:
                proc_error = error
                break
            consistency = self._consistency(run_scores)
            flip_rate = 1.0 - consistency
            per_example_consistency.append(consistency)
            per_example_flip_rate.append(flip_rate)
            self._logger.info(
                f"Example {index} — consistency: {consistency:.2f}, "
                f"flip_rate: {flip_rate:.2f}, scores: {run_scores}"
            )

        for predictor, demos in original_demos.items():
            predictor.demos = demos

        if proc_error is not None:
            return Failure(proc_error)

        avg_consistency = float(np.mean(per_example_consistency))
        avg_flip_rate = float(np.mean(per_example_flip_rate))

        self._logger.info(f"Avg consistency: {avg_consistency:.4f}  (>0.9 healthy, <0.7 brittle)")
        self._logger.info(f"Avg flip rate:   {avg_flip_rate:.4f}")

        result = PromptSensitivityResult(
            avg_consistency=avg_consistency,
            avg_flip_rate=avg_flip_rate,
            n_shuffles=self._n_shuffles,
        )
        return Success(ProcScore(value=self._score(), context=PromptSensitivityContext(result)))

    def _run_shuffles(
        self,
        index: int,
        example: dspy.Example,
        predictors: list[Any],
        original_demos: dict[Any, list[Any]],
    ) -> tuple[list[float], ProcError | None]:
        run_scores: list[float] = []
        for n_shuffle in range(self._n_shuffles):
            self._logger.info(f"Shuffle {index}/{n_shuffle}")
            for predictor in predictors:
                shuffled = original_demos[predictor].copy()
                random.shuffle(shuffled)
                predictor.demos = shuffled
            with dspy.context(cache=False):
                pred = self._llm(**example.inputs())
            score = self._scorer.extraction_metric(example, pred)
            self._logger.info(f"Shuffle done {index}/{n_shuffle} with score {score:.2f}")
            if score == INVALID_SCORE:
                return [], ProcError(
                    f"Cannot calculate sensitivity for example {index}, shuffle {n_shuffle}"
                )
            run_scores.append(score)
        return run_scores, None

    def _consistency(self, scores: list[float]) -> float:
        """Fraction of runs that agree with the majority output."""
        if not scores:
            return 0.0  # noqa: WPS358
        mode_count = Counter(scores).most_common(1)[0][1]
        return mode_count / len(scores)

    def _score(self) -> float:
        return self._SCORE
