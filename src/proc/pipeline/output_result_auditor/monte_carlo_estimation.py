import logging
from typing import Any

import dspy
import numpy as np
from returns.result import Result, Success

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.output_result_auditor.models import MonteCarloResult
from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor

_EASY_THRESHOLD: float = 0.8
_HARD_THRESHOLD: float = 0.3
_DEFAULT_SAMPLES: int = 20
_DETAIL_SAMPLES: int = 10


class MonteCarloEstimation(ProcNode):

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def invoke(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
    ) -> Result[MonteCarloResult, ProcError]:
        probs_list: list[float] = []
        for ex in dataset.load():
            prob = self._estimate_pass_probability(ex, llm, scorer, n_samples=_DETAIL_SAMPLES)
            probs_list.append(prob)
            preview = str(ex)[:60]
            self._logger.info(f"P(pass)={prob:.2f}  |  {preview}")

        probs = np.array(probs_list)
        above_hard = probs >= _HARD_THRESHOLD
        below_easy = probs <= _EASY_THRESHOLD
        easy = int((probs > _EASY_THRESHOLD).sum())
        medium = int((above_hard & below_easy).sum())
        hard = int((probs < _HARD_THRESHOLD).sum())

        self._logger.info(f"Easy (>80%):    {easy}")
        self._logger.info(f"Medium (30-80%): {medium}")
        self._logger.info(f"Hard (<30%):    {hard}")

        return Success(MonteCarloResult(easy=easy, medium=medium, hard=hard, probs=probs_list))

    def _estimate_pass_probability(
        self,
        example: dspy.Example,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
        n_samples: int = _DEFAULT_SAMPLES,
    ) -> float:
        passes = 0
        for _ in range(n_samples):
            pred = llm(**example.inputs())
            passes += int(scorer.extraction_metric(example, pred))
        return passes / n_samples
