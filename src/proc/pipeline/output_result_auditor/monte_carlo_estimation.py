import logging
from typing import Any

import dspy
import numpy as np
from returns.result import Result, Success

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.output_result_auditor.contexts import MonteCarloContext
from proc.pipeline.output_result_auditor.models import MonteCarloResult
from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor

_EASY_THRESHOLD: float = 0.8
_HARD_THRESHOLD: float = 0.3
_DEFAULT_SAMPLES: int = 20
_DETAIL_SAMPLES: int = 10
_PREVIEW_LEN: int = 60


class MonteCarloEstimation(ProcNode):

    _SCORE: float = 1.0

    def __init__(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._dataset = dataset
        self._llm = llm
        self._scorer = scorer

    def invoke(self) -> Result[ProcScore, ProcError]:
        if self._llm.config.temperature == 0:
            self._logger.warning("LLM model temperature is 0. SNR will produce useless data.")

        probs_list: list[float] = []
        for index, ex in enumerate(self._dataset.load()):
            self._logger.info(f"Running example {index}")
            prob = self._estimate_pass_probability(ex, n_samples=_DETAIL_SAMPLES)
            self._logger.info(f"Finished example {index} with prob={prob:.2f}")
            probs_list.append(prob)

        probs = np.array(probs_list)
        above_hard = probs >= _HARD_THRESHOLD
        below_easy = probs <= _EASY_THRESHOLD
        easy = int((probs > _EASY_THRESHOLD).sum())
        medium = int((above_hard & below_easy).sum())
        hard = int((probs < _HARD_THRESHOLD).sum())

        self._logger.info(f"Easy (>80%):    {easy}")
        self._logger.info(f"Medium (30-80%): {medium}")
        self._logger.info(f"Hard (<30%):    {hard}")

        result = MonteCarloResult(easy=easy, medium=medium, hard=hard, probs=probs_list)
        return Success(ProcScore(value=self._score(), context=MonteCarloContext(result)))

    def _score(self) -> float:
        return self._SCORE

    def _estimate_pass_probability(
        self,
        example: dspy.Example,
        n_samples: int = _DEFAULT_SAMPLES,
    ) -> float:
        passes = 0
        for n_sample in range(n_samples):
            self._logger.info(f"Executing iteration {n_sample}")
            pred = self._llm(**example.inputs())
            score = self._scorer.extraction_metric(example, pred)
            self._logger.info(f"Finished iteration {n_sample} with score = {score:.2f}")
            passes += int(score)
        return passes / n_samples
