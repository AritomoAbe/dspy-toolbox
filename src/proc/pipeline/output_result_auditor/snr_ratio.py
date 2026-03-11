import logging
from typing import Any

import numpy as np
from returns.result import Result, Success

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.output_result_auditor.models import SNRResult
from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor

_EPSILON: float = 1e-9
_DEFAULT_RUNS: int = 5


class SignalToNoiseRatio(ProcNode):

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def invoke(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
        n_runs: int = _DEFAULT_RUNS,
    ) -> Result[SNRResult, ProcError]:
        per_example_scores: list[list[float]] = []
        for example in dataset.load():
            run_scores: list[float] = []
            for _ in range(n_runs):
                pred = llm(**example.inputs())
                score = scorer.extraction_metric(example, pred)
                run_scores.append(score)
            per_example_scores.append(run_scores)

        variances = [float(np.var(s)) for s in per_example_scores]
        means = [float(np.mean(s)) for s in per_example_scores]

        avg_variance = float(np.mean(variances))
        avg_signal = float(np.mean(means))
        snr = avg_signal / (float(np.sqrt(avg_variance)) + _EPSILON)

        self._logger.info(f"Avg variance per example: {avg_variance:.4f}")
        self._logger.info(f"SNR: {snr:.2f}   (>5 is healthy, <2 is problematic)")

        return Success(SNRResult(snr=snr, avg_variance=avg_variance))
