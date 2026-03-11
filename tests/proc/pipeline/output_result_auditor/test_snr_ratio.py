from unittest.mock import MagicMock

import pytest
from returns.pipeline import is_successful
from returns.result import Result

from proc.base.proc_error import ProcError
from proc.pipeline.output_result_auditor.models import SNRResult
from proc.pipeline.output_result_auditor.snr_ratio import SignalToNoiseRatio

_N_EXAMPLES: int = 3
_N_RUNS: int = 4


def _make_dataset(n: int = _N_EXAMPLES) -> MagicMock:
    dataset = MagicMock()
    examples = [MagicMock(inputs=MagicMock(return_value={})) for _ in range(n)]
    dataset.load.return_value = examples
    return dataset


def _make_scorer(score: float = 0.9) -> MagicMock:
    scorer = MagicMock()
    scorer.extraction_metric.return_value = score
    return scorer


def _invoke(n: int = _N_EXAMPLES, score: float = 0.9, n_runs: int = _N_RUNS) -> Result[SNRResult, ProcError]:
    llm = MagicMock(return_value=MagicMock())
    return SignalToNoiseRatio().invoke(_make_dataset(n), llm, _make_scorer(score), n_runs=n_runs)


class TestSignalToNoiseRatio:

    def test_invoke_returns_success(self) -> None:
        assert is_successful(_invoke())

    def test_invoke_returns_snr_result(self) -> None:
        assert isinstance(_invoke().unwrap(), SNRResult)

    def test_snr_is_float(self) -> None:
        assert isinstance(_invoke().unwrap().snr, float)

    def test_avg_variance_is_float(self) -> None:
        assert isinstance(_invoke().unwrap().avg_variance, float)

    def test_constant_score_gives_zero_variance(self) -> None:
        result = _invoke(score=1.0).unwrap()
        assert result.avg_variance == pytest.approx(0)

    def test_constant_score_gives_positive_snr(self) -> None:
        assert _invoke(score=1.0).unwrap().snr > 0

    def test_scorer_called_per_run_per_example(self) -> None:
        scorer = _make_scorer()
        llm = MagicMock(return_value=MagicMock())
        SignalToNoiseRatio().invoke(_make_dataset(), llm, scorer, n_runs=_N_RUNS)
        assert scorer.extraction_metric.call_count == _N_EXAMPLES * _N_RUNS
