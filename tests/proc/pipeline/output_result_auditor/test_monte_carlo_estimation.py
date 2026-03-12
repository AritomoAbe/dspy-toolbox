from unittest.mock import MagicMock

import pytest
from returns.pipeline import is_successful
from returns.result import Result

from proc.base.proc_error import ProcError
from proc.base.proc_score import ProcScore
from proc.pipeline.output_result_auditor.contexts import MonteCarloContext
from proc.pipeline.output_result_auditor.models import MonteCarloResult
from proc.pipeline.output_result_auditor.monte_carlo_estimation import MonteCarloEstimation, _DETAIL_SAMPLES

_N_EXAMPLES: int = 5
_DEFAULT_SCORE: float = 1.0


def _make_dataset(n: int = _N_EXAMPLES) -> MagicMock:
    dataset = MagicMock()
    examples = [MagicMock(inputs=MagicMock(return_value={})) for _ in range(n)]
    dataset.load.return_value = examples
    return dataset


def _make_scorer(score: float = _DEFAULT_SCORE) -> MagicMock:
    scorer = MagicMock()
    scorer.extraction_metric.return_value = score
    return scorer


def _invoke(n: int = _N_EXAMPLES, score: float = _DEFAULT_SCORE) -> Result[ProcScore, ProcError]:
    llm = MagicMock(return_value=MagicMock())
    return MonteCarloEstimation(_make_dataset(n), llm, _make_scorer(score)).invoke()


def _mc_result(n: int = _N_EXAMPLES, score: float = _DEFAULT_SCORE) -> MonteCarloResult:
    ctx = _invoke(n, score).unwrap().context
    assert isinstance(ctx, MonteCarloContext)
    return ctx.result


@pytest.fixture(scope="module")
def result() -> MonteCarloResult:
    return _mc_result()


class TestInvoke:

    def test_returns_success(self) -> None:
        assert is_successful(_invoke())

    def test_returns_proc_score(self) -> None:
        assert isinstance(_invoke().unwrap(), ProcScore)

    def test_score_is_one(self) -> None:
        assert _invoke().unwrap().value == pytest.approx(1.0)

    def test_context_is_monte_carlo_context(self) -> None:
        assert isinstance(_invoke().unwrap().context, MonteCarloContext)

    def test_probs_length(self, result: MonteCarloResult) -> None:
        assert len(result.probs) == _N_EXAMPLES

    def test_probs_are_floats(self, result: MonteCarloResult) -> None:
        assert all(isinstance(p, float) for p in result.probs)

    def test_counts_sum_to_n_examples(self, result: MonteCarloResult) -> None:
        assert result.easy + result.medium + result.hard == _N_EXAMPLES

    def test_scorer_call_count(self) -> None:
        scorer = _make_scorer()
        llm = MagicMock(return_value=MagicMock())
        MonteCarloEstimation(_make_dataset(), llm, scorer).invoke()
        assert scorer.extraction_metric.call_count == _N_EXAMPLES * _DETAIL_SAMPLES


class TestClassification:

    def test_all_easy_when_perfect_score(self) -> None:
        assert _mc_result().easy == _N_EXAMPLES

    def test_all_hard_when_zero_score(self) -> None:
        assert _mc_result(score=0).hard == _N_EXAMPLES
