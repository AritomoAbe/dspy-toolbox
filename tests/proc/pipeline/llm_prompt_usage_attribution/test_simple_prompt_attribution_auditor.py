from unittest.mock import MagicMock, call

import pytest
from returns.pipeline import is_successful

from proc.base.proc_score import ProcScore
from proc.pipeline.llm_prompt_usage_attribution.contexts import AttributionContext
from proc.pipeline.llm_prompt_usage_attribution.models import AttributionResult
from proc.pipeline.llm_prompt_usage_attribution.simple_prompt_attribution_auditor import (
    SimplePromptAttributionAuditor,
    _ZERO_SCORE,
    _MASK_PLACEHOLDER,
)
from proc.pipeline.output_result_auditor.score_extractor import INVALID_SCORE

_N_DEMOS: int = 2
_INSTRUCTION_TEXT: str = "Test instruction text"

# Score sequence for 1 example × 2 demos:
# [baseline, instruction_ablated, demo0_ablated, demo1_ablated]
_SCORE_FULL: float = 0.8
_SCORE_INSTR_ABLATED: float = 0.6
_SCORE_DEMO0_ABLATED: float = 0.7
_SCORE_DEMO1_ABLATED: float = 0.5

_ATTRIBUTION_INSTR: float = _SCORE_FULL - _SCORE_INSTR_ABLATED
_ATTRIBUTION_DEMO0: float = _SCORE_FULL - _SCORE_DEMO0_ABLATED
_ATTRIBUTION_DEMO1: float = _SCORE_FULL - _SCORE_DEMO1_ABLATED
_AVG_DEMO_ATTRIBUTION: float = (_ATTRIBUTION_DEMO0 + _ATTRIBUTION_DEMO1) / 2


def _make_predictor(n_demos: int = _N_DEMOS) -> MagicMock:
    predictor = MagicMock()
    predictor.demos = [MagicMock() for _ in range(n_demos)]
    sig = MagicMock()
    sig.instructions = _INSTRUCTION_TEXT
    sig.with_instructions.return_value = sig
    predictor.signature = sig
    return predictor


def _make_llm(predictor: MagicMock) -> MagicMock:
    llm = MagicMock(return_value=MagicMock())
    llm.predictors.return_value = [predictor]
    return llm


def _make_dataset(n: int = 1) -> MagicMock:
    dataset = MagicMock()
    dataset.load.return_value = [
        MagicMock(inputs=MagicMock(return_value={})) for _ in range(n)
    ]
    return dataset


def _make_scorer(score: float = _SCORE_FULL) -> MagicMock:
    scorer = MagicMock()
    scorer.extraction_metric.return_value = score
    return scorer


def _make_scorer_with_sequence(*scores: float) -> MagicMock:
    scorer = MagicMock()
    scorer.extraction_metric.side_effect = list(scores)
    return scorer


def _invoke_default() -> tuple[SimplePromptAttributionAuditor, MagicMock, MagicMock]:
    """Returns (auditor, predictor, llm) using the default 4-score sequence."""
    predictor = _make_predictor()
    llm = _make_llm(predictor)
    scorer = _make_scorer_with_sequence(
        _SCORE_FULL, _SCORE_INSTR_ABLATED, _SCORE_DEMO0_ABLATED, _SCORE_DEMO1_ABLATED,
    )
    auditor = SimplePromptAttributionAuditor(_make_dataset(1), llm, scorer)
    return auditor, predictor, llm


def _attribution_result() -> AttributionResult:
    auditor, _, _ = _invoke_default()
    ctx = auditor.invoke().unwrap().context
    assert isinstance(ctx, AttributionContext)
    return ctx.result


class TestInvokeFailures:

    def test_no_predictors_returns_failure(self) -> None:
        llm = MagicMock(return_value=MagicMock())
        llm.predictors.return_value = []
        auditor = SimplePromptAttributionAuditor(_make_dataset(), llm, _make_scorer())
        assert not is_successful(auditor.invoke())

    def test_no_demos_returns_failure(self) -> None:
        predictor = _make_predictor(n_demos=0)
        llm = _make_llm(predictor)
        auditor = SimplePromptAttributionAuditor(_make_dataset(), llm, _make_scorer())
        assert not is_successful(auditor.invoke())

    def test_invalid_baseline_score_returns_failure(self) -> None:
        predictor = _make_predictor()
        llm = _make_llm(predictor)
        scorer = _make_scorer(score=INVALID_SCORE)
        auditor = SimplePromptAttributionAuditor(_make_dataset(), llm, scorer)
        assert not is_successful(auditor.invoke())

    def test_no_predictors_error_message(self) -> None:
        llm = MagicMock(return_value=MagicMock())
        llm.predictors.return_value = []
        result = SimplePromptAttributionAuditor(_make_dataset(), llm, _make_scorer()).invoke()
        assert "predictors" in (result.failure().message or "").lower()

    def test_no_demos_error_message(self) -> None:
        predictor = _make_predictor(n_demos=0)
        result = SimplePromptAttributionAuditor(
            _make_dataset(), _make_llm(predictor), _make_scorer()
        ).invoke()
        assert "demos" in (result.failure().message or "").lower()


class TestInvokeSuccess:

    def test_returns_success(self) -> None:
        auditor, _, _ = _invoke_default()
        assert is_successful(auditor.invoke())

    def test_returns_proc_score(self) -> None:
        auditor, _, _ = _invoke_default()
        assert isinstance(auditor.invoke().unwrap(), ProcScore)

    def test_score_value_is_one(self) -> None:
        auditor, _, _ = _invoke_default()
        assert auditor.invoke().unwrap().value == pytest.approx(1.0)

    def test_context_is_attribution_context(self) -> None:
        auditor, _, _ = _invoke_default()
        assert isinstance(auditor.invoke().unwrap().context, AttributionContext)

    def test_ablation_invalid_score_still_succeeds(self) -> None:
        # Ablation returning INVALID_SCORE is treated as score=0.0, not a fatal error
        predictor = _make_predictor()
        llm = _make_llm(predictor)
        scorer = _make_scorer_with_sequence(
            _SCORE_FULL, INVALID_SCORE, _SCORE_DEMO0_ABLATED, _SCORE_DEMO1_ABLATED,
        )
        auditor = SimplePromptAttributionAuditor(_make_dataset(1), llm, scorer)
        assert is_successful(auditor.invoke())

    def test_ablation_invalid_score_gives_max_attribution(self) -> None:
        # attribution = score_full - 0.0 = score_full when ablation breaks the model
        predictor = _make_predictor()
        llm = _make_llm(predictor)
        scorer = _make_scorer_with_sequence(
            _SCORE_FULL, INVALID_SCORE, _SCORE_DEMO0_ABLATED, _SCORE_DEMO1_ABLATED,
        )
        auditor = SimplePromptAttributionAuditor(_make_dataset(1), llm, scorer)
        ctx = auditor.invoke().unwrap().context
        assert isinstance(ctx, AttributionContext)
        ea = ctx.result.example_attributions[0]
        assert ea.instruction_attribution == pytest.approx(_SCORE_FULL - _ZERO_SCORE)

    def test_demo_ablation_invalid_score_still_succeeds(self) -> None:
        predictor = _make_predictor()
        llm = _make_llm(predictor)
        scorer = _make_scorer_with_sequence(
            _SCORE_FULL, _SCORE_INSTR_ABLATED, INVALID_SCORE, _SCORE_DEMO1_ABLATED,
        )
        auditor = SimplePromptAttributionAuditor(_make_dataset(1), llm, scorer)
        assert is_successful(auditor.invoke())

    def test_demo_ablation_invalid_score_gives_max_attribution(self) -> None:
        predictor = _make_predictor()
        llm = _make_llm(predictor)
        scorer = _make_scorer_with_sequence(
            _SCORE_FULL, _SCORE_INSTR_ABLATED, INVALID_SCORE, _SCORE_DEMO1_ABLATED,
        )
        auditor = SimplePromptAttributionAuditor(_make_dataset(1), llm, scorer)
        ctx = auditor.invoke().unwrap().context
        assert isinstance(ctx, AttributionContext)
        ea = ctx.result.example_attributions[0]
        assert ea.demo_attributions[0] == pytest.approx(_SCORE_FULL - _ZERO_SCORE)


class TestCallCounts:

    def test_llm_called_once_per_ablation_per_example(self) -> None:
        # 1 example × (1 baseline + 1 instruction + n_demos) = 1 × (2 + 2) = 4
        predictor = _make_predictor(_N_DEMOS)
        llm = _make_llm(predictor)
        scorer = _make_scorer_with_sequence(
            _SCORE_FULL, _SCORE_INSTR_ABLATED, _SCORE_DEMO0_ABLATED, _SCORE_DEMO1_ABLATED,
        )
        SimplePromptAttributionAuditor(_make_dataset(1), llm, scorer).invoke()
        assert llm.call_count == 2 + _N_DEMOS

    def test_scorer_called_once_per_ablation_per_example(self) -> None:
        predictor = _make_predictor(_N_DEMOS)
        llm = _make_llm(predictor)
        scorer = _make_scorer_with_sequence(
            _SCORE_FULL, _SCORE_INSTR_ABLATED, _SCORE_DEMO0_ABLATED, _SCORE_DEMO1_ABLATED,
        )
        SimplePromptAttributionAuditor(_make_dataset(1), llm, scorer).invoke()
        assert scorer.extraction_metric.call_count == 2 + _N_DEMOS

    def test_llm_scales_with_example_count(self) -> None:
        n_examples = 3
        predictor = _make_predictor(_N_DEMOS)
        llm = _make_llm(predictor)
        calls_per_example = 2 + _N_DEMOS
        scorer = _make_scorer_with_sequence(
            *([_SCORE_FULL, _SCORE_INSTR_ABLATED, _SCORE_DEMO0_ABLATED, _SCORE_DEMO1_ABLATED] * n_examples)
        )
        SimplePromptAttributionAuditor(_make_dataset(n_examples), llm, scorer).invoke()
        assert llm.call_count == n_examples * calls_per_example


class TestAttributionScores:

    def test_n_examples(self) -> None:
        assert _attribution_result().n_examples == 1

    def test_n_demos(self) -> None:
        assert _attribution_result().n_demos == _N_DEMOS

    def test_avg_instruction_attribution(self) -> None:
        assert _attribution_result().avg_instruction_attribution == pytest.approx(_ATTRIBUTION_INSTR)

    def test_avg_demo_attribution(self) -> None:
        assert _attribution_result().avg_demo_attribution == pytest.approx(_AVG_DEMO_ATTRIBUTION)

    def test_per_demo_avg_attribution_has_correct_keys(self) -> None:
        assert set(_attribution_result().per_demo_avg_attribution.keys()) == {0, 1}

    def test_per_demo0_avg_attribution(self) -> None:
        assert _attribution_result().per_demo_avg_attribution[0] == pytest.approx(_ATTRIBUTION_DEMO0)

    def test_per_demo1_avg_attribution(self) -> None:
        assert _attribution_result().per_demo_avg_attribution[1] == pytest.approx(_ATTRIBUTION_DEMO1)

    def test_example_attribution_score_full(self) -> None:
        ea = _attribution_result().example_attributions[0]
        assert ea.score_full == pytest.approx(_SCORE_FULL)

    def test_example_attribution_instruction_score(self) -> None:
        ea = _attribution_result().example_attributions[0]
        assert ea.instruction_attribution == pytest.approx(_ATTRIBUTION_INSTR)

    def test_example_attribution_demo_scores(self) -> None:
        ea = _attribution_result().example_attributions[0]
        assert ea.demo_attributions == pytest.approx([_ATTRIBUTION_DEMO0, _ATTRIBUTION_DEMO1])


class TestAblationMechanics:

    def test_instruction_masked_with_placeholder(self) -> None:
        auditor, predictor, _ = _invoke_default()
        auditor.invoke()
        first_mask_call = predictor.signature.with_instructions.call_args_list[0]
        assert first_mask_call == call(_MASK_PLACEHOLDER)

    def test_instruction_restored_after_ablation(self) -> None:
        auditor, predictor, _ = _invoke_default()
        auditor.invoke()
        restore_call = predictor.signature.with_instructions.call_args_list[1]
        assert restore_call == call(_INSTRUCTION_TEXT)

    def test_demos_restored_after_invoke(self) -> None:
        predictor = _make_predictor()
        original_demos = list(predictor.demos)
        llm = _make_llm(predictor)
        scorer = _make_scorer_with_sequence(
            _SCORE_FULL, _SCORE_INSTR_ABLATED, _SCORE_DEMO0_ABLATED, _SCORE_DEMO1_ABLATED,
        )
        SimplePromptAttributionAuditor(_make_dataset(1), llm, scorer).invoke()
        assert predictor.demos == original_demos

    def test_demos_restored_when_baseline_fails(self) -> None:
        predictor = _make_predictor()
        original_demos = list(predictor.demos)
        llm = _make_llm(predictor)
        # Baseline fails — demos are restored by the belt-and-braces assignment
        scorer = _make_scorer(score=INVALID_SCORE)
        SimplePromptAttributionAuditor(_make_dataset(1), llm, scorer).invoke()
        assert predictor.demos == original_demos
