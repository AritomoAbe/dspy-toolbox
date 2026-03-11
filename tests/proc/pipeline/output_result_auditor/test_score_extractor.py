from typing import Any

import pytest

from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor


class _ConcreteExtractor(ScoreExtractor):
    def extraction_metric(self, example: Any, prediction: Any, trace: Any = None) -> float:
        return 1.0


class TestScoreExtractor:

    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            ScoreExtractor()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self) -> None:
        assert _ConcreteExtractor() is not None

    def test_extraction_metric_returns_float(self) -> None:
        result = _ConcreteExtractor().extraction_metric("example", "prediction")
        assert isinstance(result, float)

    def test_extraction_metric_returns_positive(self) -> None:
        result = _ConcreteExtractor().extraction_metric("example", "prediction")
        assert result > 0

    def test_extraction_metric_accepts_trace_kwarg(self) -> None:
        result = _ConcreteExtractor().extraction_metric("example", "prediction", trace=None)
        assert isinstance(result, float)

    def test_incomplete_subclass_raises(self) -> None:
        class _IncompleteExtractor(ScoreExtractor):
            pass

        with pytest.raises(TypeError):
            _IncompleteExtractor()  # type: ignore[abstract]
