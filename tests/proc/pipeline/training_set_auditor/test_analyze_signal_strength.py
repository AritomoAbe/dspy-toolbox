import json
from pathlib import Path

import pytest
from returns.pipeline import is_successful

from proc.pipeline.training_set_auditor.analyze_signal_strength import AnalyzeSignalStrength
from proc.pipeline.training_set_auditor.enums import CorrelationRisk, LearnabilityLevel, SeparabilityLevel
from proc.pipeline.training_set_auditor.models import (
    ClassifierError,
    ClassifierStats,
    ClassPairSeparability,
    FieldSignalStats,
    KeywordScore,
    SeparabilityStats,
    SkippedField,
)

DATASET = Path(__file__).parent / "dataset" / "emails_20.jsonl"
TEXT_FIELDS = ["email_body"]


@pytest.fixture(scope="module")
def analysis() -> dict:
    return AnalyzeSignalStrength(DATASET, TEXT_FIELDS).invoke().unwrap()


def _write_jsonl(path: Path, records: list) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records))


def _make_clear_signal_records(n_per_class: int = 10) -> list:
    records = []
    for _ in range(n_per_class):
        records.append({"text": "urgent deadline critical asap today", "expected": {"label": "HIGH"}})
    for _ in range(n_per_class):
        records.append({"text": "flexible whenever relax no rush open", "expected": {"label": "LOW"}})
    return records


class TestInvoke:

    def test_success_on_valid_file(self) -> None:
        result = AnalyzeSignalStrength(DATASET, TEXT_FIELDS).invoke()
        assert is_successful(result)

    def test_failure_on_missing_file(self) -> None:
        result = AnalyzeSignalStrength(Path("/nonexistent/path.jsonl"), TEXT_FIELDS).invoke()
        assert not is_successful(result)

    def test_failure_message_mentions_path(self) -> None:
        path = Path("/nonexistent/path.jsonl")
        result = AnalyzeSignalStrength(path, TEXT_FIELDS).invoke()
        assert str(path) in result.failure().message

    def test_returns_dict(self, analysis: dict) -> None:
        assert isinstance(analysis, dict)

    def test_expected_fields_in_result(self, analysis: dict) -> None:
        assert "urgency" in analysis
        assert "flexibility" in analysis
        assert "preferred_windows" in analysis


class TestEmailDataset:

    def test_meeting_topic_is_skipped(self, analysis: dict) -> None:
        assert isinstance(analysis["meeting_topic"], SkippedField)

    def test_skipped_field_has_reason(self, analysis: dict) -> None:
        assert analysis["meeting_topic"].skipped

    def test_analyzed_fields_are_field_signal_stats(self, analysis: dict) -> None:
        for field, value in analysis.items():
            if not isinstance(value, SkippedField):
                assert isinstance(value, FieldSignalStats)

    def test_analyzed_fields_have_proxy_classifier(self, analysis: dict) -> None:
        for value in analysis.values():
            if isinstance(value, FieldSignalStats):
                assert isinstance(value.proxy_classifier, (ClassifierError, ClassifierStats))

    def test_analyzed_fields_have_separability(self, analysis: dict) -> None:
        for value in analysis.values():
            if isinstance(value, FieldSignalStats):
                assert isinstance(value.separability, SeparabilityStats)

    def test_analyzed_fields_have_keywords(self, analysis: dict) -> None:
        for value in analysis.values():
            if isinstance(value, FieldSignalStats):
                assert isinstance(value.discriminating_keywords, dict)

    def test_flexibility_has_two_class_pairwise(self, analysis: dict) -> None:
        stats = analysis["flexibility"]
        assert isinstance(stats, FieldSignalStats)
        assert len(stats.separability.pairwise) == 1

    def test_separability_pairwise_are_class_pair_separability(self, analysis: dict) -> None:
        stats = analysis["flexibility"]
        assert isinstance(stats, FieldSignalStats)
        for entry in stats.separability.pairwise:
            assert isinstance(entry, ClassPairSeparability)

    def test_separability_risk_is_separability_level(self, analysis: dict) -> None:
        stats = analysis["flexibility"]
        assert isinstance(stats, FieldSignalStats)
        for entry in stats.separability.pairwise:
            assert isinstance(entry.separability, SeparabilityLevel)

    def test_keyword_scores_are_keyword_score_instances(self, analysis: dict) -> None:
        for value in analysis.values():
            if isinstance(value, FieldSignalStats):
                for keywords in value.discriminating_keywords.values():
                    for kw in keywords:
                        assert isinstance(kw, KeywordScore)

    def test_keyword_scores_are_positive(self, analysis: dict) -> None:
        for value in analysis.values():
            if isinstance(value, FieldSignalStats):
                for keywords in value.discriminating_keywords.values():
                    for kw in keywords:
                        assert kw.score > 0


class TestWithSyntheticData:

    def test_clear_signal_yields_classifier_stats(self, tmp_path: Path) -> None:
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, _make_clear_signal_records())
        result = AnalyzeSignalStrength(dataset, ["text"]).invoke().unwrap()
        assert isinstance(result["label"].proxy_classifier, ClassifierStats)

    def test_classifier_stats_learnability_is_learnability_level(self, tmp_path: Path) -> None:
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, _make_clear_signal_records())
        result = AnalyzeSignalStrength(dataset, ["text"]).invoke().unwrap()
        stats = result["label"].proxy_classifier
        assert isinstance(stats, ClassifierStats)
        assert isinstance(stats.learnability, LearnabilityLevel)

    def test_clear_signal_accuracy_above_chance(self, tmp_path: Path) -> None:
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, _make_clear_signal_records())
        result = AnalyzeSignalStrength(dataset, ["text"]).invoke().unwrap()
        stats = result["label"].proxy_classifier
        assert isinstance(stats, ClassifierStats)
        assert stats.accuracy > stats.chance_baseline

    def test_clear_signal_separability_has_one_pair(self, tmp_path: Path) -> None:
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, _make_clear_signal_records())
        result = AnalyzeSignalStrength(dataset, ["text"]).invoke().unwrap()
        assert len(result["label"].separability.pairwise) == 1

    def test_clear_signal_keywords_not_empty(self, tmp_path: Path) -> None:
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, _make_clear_signal_records())
        result = AnalyzeSignalStrength(dataset, ["text"]).invoke().unwrap()
        keywords = result["label"].discriminating_keywords
        assert any(len(kws) > 0 for kws in keywords.values())

    def test_single_class_field_is_skipped(self, tmp_path: Path) -> None:
        records = [{"text": "hello", "expected": {"label": "ONLY"}} for _ in range(10)]
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, records)
        result = AnalyzeSignalStrength(dataset, ["text"]).invoke().unwrap()
        assert isinstance(result["label"], SkippedField)

    def test_non_dict_expected_is_ignored(self, tmp_path: Path) -> None:
        records = [{"text": "a", "expected": "not_a_dict"}] * 5
        records += _make_clear_signal_records(5)
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, records)
        result = AnalyzeSignalStrength(dataset, ["text"]).invoke()
        assert is_successful(result)