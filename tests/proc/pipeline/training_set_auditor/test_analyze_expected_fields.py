from pathlib import Path

import pytest
from returns.pipeline import is_successful

from proc.pipeline.training_set_auditor.enums import FieldType
from proc.pipeline.training_set_auditor.models import CategoricalStats, NumericStats
from proc.pipeline.training_set_auditor.analyze_expected_fields import AnalyzeExpectedFields

DATASET = Path(__file__).parent / "dataset" / "emails_20.jsonl"


@pytest.fixture(scope="module")
def analysis() -> dict:
    return AnalyzeExpectedFields(DATASET).invoke().unwrap()


class TestInvoke:

    def test_success_on_valid_file(self) -> None:
        result = AnalyzeExpectedFields(DATASET).invoke()
        assert is_successful(result)

    def test_failure_on_missing_file(self) -> None:
        result = AnalyzeExpectedFields(Path("/nonexistent/path.jsonl")).invoke()
        assert not is_successful(result)

    def test_failure_message_mentions_path(self) -> None:
        path = Path("/nonexistent/path.jsonl")
        result = AnalyzeExpectedFields(path).invoke()
        assert str(path) in result.failure().message

    def test_all_expected_fields_present(self, analysis: dict) -> None:
        assert set(analysis.keys()) == {
            "sender_iana_timezone",
            "duration_minutes",
            "urgency",
            "flexibility",
            "preferred_windows",
            "meeting_topic",
        }

    def test_all_values_are_field_analysis(self, analysis: dict) -> None:
        from proc.pipeline.training_set_auditor.models import FieldAnalysis
        assert all(isinstance(v, FieldAnalysis) for v in analysis.values())


class TestFieldTypes:

    def test_duration_minutes_is_numeric(self, analysis: dict) -> None:
        assert analysis["duration_minutes"].type == FieldType.numeric

    def test_urgency_is_categorical(self, analysis: dict) -> None:
        assert analysis["urgency"].type == FieldType.categorical

    def test_flexibility_is_categorical(self, analysis: dict) -> None:
        assert analysis["flexibility"].type == FieldType.categorical

    def test_sender_iana_timezone_is_categorical(self, analysis: dict) -> None:
        assert analysis["sender_iana_timezone"].type == FieldType.categorical

    def test_preferred_windows_is_list(self, analysis: dict) -> None:
        assert analysis["preferred_windows"].type == FieldType.list

    def test_meeting_topic_is_free_text(self, analysis: dict) -> None:
        assert analysis["meeting_topic"].type == FieldType.free_text


class TestFieldStats:

    def test_duration_minutes_count(self, analysis: dict) -> None:
        assert analysis["duration_minutes"].stats.count == 20

    def test_duration_minutes_is_numeric_stats(self, analysis: dict) -> None:
        assert isinstance(analysis["duration_minutes"].stats, NumericStats)

    def test_flexibility_has_two_unique_values(self, analysis: dict) -> None:
        assert isinstance(analysis["flexibility"].stats, CategoricalStats)
        assert analysis["flexibility"].stats.unique_count == 2

    def test_flexibility_distribution_keys(self, analysis: dict) -> None:
        assert set(analysis["flexibility"].stats.distribution.keys()) == {"FLEXIBLE", "SPECIFIC"}

    def test_urgency_has_four_unique_values(self, analysis: dict) -> None:
        assert analysis["urgency"].stats.unique_count == 4

    def test_preferred_windows_count(self, analysis: dict) -> None:
        assert analysis["preferred_windows"].stats.count == 20

    def test_preferred_windows_has_non_empty_samples(self, analysis: dict) -> None:
        assert len(analysis["preferred_windows"].stats.non_empty_samples) > 0

    def test_meeting_topic_count(self, analysis: dict) -> None:
        assert analysis["meeting_topic"].stats.count == 20