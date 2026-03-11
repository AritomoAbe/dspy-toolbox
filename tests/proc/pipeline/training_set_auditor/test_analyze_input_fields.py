from pathlib import Path

import pytest
from returns.pipeline import is_successful

from proc.pipeline.training_set_auditor.analyze_input_fields import AnalyzeInputFields
from proc.pipeline.training_set_auditor.enums import FieldType
from proc.pipeline.training_set_auditor.models import CategoricalStats, FreeTextStats

DATASET = Path(__file__).parent / "dataset" / "emails_20.jsonl"


@pytest.fixture(scope="module")
def analysis() -> dict:
    return AnalyzeInputFields(DATASET).invoke().unwrap()


class TestInvoke:

    def test_success_on_valid_file(self) -> None:
        result = AnalyzeInputFields(DATASET).invoke()
        assert is_successful(result)

    def test_failure_on_missing_file(self) -> None:
        result = AnalyzeInputFields(Path("/nonexistent/path.jsonl")).invoke()
        assert not is_successful(result)

    def test_failure_message_mentions_path(self) -> None:
        path = Path("/nonexistent/path.jsonl")
        result = AnalyzeInputFields(path).invoke()
        assert str(path) in result.failure().message

    def test_all_input_fields_present(self, analysis: dict) -> None:
        assert set(analysis.keys()) == {"email_from", "email_to", "email_body", "current_date"}

    def test_expected_key_excluded(self, analysis: dict) -> None:
        assert "expected" not in analysis.keys()

    def test_all_values_are_field_analysis(self, analysis: dict) -> None:
        from proc.pipeline.training_set_auditor.models import FieldAnalysis
        assert all(isinstance(v, FieldAnalysis) for v in analysis.values())


class TestFieldTypes:

    def test_email_to_is_categorical(self, analysis: dict) -> None:
        assert analysis["email_to"].type == FieldType.categorical

    def test_email_from_is_free_text(self, analysis: dict) -> None:
        assert analysis["email_from"].type == FieldType.free_text

    def test_email_body_is_free_text(self, analysis: dict) -> None:
        assert analysis["email_body"].type == FieldType.free_text

    def test_current_date_is_free_text(self, analysis: dict) -> None:
        assert analysis["current_date"].type == FieldType.free_text


class TestFieldStats:

    def test_all_fields_have_20_records(self, analysis: dict) -> None:
        for field_analysis in analysis.values():
            assert field_analysis.stats.count == 20

    def test_email_to_is_categorical_stats(self, analysis: dict) -> None:
        assert isinstance(analysis["email_to"].stats, CategoricalStats)

    def test_email_to_has_eight_unique_recipients(self, analysis: dict) -> None:
        assert analysis["email_to"].stats.unique_count == 8

    def test_email_body_is_free_text_stats(self, analysis: dict) -> None:
        assert isinstance(analysis["email_body"].stats, FreeTextStats)

    def test_email_body_all_unique(self, analysis: dict) -> None:
        assert analysis["email_body"].stats.unique_count == 20

    def test_email_body_has_samples(self, analysis: dict) -> None:
        assert len(analysis["email_body"].stats.samples) > 0

    def test_current_date_all_unique(self, analysis: dict) -> None:
        assert analysis["current_date"].stats.unique_count == 20