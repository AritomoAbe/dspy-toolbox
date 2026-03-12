from pathlib import Path

import pytest
from returns.pipeline import is_successful

from proc.pipeline.dataset.training_dataset import TrainingSetDataset
from proc.pipeline.training_set_auditor.analyze_near_duplicates import AnalyzeNearDuplicates
from proc.pipeline.training_set_auditor.models import FieldDuplicateStats, NearDuplicatePair

_DATASET_PATH: Path = Path(__file__).parent / "dataset" / "emails_20.jsonl"
_TEXT_FIELDS = ["email_body", "email_to"]


@pytest.fixture(scope="module")
def dataset() -> TrainingSetDataset:
    return TrainingSetDataset(_DATASET_PATH)


@pytest.fixture(scope="module")
def analysis(dataset: TrainingSetDataset) -> dict:
    return AnalyzeNearDuplicates(dataset, _TEXT_FIELDS).invoke().unwrap().context.result


class TestInvoke:

    def test_success_on_valid_dataset(self, dataset: TrainingSetDataset) -> None:
        result = AnalyzeNearDuplicates(dataset, _TEXT_FIELDS).invoke()
        assert is_successful(result)

    def test_returns_only_requested_fields(self, analysis: dict) -> None:
        assert set(analysis.keys()) == set(_TEXT_FIELDS)

    def test_unknown_field_excluded(self, dataset: TrainingSetDataset) -> None:
        result = AnalyzeNearDuplicates(dataset, ["nonexistent_field"]).invoke()
        assert is_successful(result)
        assert result.unwrap().context.result == {}

    def test_score_is_one(self, dataset: TrainingSetDataset) -> None:
        score = AnalyzeNearDuplicates(dataset, _TEXT_FIELDS).invoke().unwrap()
        assert score.value == pytest.approx(1.0)


class TestFieldStats:

    def test_all_values_are_field_duplicate_stats(self, analysis: dict) -> None:
        assert all(isinstance(v, FieldDuplicateStats) for v in analysis.values())

    def test_email_body_total_is_20(self, analysis: dict) -> None:
        assert analysis["email_body"].total == 20

    def test_email_to_total_is_20(self, analysis: dict) -> None:
        assert analysis["email_to"].total == 20

    def test_email_body_pairs_non_negative(self, analysis: dict) -> None:
        assert analysis["email_body"].near_duplicate_pairs >= 0

    def test_email_body_rate_pct_non_negative(self, analysis: dict) -> None:
        assert analysis["email_body"].near_duplicate_rate_pct >= 0.0

    def test_email_body_examples_at_most_five(self, analysis: dict) -> None:
        assert len(analysis["email_body"].examples) <= 5

    def test_email_to_has_near_duplicates(self, analysis: dict) -> None:
        assert analysis["email_to"].near_duplicate_pairs > 0

    def test_email_to_rate_pct_positive(self, analysis: dict) -> None:
        assert analysis["email_to"].near_duplicate_rate_pct > 0.0


class TestNearDuplicatePairs:

    def test_examples_are_near_duplicate_pair_instances(self, analysis: dict) -> None:
        for field_stats in analysis.values():
            for example in field_stats.examples:
                assert isinstance(example, NearDuplicatePair)

    def test_example_scores_above_threshold(self, analysis: dict) -> None:
        for field_stats in analysis.values():
            for example in field_stats.examples:
                assert example.score > 0.85

    def test_example_scores_at_most_one(self, analysis: dict) -> None:
        for field_stats in analysis.values():
            for example in field_stats.examples:
                assert example.score <= 1.0

    def test_examples_sorted_descending_by_score(self, analysis: dict) -> None:
        for field_stats in analysis.values():
            scores = [e.score for e in field_stats.examples]
            assert scores == sorted(scores, reverse=True)

    def test_pair_indices_are_ordered(self, analysis: dict) -> None:
        for field_stats in analysis.values():
            for example in field_stats.examples:
                assert example.index_a < example.index_b
