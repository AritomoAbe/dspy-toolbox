import json
from pathlib import Path

import pytest
from returns.pipeline import is_successful

from proc.pipeline.dataset.training_dataset import TrainingSetDataset
from proc.pipeline.training_set_auditor.analyze_co_occurrence import AnalyzeCoOccurrence
from proc.pipeline.training_set_auditor.enums import CorrelationRisk
from proc.pipeline.training_set_auditor.models import CoOccurrenceEntry

_DATASET_PATH: Path = Path(__file__).parent / "dataset" / "emails_20.jsonl"


def _write_jsonl(path: Path, records: list) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records))


@pytest.fixture(scope="module")
def dataset() -> TrainingSetDataset:
    return TrainingSetDataset(_DATASET_PATH)


@pytest.fixture(scope="module")
def analysis(dataset: TrainingSetDataset) -> list:
    return AnalyzeCoOccurrence(dataset).invoke().unwrap().context.result


class TestInvoke:

    def test_success_on_valid_dataset(self, dataset: TrainingSetDataset) -> None:
        result = AnalyzeCoOccurrence(dataset).invoke()
        assert is_successful(result)

    def test_returns_list(self, analysis: list) -> None:
        assert isinstance(analysis, list)

    def test_all_entries_are_co_occurrence_entries(self, analysis: list) -> None:
        assert all(isinstance(entry, CoOccurrenceEntry) for entry in analysis)

    def test_no_correlations_in_diverse_dataset(self, analysis: list) -> None:
        assert analysis == []

    def test_score_is_one(self, dataset: TrainingSetDataset) -> None:
        score = AnalyzeCoOccurrence(dataset).invoke().unwrap()
        assert score.value == pytest.approx(1.0)


class TestCoOccurrenceEntries:

    def test_entries_have_field_a_and_b(self, analysis: list) -> None:
        for entry in analysis:
            assert entry.field_a
            assert entry.field_b

    def test_field_a_and_b_are_different(self, analysis: list) -> None:
        for entry in analysis:
            assert entry.field_a != entry.field_b

    def test_dominant_pct_between_zero_and_hundred(self, analysis: list) -> None:
        for entry in analysis:
            assert 0.0 < entry.dominant_pct <= 100.0

    def test_dominant_pct_above_threshold(self, analysis: list) -> None:
        for entry in analysis:
            assert entry.dominant_pct > 70.0

    def test_dominant_pair_is_two_strings(self, analysis: list) -> None:
        for entry in analysis:
            assert isinstance(entry.dominant_pair, tuple)
            assert len(entry.dominant_pair) == 2
            assert all(isinstance(v, str) for v in entry.dominant_pair)

    def test_risk_is_correlation_risk(self, analysis: list) -> None:
        for entry in analysis:
            assert isinstance(entry.risk, CorrelationRisk)

    def test_high_risk_entries_above_85_pct(self, analysis: list) -> None:
        for entry in analysis:
            if entry.risk == CorrelationRisk.high:
                assert entry.dominant_pct > 85.0

    def test_moderate_risk_entries_between_70_and_85_pct(self, analysis: list) -> None:
        for entry in analysis:
            if entry.risk == CorrelationRisk.moderate:
                assert 70.0 < entry.dominant_pct <= 85.0

    def test_entries_sorted_descending_by_dominant_pct(self, analysis: list) -> None:
        pcts = [e.dominant_pct for e in analysis]
        assert pcts == sorted(pcts, reverse=True)


class TestWithSyntheticData:

    def test_detects_moderate_risk_correlation(self, tmp_path: Path) -> None:
        records = [{"expected": {"status": "A", "tier": "X"}}] * 8
        records += [{"expected": {"status": "B", "tier": "Y"}}] * 2
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, records)
        result = AnalyzeCoOccurrence(TrainingSetDataset(dataset)).invoke().unwrap().context.result
        assert len(result) == 1
        assert result[0].risk == CorrelationRisk.moderate
        assert result[0].dominant_pair == ("A", "X")

    def test_detects_high_risk_correlation(self, tmp_path: Path) -> None:
        records = [{"expected": {"status": "A", "tier": "X"}}] * 9
        records += [{"expected": {"status": "B", "tier": "Y"}}] * 1
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, records)
        result = AnalyzeCoOccurrence(TrainingSetDataset(dataset)).invoke().unwrap().context.result
        assert len(result) == 1
        assert result[0].risk == CorrelationRisk.high
        assert result[0].dominant_pct > 85.0

    def test_ignores_non_dict_expected(self, tmp_path: Path) -> None:
        records = [{"expected": "not_a_dict"}, {"expected": {"status": "A", "tier": "X"}}]
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, records)
        result = AnalyzeCoOccurrence(TrainingSetDataset(dataset)).invoke()
        assert is_successful(result)

    def test_sorted_by_dominant_pct_descending(self, tmp_path: Path) -> None:
        records = (
            [{"expected": {"a": "X", "b": "P"}}] * 9
            + [{"expected": {"a": "Y", "b": "Q"}}] * 1
            + [{"expected": {"a": "X", "c": "M"}}] * 8
            + [{"expected": {"a": "Y", "c": "N"}}] * 2
        )
        dataset = tmp_path / "data.jsonl"
        _write_jsonl(dataset, records)
        result = AnalyzeCoOccurrence(TrainingSetDataset(dataset)).invoke().unwrap().context.result
        pcts = [e.dominant_pct for e in result]
        assert pcts == sorted(pcts, reverse=True)
