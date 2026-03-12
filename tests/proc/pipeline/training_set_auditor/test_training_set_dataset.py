import json
from pathlib import Path
from typing import Any

import pytest

from proc.pipeline.dataset.training_dataset import TrainingSetDataset

_DATASET_PATH: Path = Path(__file__).parent / "dataset" / "emails_20.jsonl"
_N_EMAILS: int = 20


@pytest.fixture(scope="module")
def examples() -> list[Any]:
    return TrainingSetDataset(_DATASET_PATH).load()


class TestLoad:

    def test_loads_correct_count(self, examples: list[Any]) -> None:
        assert len(examples) == _N_EMAILS

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            TrainingSetDataset(Path("/nonexistent/path.jsonl")).load()

    def test_raises_on_invalid_jsonl(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("not json\n")
        with pytest.raises(Exception):
            TrainingSetDataset(bad).load()

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('\n{"email_body": "hello"}\n\n{"email_body": "world"}\n')
        result = TrainingSetDataset(path).load()
        assert len(result) == 2

    def test_load_is_cached(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps({"email_body": "hello"}) + "\n")
        dataset = TrainingSetDataset(path)
        first = dataset.load()
        second = dataset.load()
        assert first is second


class TestExamples:

    def test_examples_have_email_body(self, examples: list[Any]) -> None:
        assert all(hasattr(ex, "email_body") for ex in examples)

    def test_examples_have_expected(self, examples: list[Any]) -> None:
        assert all(hasattr(ex, "expected") for ex in examples)

    def test_expected_is_dict(self, examples: list[Any]) -> None:
        assert all(isinstance(ex.expected, dict) for ex in examples)

    def test_input_fields_accessible(self, examples: list[Any]) -> None:
        keys = set(examples[0].inputs().keys())
        assert "email_body" in keys
        assert "current_date" in keys

    def test_expected_not_an_input_field(self, examples: list[Any]) -> None:
        keys = set(examples[0].inputs().keys())
        assert "expected" not in keys

    def test_items_includes_all_fields(self, examples: list[Any]) -> None:
        record = dict(examples[0].items())
        assert "email_body" in record
        assert "expected" in record

    def test_synthetic_record_without_expected(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps({"text": "hello"}) + "\n")
        result = TrainingSetDataset(path).load()
        assert result[0].expected == {}
