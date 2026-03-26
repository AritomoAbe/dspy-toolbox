import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from proc.demos.meeting_invite.meeting_invite_dataset import MeetingInviteDataset
from proc.demos.meeting_invite.meeting_invite_extractor_llm import MeetingInviteLLM, _parse_llm_output
from proc.demos.meeting_invite.meeting_invite_score_extractor import MeetingInviteScoreExtractor
from proc.demos.meeting_invite.models import EmailMeetingInfo

_DATASET_PATH: Path = Path(__file__).parent / "dataset" / "emails_20.jsonl"
_N_EXAMPLES: int = 20
_DUMP_MODE: str = "json"


@pytest.fixture(scope="module")
def examples() -> list[Any]:
    return MeetingInviteDataset(_DATASET_PATH).load()


class TestMeetingInviteDatasetLoad:

    def test_loads_correct_count(self, examples: list[Any]) -> None:
        assert len(examples) == _N_EXAMPLES

    def test_examples_have_email_body(self, examples: list[Any]) -> None:
        assert all(hasattr(ex, "email_body") for ex in examples)

    def test_examples_have_expected(self, examples: list[Any]) -> None:
        assert all(hasattr(ex, "expected") for ex in examples)

    def test_expected_is_email_meeting_info(self, examples: list[Any]) -> None:
        assert all(isinstance(ex.expected, EmailMeetingInfo) for ex in examples)

    def test_input_fields_accessible(self, examples: list[Any]) -> None:
        keys = set(examples[0].inputs().keys())
        assert "email_body" in keys
        assert "current_date" in keys


class TestMeetingInviteParseOnRealData:

    def test_parse_all_expected_outputs(self, examples: list[Any]) -> None:
        for ex in examples:
            raw = json.dumps(ex.expected.model_dump(mode=_DUMP_MODE))
            assert isinstance(_parse_llm_output(raw), EmailMeetingInfo)

    def test_parsed_urgency_matches_gold(self, examples: list[Any]) -> None:
        for ex in examples:
            raw = json.dumps(ex.expected.model_dump(mode=_DUMP_MODE))
            assert _parse_llm_output(raw).urgency == ex.expected.urgency

    def test_parsed_duration_matches_gold(self, examples: list[Any]) -> None:
        for ex in examples:
            raw = json.dumps(ex.expected.model_dump(mode=_DUMP_MODE))
            assert _parse_llm_output(raw).duration_minutes == ex.expected.duration_minutes


class TestScoreExtractorOnRealData:

    def test_perfect_prediction_scores_one(self, examples: list[Any]) -> None:
        extractor = MeetingInviteScoreExtractor()
        for ex in examples:
            raw = json.dumps(ex.expected.model_dump(mode=_DUMP_MODE))
            pred = MagicMock(extracted_json=raw)
            assert extractor.extraction_metric(ex, pred) == pytest.approx(1.0)
