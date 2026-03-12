import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from proc.base.base_llm import BaseLLMConfig
from proc.demos.meeting_invite.meeting_invite_dataset import MeetingInviteDataset
from proc.demos.meeting_invite.meeting_invite_extractor_llm import MeetingInviteLLM, MeetingInvitePayload
from proc.demos.meeting_invite.meeting_invite_score_extractor import MeetingInviteScoreExtractor

_DATASET_PATH: Path = Path(__file__).parent / "dataset" / "emails_20.jsonl"
_N_EXAMPLES: int = 20
_MIN_AVG_SCORE: float = 0.5

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def examples() -> list[Any]:
    return MeetingInviteDataset(_DATASET_PATH).load()


@pytest.fixture(scope="module")
def scores(examples: list[Any]) -> list[float]:
    llm = MeetingInviteLLM(config=BaseLLMConfig())
    extractor = MeetingInviteScoreExtractor()
    result: list[float] = []
    for ex in examples:
        payload = MeetingInvitePayload(
            email_from=ex.email_from,
            email_to=ex.email_to,
            email_body=ex.email_body,
            current_date=ex.current_date,
        )
        outcome = llm.invoke(payload)
        raw = json.dumps(outcome.unwrap().model_dump(mode="json"))
        pred = MagicMock(extracted_json=raw)
        result.append(extractor.extraction_metric(ex, pred))
    return result


class TestMeetingInviteLLMScores:

    def test_all_examples_scored(self, scores: list[float]) -> None:
        assert len(scores) == _N_EXAMPLES

    def test_scores_in_valid_range(self, scores: list[float]) -> None:
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_average_meets_threshold(self, scores: list[float]) -> None:
        avg = sum(scores) / len(scores)
        assert avg >= _MIN_AVG_SCORE
