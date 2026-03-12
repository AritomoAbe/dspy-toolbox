import json
import logging
from typing import Any

import dspy

from proc.demos.meeting_invite.calendar_utils import _normalise_time_of_day
from proc.demos.meeting_invite.meeting_invite_extractor_llm import _strip_fences, _dict_to_email_meeting_info
from proc.demos.meeting_invite.models import EmailMeetingInfo
from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor


class _ScoreWeights:
    duration: float = 0.2
    urgency: float = 0.2
    flexibility: float = 0.15
    timezone: float = 0.15
    windows: float = 0.15
    time_of_day: float = 0.15
    duration_tolerance: int = 10
    windows_tolerance: int = 1


class MeetingInviteScoreExtractor(ScoreExtractor):

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def extraction_metric(self, example: dspy.Example, prediction: dspy.Prediction, trace: Any = None) -> float:
        expected = example.expected
        if isinstance(expected, EmailMeetingInfo):
            gold: EmailMeetingInfo = expected
        else:
            try:
                gold = _dict_to_email_meeting_info(expected)
            except Exception:
                return 0
        raw_json = getattr(prediction, "extracted_json", "") or ""
        try:
            pred = _dict_to_email_meeting_info(json.loads(_strip_fences(raw_json)))
        except Exception:
            return 0

        score: float = 0

        if abs(pred.duration_minutes - gold.duration_minutes) <= _ScoreWeights.duration_tolerance:
            score += _ScoreWeights.duration

        if pred.urgency == gold.urgency:
            score += _ScoreWeights.urgency

        if pred.flexibility == gold.flexibility:
            score += _ScoreWeights.flexibility

        if pred.sender_iana_timezone == gold.sender_iana_timezone:
            score += _ScoreWeights.timezone

        windows_diff = abs(len(pred.preferred_windows) - len(gold.preferred_windows))
        if windows_diff <= _ScoreWeights.windows_tolerance:
            score += _ScoreWeights.windows

        gold_times = {
            _normalise_time_of_day(w.time_of_day)
            for w in gold.preferred_windows
            if w.time_of_day
        }
        pred_times = {
            _normalise_time_of_day(w.time_of_day)
            for w in pred.preferred_windows
            if w.time_of_day
        }
        if gold_times:
            overlap = len(gold_times & pred_times) / len(gold_times)
            score += _ScoreWeights.time_of_day * overlap
        else:
            score += _ScoreWeights.time_of_day

        self._logger.debug(
            "metric: duration=%.0f urgency=%s flex=%s tz=%s windows=%d -> score=%.2f",
            pred.duration_minutes, pred.urgency, pred.flexibility,
            pred.sender_iana_timezone, len(pred.preferred_windows), score,
        )
        return score
