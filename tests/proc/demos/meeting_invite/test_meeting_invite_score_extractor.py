import json
from types import MappingProxyType
from unittest.mock import MagicMock

import pytest

from proc.demos.meeting_invite.meeting_invite_score_extractor import MeetingInviteScoreExtractor
from proc.demos.meeting_invite.models import EmailMeetingInfo, Flexibility, PreferredWindow, Urgency


class _F:
    tz: str = "America/New_York"
    wk: str = "preferred_windows"
    tk: str = "time_of_day"
    tod: str = "morning"
    dur: int = 60
    bad_dur: int = 999
    sc_miss_major: float = 0.8
    sc_miss_minor: float = 0.85
    sc_partial_tod: float = 0.925
    extractor: MeetingInviteScoreExtractor = MeetingInviteScoreExtractor()
    base: MappingProxyType[str, object] = MappingProxyType({
        "sender_iana_timezone": "America/New_York",
        "duration_minutes": 60,
        "urgency": Urgency.TODAY,
        "flexibility": Flexibility.SPECIFIC,
        "preferred_windows": [{"time_of_day": "morning"}],
    })

    @staticmethod
    def gold(windows: list[PreferredWindow] | None = None) -> EmailMeetingInfo:
        wlist = [PreferredWindow(time_of_day=_F.tod)] if windows is None else windows
        return EmailMeetingInfo(
            sender_iana_timezone=_F.tz,
            duration_minutes=_F.dur,
            urgency=Urgency.TODAY,
            flexibility=Flexibility.SPECIFIC,
            preferred_windows=wlist,
        )

    @staticmethod
    def score(raw_json: str, gold: EmailMeetingInfo | None = None) -> float:
        actual_gold = _F.gold() if gold is None else gold
        ex = MagicMock()
        ex.expected = actual_gold
        pred = MagicMock()
        pred.extracted_json = raw_json
        return _F.extractor.extraction_metric(ex, pred)


class TestExtractionMetricPerfect:

    def test_perfect_match_score_is_one(self) -> None:
        raw = json.dumps(dict(_F.base))
        assert _F.score(raw) == pytest.approx(1.0)

    def test_invalid_json_returns_zero(self) -> None:
        assert _F.score("not json") == 0

    def test_empty_prediction_returns_zero(self) -> None:
        assert _F.score("") == 0


class TestExtractionMetricPenalties:

    def test_duration_mismatch_loses_weight(self) -> None:
        raw = json.dumps({**_F.base, "duration_minutes": _F.bad_dur})
        assert _F.score(raw) == pytest.approx(_F.sc_miss_major)

    def test_urgency_mismatch_loses_weight(self) -> None:
        raw = json.dumps({**_F.base, "urgency": "flexible"})
        assert _F.score(raw) == pytest.approx(_F.sc_miss_major)

    def test_flexibility_mismatch_loses_weight(self) -> None:
        raw = json.dumps({**_F.base, "flexibility": "flexible"})
        assert _F.score(raw) == pytest.approx(_F.sc_miss_minor)

    def test_timezone_mismatch_loses_weight(self) -> None:
        raw = json.dumps({**_F.base, "sender_iana_timezone": "Europe/London"})
        assert _F.score(raw) == pytest.approx(_F.sc_miss_minor)


class TestExtractionMetricWindows:

    def test_count_mismatch_loses_windows_weight(self) -> None:
        afternoon = {_F.tk: "afternoon"}
        evening = {_F.tk: "evening"}
        three = [{_F.tk: _F.tod}, afternoon, evening]
        raw = json.dumps({**_F.base, _F.wk: three})
        assert _F.score(raw) == pytest.approx(_F.sc_miss_minor)

    def test_empty_gold_windows_gives_full_score(self) -> None:
        gold = _F.gold([])
        raw = json.dumps({**_F.base, _F.wk: []})
        assert _F.score(raw, gold) == pytest.approx(1.0)

    def test_partial_tod_overlap_reduces_score(self) -> None:
        win1 = PreferredWindow(time_of_day=_F.tod)
        win2 = PreferredWindow(time_of_day="afternoon")
        gold = _F.gold([win1, win2])
        one_win = [{_F.tk: _F.tod}]
        raw = json.dumps({**_F.base, _F.wk: one_win})
        assert _F.score(raw, gold) == pytest.approx(_F.sc_partial_tod)
