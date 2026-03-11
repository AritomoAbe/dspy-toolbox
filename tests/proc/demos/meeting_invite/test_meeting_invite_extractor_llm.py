import json

import pytest
from pydantic import ValidationError

from proc.demos.meeting_invite.meeting_invite_extractor_llm import (
    MeetingInviteLLM,
    MeetingInvitePayload,
    _dict_to_email_meeting_info,
    _strip_fences,
)
from proc.demos.meeting_invite.models import EmailMeetingInfo, _DEFAULT_DURATION, _UNKNOWN_TZ

_PLAIN: str = "hello"
_WINDOWS_KEY: str = "preferred_windows"


class TestStripFences:

    def test_plain_text_unchanged(self) -> None:
        assert _strip_fences(_PLAIN) == _PLAIN

    def test_backtick_fences_removed(self) -> None:
        assert _strip_fences(f"```\n{_PLAIN}\n```") == _PLAIN

    def test_language_tag_stripped(self) -> None:
        assert _strip_fences("```json\n{}\n```") == "{}"

    def test_whitespace_stripped(self) -> None:
        assert _strip_fences(f"  {_PLAIN}  ") == _PLAIN

    def test_multiline_inner_preserved(self) -> None:
        result = _strip_fences("```\nline1\nline2\n```")
        assert result == "line1\nline2"


class TestDictToEmailMeetingInfo:

    def test_valid_dict_returns_model(self) -> None:
        result = _dict_to_email_meeting_info({_WINDOWS_KEY: []})
        assert isinstance(result, EmailMeetingInfo)

    def test_non_dict_window_skipped(self) -> None:
        data = {_WINDOWS_KEY: ["not_a_dict", {"time_of_day": "morning"}]}
        result = _dict_to_email_meeting_info(data)
        assert len(result.preferred_windows) == 1

    def test_empty_windows_list(self) -> None:
        result = _dict_to_email_meeting_info({_WINDOWS_KEY: []})
        assert result.preferred_windows == []

    def test_urgency_parsed_from_string(self) -> None:
        result = _dict_to_email_meeting_info({"urgency": "today", _WINDOWS_KEY: []})
        assert result.urgency.value == "today"


class TestParse:

    def test_valid_json_returns_model(self) -> None:
        raw = json.dumps({_WINDOWS_KEY: []})
        assert isinstance(MeetingInviteLLM._parse(raw), EmailMeetingInfo)

    def test_fenced_json_returns_model(self) -> None:
        inner = json.dumps({_WINDOWS_KEY: []})
        raw = f"```json\n{inner}\n```"
        assert isinstance(MeetingInviteLLM._parse(raw), EmailMeetingInfo)

    def test_invalid_json_returns_defaults(self) -> None:
        result = MeetingInviteLLM._parse("not json")
        assert result.sender_iana_timezone == _UNKNOWN_TZ
        assert result.duration_minutes == _DEFAULT_DURATION

    def test_empty_string_returns_defaults(self) -> None:
        result = MeetingInviteLLM._parse("")
        assert result.sender_iana_timezone == _UNKNOWN_TZ


class TestMeetingInvitePayload:

    def test_valid_payload_created(self) -> None:
        payload = MeetingInvitePayload(
            email_from="a@b.com",
            email_to="c@d.com",
            email_body="Hello",
            current_date="2026-01-01",
        )
        assert payload.email_from == "a@b.com"

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            MeetingInvitePayload(email_from="a@b.com")  # type: ignore[call-arg]
