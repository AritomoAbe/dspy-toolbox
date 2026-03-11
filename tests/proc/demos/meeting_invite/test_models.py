import pytest
from pydantic import ValidationError

from proc.demos.meeting_invite.models import (
    AttendeeStatus,
    EmailMeetingInfo,
    EventStatus,
    Flexibility,
    PreferredWindow,
    Urgency,
    _DEFAULT_DURATION,
    _UNKNOWN_TZ,
)


class TestEnumValues:

    def test_urgency_today(self) -> None:
        assert Urgency.TODAY.value == "today"

    def test_urgency_this_week(self) -> None:
        assert Urgency.THIS_WEEK.value == "this_week"

    def test_urgency_next_week(self) -> None:
        assert Urgency.NEXT_WEEK.value == "next_week"

    def test_urgency_flexible(self) -> None:
        assert Urgency.FLEXIBLE.value == "flexible"

    def test_flexibility_specific(self) -> None:
        assert Flexibility.SPECIFIC.value == "specific"

    def test_flexibility_flexible(self) -> None:
        assert Flexibility.FLEXIBLE.value == "flexible"


class TestEventAttendeeStatus:

    def test_event_confirmed(self) -> None:
        assert EventStatus.CONFIRMED.value == "confirmed"

    def test_event_tentative(self) -> None:
        assert EventStatus.TENTATIVE.value == "tentative"

    def test_event_cancelled(self) -> None:
        assert EventStatus.CANCELLED.value == "cancelled"

    def test_attendee_accepted(self) -> None:
        assert AttendeeStatus.ACCEPTED.value == "accepted"

    def test_attendee_declined(self) -> None:
        assert AttendeeStatus.DECLINED.value == "declined"

    def test_attendee_needs_action(self) -> None:
        assert AttendeeStatus.NEEDS_ACTION.value == "needsAction"


class TestPreferredWindow:

    def test_all_fields_default_to_none(self) -> None:
        window = PreferredWindow()
        assert window.day_of_week is None
        assert window.date is None
        assert window.time_of_day is None

    def test_model_validate_from_dict(self) -> None:
        window = PreferredWindow.model_validate({"time_of_day": "morning"})
        assert window.time_of_day == "morning"

    def test_duration_hint_accepted(self) -> None:
        window = PreferredWindow.model_validate({"duration_hint": _DEFAULT_DURATION})
        assert window.duration_hint == _DEFAULT_DURATION


class TestEmailMeetingInfo:

    def test_default_timezone(self) -> None:
        assert EmailMeetingInfo().sender_iana_timezone == _UNKNOWN_TZ

    def test_default_duration(self) -> None:
        assert EmailMeetingInfo().duration_minutes == _DEFAULT_DURATION

    def test_default_urgency(self) -> None:
        assert EmailMeetingInfo().urgency == Urgency.FLEXIBLE

    def test_default_flexibility(self) -> None:
        assert EmailMeetingInfo().flexibility == Flexibility.FLEXIBLE

    def test_default_windows_empty(self) -> None:
        assert EmailMeetingInfo().preferred_windows == []

    def test_urgency_from_string(self) -> None:
        info = EmailMeetingInfo.model_validate({"urgency": "today"})
        assert info.urgency == Urgency.TODAY

    def test_invalid_urgency_raises(self) -> None:
        with pytest.raises(ValidationError):
            EmailMeetingInfo.model_validate({"urgency": "invalid"})
