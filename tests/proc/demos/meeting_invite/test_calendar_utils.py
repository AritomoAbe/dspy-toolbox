from proc.demos.meeting_invite.calendar_utils import _normalise_time_of_day

_MORNING: str = "morning"


class TestNormaliseKeywords:

    def test_morning(self) -> None:
        assert _normalise_time_of_day(_MORNING) == _MORNING

    def test_afternoon(self) -> None:
        assert _normalise_time_of_day("afternoon") == "afternoon"

    def test_evening(self) -> None:
        assert _normalise_time_of_day("evening") == "evening"

    def test_lunch(self) -> None:
        assert _normalise_time_of_day("lunch") == "lunch"

    def test_uppercase_normalised(self) -> None:
        assert _normalise_time_of_day("MORNING") == _MORNING

    def test_mixed_case_normalised(self) -> None:
        assert _normalise_time_of_day("Afternoon") == "afternoon"


class TestNormaliseTwelveHour:

    def test_am_hour(self) -> None:
        assert _normalise_time_of_day("9am") == "09:00"

    def test_pm_hour(self) -> None:
        assert _normalise_time_of_day("2pm") == "14:00"

    def test_noon_pm(self) -> None:
        assert _normalise_time_of_day("12pm") == "12:00"

    def test_midnight_am(self) -> None:
        assert _normalise_time_of_day("12am") == "00:00"

    def test_am_with_minutes(self) -> None:
        assert _normalise_time_of_day("9:30am") == "09:30"

    def test_pm_with_minutes(self) -> None:
        assert _normalise_time_of_day("3:15pm") == "15:15"


class TestNormaliseTwentyFourHour:

    def test_single_digit_hour(self) -> None:
        assert _normalise_time_of_day("9:00") == "09:00"

    def test_double_digit_hour(self) -> None:
        assert _normalise_time_of_day("14:00") == "14:00"

    def test_with_minutes(self) -> None:
        assert _normalise_time_of_day("8:30") == "08:30"


class TestNormaliseEdgeCases:

    def test_empty_string(self) -> None:
        assert _normalise_time_of_day("") == ""

    def test_unknown_string_returned_as_is(self) -> None:
        assert _normalise_time_of_day("late night") == "late night"

    def test_whitespace_stripped(self) -> None:
        assert _normalise_time_of_day(f"  {_MORNING}  ") == _MORNING
