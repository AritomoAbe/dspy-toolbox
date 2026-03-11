from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field

_DEFAULT_DURATION: int = 30
_UNKNOWN_TZ: str = "UNKNOWN"


class Urgency(StrEnum):
    TODAY = "today"
    THIS_WEEK = "this_week"
    NEXT_WEEK = "next_week"
    FLEXIBLE = "flexible"


class Flexibility(StrEnum):
    SPECIFIC = "specific"
    FLEXIBLE = "flexible"


class EventStatus(StrEnum):
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class AttendeeStatus(StrEnum):
    ACCEPTED = "accepted"
    DECLINED = "declined"
    TENTATIVE = "tentative"
    NEEDS_ACTION = "needsAction"


class PreferredWindow(BaseModel):
    day_of_week: Optional[str] = Field(None, description="e.g. 'Monday', 'Wednesday'")
    date: Optional[str] = Field(None, description="ISO date YYYY-MM-DD if explicit")
    time_of_day: Optional[str] = Field(None, description="e.g. '09:00', '14:00', or 'afternoon'")
    iana_timezone: Optional[str] = Field(None, description="e.g. 'America/Los_Angeles'")
    duration_hint: Optional[int] = Field(None, description="Duration in minutes if mentioned here")


class EmailMeetingInfo(BaseModel):
    sender_iana_timezone: str = Field(_UNKNOWN_TZ, description="Sender's IANA timezone")
    duration_minutes: int = Field(_DEFAULT_DURATION, description="Meeting length in minutes")
    urgency: Urgency = Field(Urgency.FLEXIBLE)
    flexibility: Flexibility = Field(Flexibility.FLEXIBLE)
    preferred_windows: list[PreferredWindow] = Field(
        default_factory=list, description="Times the sender explicitly proposed",
    )
    meeting_topic: Optional[str] = Field(None, description="Subject / purpose of meeting")
