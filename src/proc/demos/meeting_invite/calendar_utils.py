import re
from types import MappingProxyType

_TOD_RANGES: MappingProxyType[str, tuple[int, int]] = MappingProxyType({
    "morning": (9, 12),
    "afternoon": (12, 17),
    "evening": (17, 20),
    "lunch": (12, 14),
})

_NOON_HOUR: int = 12


def _parse_12h(s: str) -> str | None:
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", s)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or "0")
    period = m.group(3)
    if period == "pm":
        if hour != _NOON_HOUR:
            hour += _NOON_HOUR
    elif hour == _NOON_HOUR:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def _normalise_time_of_day(raw: str) -> str:
    if not raw:
        return raw
    s = raw.strip().lower()
    if s in _TOD_RANGES:
        return s
    result = _parse_12h(s)
    if result:
        return result
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return s
