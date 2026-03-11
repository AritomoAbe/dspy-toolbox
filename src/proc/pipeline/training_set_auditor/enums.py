from enum import StrEnum


class FieldType(StrEnum):
    empty = "empty"
    numeric = "numeric"
    list = "list"
    free_text = "free_text"
    categorical = "categorical"


class ListPresence(StrEnum):
    empty = "empty"
    non_empty = "non_empty"


class CorrelationRisk(StrEnum):
    high = "high"
    moderate = "moderate"


class LearnabilityLevel(StrEnum):
    strong = "strong"
    moderate = "moderate"
    weak = "weak"
    no_signal = "no_signal"


class SeparabilityLevel(StrEnum):
    distinct = "distinct"
    overlapping = "overlapping"
    indistinct = "indistinct"
