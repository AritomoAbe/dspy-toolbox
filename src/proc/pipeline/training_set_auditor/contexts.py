from proc.base.proc_score import ProcScoreContext
from proc.pipeline.training_set_auditor.models import (
    CoOccurrenceEntry,
    FieldAnalysis,
    FieldDuplicateStats,
    FieldSignalResult,
)


class ExpectedFieldsContext(ProcScoreContext):

    def __init__(self, result: dict[str, FieldAnalysis]) -> None:
        super().__init__()
        self.result = result


class InputFieldsContext(ProcScoreContext):

    def __init__(self, result: dict[str, FieldAnalysis]) -> None:
        super().__init__()
        self.result = result


class NearDuplicatesContext(ProcScoreContext):

    def __init__(self, result: dict[str, FieldDuplicateStats]) -> None:
        super().__init__()
        self.result = result


class CoOccurrenceContext(ProcScoreContext):

    def __init__(self, result: list[CoOccurrenceEntry]) -> None:
        super().__init__()
        self.result = result


class SignalStrengthContext(ProcScoreContext):

    def __init__(self, result: dict[str, FieldSignalResult]) -> None:
        super().__init__()
        self.result = result
