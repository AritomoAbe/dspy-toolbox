from proc.base.proc_score import ProcScoreContext
from proc.pipeline.output_result_auditor.models import MonteCarloResult, SNRResult, PromptSensitivityResult, \
    AccuracyResult, FailureClusterResult


class SNRContext(ProcScoreContext):

    def __init__(self, result: SNRResult) -> None:
        super().__init__()
        self.result = result


class MonteCarloContext(ProcScoreContext):

    def __init__(self, result: MonteCarloResult) -> None:
        super().__init__()
        self.result = result


class PromptSensitivityContext(ProcScoreContext):

    def __init__(self, result: PromptSensitivityResult) -> None:
        super().__init__()
        self.result = result


class AccuracyContext(ProcScoreContext):

    def __init__(self, result: AccuracyResult) -> None:
        super().__init__()
        self.result = result


class FailureClusterContext(ProcScoreContext):

    def __init__(self, result: FailureClusterResult) -> None:
        super().__init__()
        self.result = result
