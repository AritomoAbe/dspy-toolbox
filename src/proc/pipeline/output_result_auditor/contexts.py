from proc.base.proc_score import ProcScoreContext
from proc.pipeline.output_result_auditor.models import MonteCarloResult, SNRResult


class SNRContext(ProcScoreContext):

    def __init__(self, result: SNRResult) -> None:
        super().__init__()
        self.result = result


class MonteCarloContext(ProcScoreContext):

    def __init__(self, result: MonteCarloResult) -> None:
        super().__init__()
        self.result = result
