from proc.base.proc_score import ProcScoreContext
from proc.pipeline.llm_prompt_usage_attribution.models import AttributionResult, LIGAttributionResult


class AttributionContext(ProcScoreContext):

    def __init__(self, result: AttributionResult) -> None:
        super().__init__()
        self.result = result


class LIGAttributionContext(ProcScoreContext):

    def __init__(self, result: LIGAttributionResult) -> None:
        super().__init__()
        self.result = result
