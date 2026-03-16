import logging
from pathlib import Path

import dspy
import torch
from returns.result import Result

from proc.base.base_llm import BaseLLMConfig
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.base.test_suite import TestSuite
from proc.demos.meeting_invite.meeting_invite_score_extractor import MeetingInviteScoreExtractor
from proc.demos.meeting_invite.tuning.abe_gpt.dspy_gpt_test import MeetingInviteAbeGPT
from proc.demos.meeting_invite.tuning.abe_gpt.gpt import GPTLanguageModel, encode, decode, vocab_size
from proc.demos.meeting_invite.tuning.abe_gpt.gpt_hf_adapter import GptLIGAttributionAuditor
from proc.pipeline.dataset.training_dataset import TrainingSetDataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

_DEFAULT_IG_STEPS: int = 50


class PromptAttributionAbeGptTestSuite(TestSuite):

    def __init__(self, dataset: TrainingSetDataset) -> None:
        config = BaseLLMConfig()
        llm = MeetingInviteAbeGPT(config=config)
        scorer = MeetingInviteScoreExtractor()

        gpt_pt = Path(__file__).parent.parent / "abe_gpt" / "gpt.pt"
        gpt = GPTLanguageModel()  # type: ignore[no-untyped-call]
        gpt.load_state_dict(torch.load(str(gpt_pt), map_location="cpu"))

        auditor = GptLIGAttributionAuditor(
            gpt_model=gpt,
            encode_fn=encode,
            decode_fn=decode,
            vocab_size=vocab_size,
            dataset=dataset,
            llm=llm,
            scorer=scorer,
            ig_steps=_DEFAULT_IG_STEPS,
        )

        nodes: list[ProcNode] = [
            auditor,  # type: ignore[list-item]
        ]

        super().__init__(nodes)

    def run(self) -> Result[ProcScore, ProcError]:
        original_cache = dspy.settings.lm.cache
        dspy.settings.lm.cache = False
        try:  # noqa: WPS501
            return super().run()
        finally:
            dspy.settings.lm.cache = original_cache


def _main() -> None:
    # dataset = TrainingSetDataset(Path(__file__).parent.parent / "dataset" / "testset_emails_20.jsonl")
    dataset = TrainingSetDataset(Path(__file__).parent.parent / "dataset" / "testset_edge_cases_emails_7.jsonl")

    PromptAttributionAbeGptTestSuite(dataset).run()


if __name__ == "__main__":
    _main()
