import logging
from pathlib import Path

import dspy
import torch
from returns.result import Result

from proc.base.base_llm import BaseLLMConfig, MainModelNames
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.base.test_suite import TestSuite
from proc.demos.meeting_invite.meeting_invite_extractor_llm import MeetingInviteLLM
from proc.demos.meeting_invite.meeting_invite_score_extractor import MeetingInviteScoreExtractor
from proc.pipeline.dataset.training_dataset import TrainingSetDataset
from proc.pipeline.lora_fine_tuning.lora_fine_tuning_node import LoRAFineTuningNode
from proc.pipeline.lora_fine_tuning.models import (
    AttributionComparisonConfig,
    LoRAHyperParams,
    TrainingHyperParams,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

_LEARNING_RATE: float = 5e-5


class LoRAFineTuningTestSuite(TestSuite):
    def __init__(self, dataset: TrainingSetDataset) -> None:
        config = BaseLLMConfig(name=MainModelNames.QWEN_3_4B_INSTRUCT)
        optimized_path = Path(__file__).parent.parent / '2_bootstrap' / 'optimized_extractor_v2_0_0.json'
        llm = MeetingInviteLLM(config=config, optimized_path=optimized_path)
        scorer = MeetingInviteScoreExtractor()
        nodes: list[ProcNode] = [
            LoRAFineTuningNode(
                dataset=dataset,
                llm=llm,
                scorer=scorer,
                hf_model_name='Qwen/Qwen2.5-1.5B-Instruct',
                output_dir=Path(__file__).parent / 'runs' / 'lora_fine_tuning',
                lora=LoRAHyperParams(),
                training=TrainingHyperParams(
                    n_epochs=3,
                    learning_rate=_LEARNING_RATE,
                    warmup_steps=5,
                ),
                attr_device='cpu',
                force_dtype=torch.float32,
                attribution_comparison=AttributionComparisonConfig(enabled=False),
            ),
        ]
        super().__init__(nodes)

    def run(self) -> Result[ProcScore, ProcError]:
        original_cache = dspy.settings.lm.cache
        dspy.settings.lm.cache = False
        try:
            return super().run()
        except Exception:
            raise
        finally:
            dspy.settings.lm.cache = original_cache


def _main() -> None:
    dataset = TrainingSetDataset(
        Path(__file__).parent.parent / 'dataset' / 'testset_edge_cases_emails_7.jsonl'
    )
    LoRAFineTuningTestSuite(dataset).run()


if __name__ == '__main__':
    _main()
