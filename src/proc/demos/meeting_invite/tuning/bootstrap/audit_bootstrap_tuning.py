import logging
from pathlib import Path

from proc.base.base_llm import BaseLLMConfig
from proc.base.test_suite import TestSuite
from proc.demos.meeting_invite.meeting_invite_extractor_llm import MeetingInviteLLM
from proc.demos.meeting_invite.meeting_invite_score_extractor import MeetingInviteScoreExtractor
from proc.pipeline.dataset.training_dataset import TrainingSetDataset
from proc.pipeline.output_result_auditor.monte_carlo_estimation import MonteCarloEstimation
from proc.pipeline.output_result_auditor.snr_ratio import SignalToNoiseRatio

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')


class BootstrapTestSuite(TestSuite):

    def __init__(self, dataset: TrainingSetDataset) -> None:
        config = BaseLLMConfig()
        optimized_path = Path(__file__).parent / "optimized_extractor.json"
        llm = MeetingInviteLLM(config=config, optimized_path=optimized_path)
        scorer = MeetingInviteScoreExtractor()

        nodes = [
            SignalToNoiseRatio(dataset=dataset, llm=llm, scorer=scorer),
            MonteCarloEstimation(dataset=dataset, llm=llm, scorer=scorer),
        ]

        super().__init__(nodes)


def _main() -> None:
    dataset = TrainingSetDataset(Path(__file__).parent.parent / "dataset" / "emails_50.jsonl")
    BootstrapTestSuite(dataset).run()


if __name__ == "__main__":
    _main()
