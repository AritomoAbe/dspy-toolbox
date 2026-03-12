import logging
import time
from pathlib import Path

from dspy import BootstrapFewShot

from proc.base.base_llm import BaseLLMConfig
from proc.demos.meeting_invite.meeting_invite_extractor_llm import MeetingInviteLLM
from proc.demos.meeting_invite.meeting_invite_score_extractor import MeetingInviteScoreExtractor
from proc.pipeline.dataset.training_dataset import TrainingSetDataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

LOGGER = logging.getLogger(__name__)


def _main() -> None:
    dataset = TrainingSetDataset(Path(__file__).parent.parent / "dataset" / "emails_50.jsonl")

    config = BaseLLMConfig()
    llm = MeetingInviteLLM(config=config)

    scorer = MeetingInviteScoreExtractor()

    optimizer = BootstrapFewShot(
        metric=scorer.extraction_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        max_rounds=2,
    )

    trainset = dataset.load()
    LOGGER.info("Starting BootstrapFewShot optimisation on %d examples…", len(trainset))
    optimized = optimizer.compile(llm, trainset=trainset)
    LOGGER.info("BootstrapFewShot complete.")

    save_path = Path(__file__).parent / f'optimized_extractor_bootstrap_{time.time()}.json'
    optimized.save(str(save_path))


if __name__ == "__main__":
    _main()
