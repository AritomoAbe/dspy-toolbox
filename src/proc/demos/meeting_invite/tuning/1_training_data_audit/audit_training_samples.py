import logging
from pathlib import Path

from proc.base.test_suite import TestSuite
from proc.pipeline.dataset.training_dataset import TrainingSetDataset
from proc.pipeline.training_set_auditor.analyze_co_occurrence import AnalyzeCoOccurrence
from proc.pipeline.training_set_auditor.analyze_expected_fields import AnalyzeExpectedFields
from proc.pipeline.training_set_auditor.analyze_input_fields import AnalyzeInputFields
from proc.pipeline.training_set_auditor.analyze_near_duplicates import AnalyzeNearDuplicates
from proc.pipeline.training_set_auditor.analyze_signal_strength import AnalyzeSignalStrength

_NEAR_DUP_TEXT_FIELDS: tuple[str, ...] = ("email_body", "email_to")
_SIG_STRENGTH_TEXT_FIELDS: tuple[str, ...] = ("email_body",)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')


class AuditTrainingSamples(TestSuite):

    def __init__(self, dataset: TrainingSetDataset) -> None:
        nodes = [
            AnalyzeExpectedFields(dataset=dataset),
            AnalyzeInputFields(dataset=dataset),
            AnalyzeNearDuplicates(dataset=dataset, text_fields=_NEAR_DUP_TEXT_FIELDS),
            AnalyzeCoOccurrence(dataset=dataset),
            AnalyzeSignalStrength(dataset=dataset, text_fields=_SIG_STRENGTH_TEXT_FIELDS),
        ]
        super().__init__(nodes)


def _main() -> None:
    dataset = TrainingSetDataset(Path(__file__).parent.parent / "dataset" / "trainset_emails_50.jsonl")
    AuditTrainingSamples(dataset).run()


if __name__ == "__main__":
    _main()
