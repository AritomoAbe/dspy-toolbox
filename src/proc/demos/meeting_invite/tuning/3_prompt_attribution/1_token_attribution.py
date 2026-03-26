import logging
import threading
from pathlib import Path

import dspy
import psutil
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
from proc.pipeline.llm_prompt_usage_attribution.token_attribution_auditor import TokenAttributionAuditor

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

_IG_STEPS: int = 360
_SWAP_ALERT_MB: int = 200


def _watch_swap(stop: threading.Event) -> None:
    baseline = psutil.swap_memory().used
    while not stop.is_set():
        s = psutil.swap_memory()
        v = psutil.virtual_memory()
        delta = (s.used - baseline) / 1024 ** 2
        if delta > _SWAP_ALERT_MB:
            logging.warning(
                f"SWAP +{delta:.0f} MB | "
                f"RAM free: {v.available / 1024 ** 2:.0f} MB | "
                f"RAM used: {v.percent:.0f}%"
            )
        stop.wait(10)


def _stop_watcher(stop: threading.Event, watcher: threading.Thread) -> None:
    stop.set()
    watcher.join()


class TokenAttributionTestSuite(TestSuite):

    def __init__(self, dataset: TrainingSetDataset) -> None:
        config = BaseLLMConfig(
            name=MainModelNames.QWEN_3_4B_INSTRUCT
        )
        optimized_path = Path(__file__).parent / "optimized_extractor_v2_0_0.json"
        llm = MeetingInviteLLM(config=config, optimized_path=optimized_path)
        scorer = MeetingInviteScoreExtractor()

        # h_model_name = "Qwen/Qwen3-4B-Instruct-2507"
        h_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        nodes: list[ProcNode] = [
            TokenAttributionAuditor(
                dataset=dataset,
                llm=llm,
                scorer=scorer,
                hf_model_name=h_model_name,
                ig_steps=_IG_STEPS,
                attr_device="cpu",
                force_dtype=torch.float32,
            ),
        ]

        super().__init__(nodes)

    def run(self) -> Result[ProcScore, ProcError]:
        stop = threading.Event()
        watcher = threading.Thread(target=_watch_swap, args=(stop,), daemon=True)
        watcher.start()

        original_cache = dspy.settings.lm.cache
        dspy.settings.lm.cache = False
        try:
            return super().run()
        except Exception:
            raise
        finally:
            dspy.settings.lm.cache = original_cache
            _stop_watcher(stop, watcher)


def _main() -> None:
    # dataset = TrainingSetDataset(Path(__file__).parent.parent / "dataset" / "testset_emails_20.jsonl")
    dataset = TrainingSetDataset(Path(__file__).parent.parent / "dataset" / "testset_edge_cases_emails_7.jsonl")

    TokenAttributionTestSuite(dataset).run()


if __name__ == "__main__":
    _main()
