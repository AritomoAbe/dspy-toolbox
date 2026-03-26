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
from proc.pipeline.llm_prompt_usage_attribution.prompt_attribution_node import PromptAttributionNode

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

_SWAP_ALERT_MB: int = 200
_TOKEN_LIMIT: int = 12


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


class PromptAttributionTestSuite(TestSuite):
    def __init__(self, dataset: TrainingSetDataset) -> None:
        config = BaseLLMConfig(name=MainModelNames.QWEN_3_4B_INSTRUCT)
        optimized_path = Path(__file__).parent / 'optimized_extractor_v2_0_0.json'
        llm = MeetingInviteLLM(config=config, optimized_path=optimized_path)
        scorer = MeetingInviteScoreExtractor()
        nodes: list[ProcNode] = [
            PromptAttributionNode(
                dataset=dataset,
                llm=llm,
                scorer=scorer,
                hf_model_name='Qwen/Qwen2.5-1.5B-Instruct',
                output_dir=Path(__file__).parent / 'runs' / 'prompt_attribution',
                generation_max_new_tokens=_TOKEN_LIMIT,
                ig_steps=24,
                attr_device='cpu',
                force_dtype=torch.float32,
                target_text=None,
                save_html=True,
                save_plots=True,
                top_k_tokens=_TOKEN_LIMIT,
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
    dataset = TrainingSetDataset(Path(__file__).parent.parent / 'dataset' / 'testset_edge_cases_emails_7.jsonl')
    PromptAttributionTestSuite(dataset).run()


if __name__ == '__main__':
    _main()
