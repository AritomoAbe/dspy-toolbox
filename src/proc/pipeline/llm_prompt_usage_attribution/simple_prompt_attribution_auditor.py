import logging
from collections import defaultdict
from typing import Any

import dspy
import numpy as np
from returns.result import Result, Success, Failure

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.llm_prompt_usage_attribution.contexts import AttributionContext
from proc.pipeline.llm_prompt_usage_attribution.models import (
    ExampleAttribution, SegmentAttribution, PromptSegmentType, AttributionResult,
)
from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor, INVALID_SCORE

_ZERO_SCORE: float = float()
_SEGMENT_PREVIEW_LEN: int = 80
_MASK_PLACEHOLDER: str = "[SEGMENT REMOVED]"
_ACTIVE_THRESHOLD: float = 0.1
_DEADWEIGHT_THRESHOLD: float = 0.02


def _demo_marker(score: float) -> str:
    if score > _ACTIVE_THRESHOLD:
        return "✓"
    if score > _DEADWEIGHT_THRESHOLD:
        return "~"
    return "✗"


class SimplePromptAttributionAuditor(ProcNode):
    """
    Stage 3 — Score How the LLM Uses the Prompt.

    Uses leave-one-out perturbation attribution: for each logical segment of
    the compiled prompt (INSTRUCTION, each DEMO, INPUT), we temporarily
    replace it with a blank placeholder and re-run the model. The attribution
    score for that segment is:

        attribution = score_full - score_ablated

    A high positive score means the segment is actively driving correct
    predictions. A near-zero score means the model is ignoring it. A negative
    score means the segment is actively hurting performance.

    When ablating a segment causes the LLM to produce unparseable output
    (INVALID_SCORE), score_ablated is treated as 0.0 — the segment was so
    critical that the model could not produce any valid output without it.
    This gives attribution = score_full, the maximum possible signal.

    This approach works with any API-based LLM (no access to model weights
    required), making it compatible with the existing DSpyLLM abstraction.
    It is the practical alternative to Captum's gradient-based attribution,
    which requires a local PyTorch model.

    What it reveals:
        - Whether few-shot demos are doing real work or are dead weight
        - Whether the instruction text is being attended to
        - Which specific demos to prune from the compiled prompt
        - Whether the model is solving the task from the input alone

    Thresholds:
        avg_instruction_attribution > 0.1   →  instruction is actively used
        avg_instruction_attribution < 0.05  →  model is ignoring instruction
        avg_demo_attribution > 0.1          →  demos are doing real work
        avg_demo_attribution < 0.02         →  demos are dead weight
    """

    _SCORE: float = 1.0

    def __init__(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._dataset = dataset
        self._llm = llm
        self._scorer = scorer

    def invoke(self) -> Result[ProcScore, ProcError]:
        predictors = self._llm.predictors()

        if not predictors:
            return Failure(ProcError(
                "No predictors found in LLM — cannot access prompt segments"
            ))

        # We work with the first predictor — in a single-predictor module
        # this is the only one; in a chain, the first is the primary extractor.
        predictor = predictors[0]
        original_demos = list(predictor.demos)
        n_demos = len(original_demos)

        if n_demos == 0:
            return Failure(ProcError(
                "Predictor has no demos — run BootstrapFewShot before PromptAttributionAuditor"
            ))

        example_attributions: list[ExampleAttribution] = []

        for index, example in enumerate(self._dataset.load()):
            self._logger.info(f"Processing example {index}")

            # --- Baseline: full prompt, no masking ---
            score_full = self._run_score(example, predictor, original_demos)

            if score_full is None:
                return Failure(ProcError(f"Cannot score example {index} (baseline)"))

            self._logger.info(f"Example {index} baseline score: {score_full:.4f}")

            segments: list[SegmentAttribution] = []

            # --- Ablate INSTRUCTION ---
            instruction_result = self._ablate_instruction(
                example, predictor, original_demos, score_full
            )
            segments.append(instruction_result)

            # --- Ablate each DEMO one at a time ---
            for demo_index, demo in enumerate(original_demos):
                demo_result = self._ablate_demo(
                    example, predictor, original_demos,
                    demo_index, score_full
                )
                segments.append(demo_result)

            example_attributions.append(ExampleAttribution(
                example_index=index,
                score_full=score_full,
                segments=segments,
            ))

        # Restore always (though we restore after each ablation, belt-and-braces)
        predictor.demos = original_demos

        result = self._aggregate(example_attributions, n_demos)

        return Success(ProcScore(
            value=self._score(),
            context=AttributionContext(result)
        ))

    # ------------------------------------------------------------------
    # Ablation helpers
    # ------------------------------------------------------------------

    def _ablate_instruction(
        self,
        example: Any,
        predictor: Any,
        original_demos: list,
        score_full: float,
    ) -> SegmentAttribution:
        """Replace the signature instructions with a blank and re-score."""
        original_instructions = predictor.signature.instructions
        predictor.signature = predictor.signature.with_instructions(_MASK_PLACEHOLDER)
        try:
            score_ablated = self._run_score(example, predictor, original_demos, is_ablation=True) or _ZERO_SCORE
        except Exception:
            raise
        finally:
            predictor.signature = predictor.signature.with_instructions(original_instructions)
        result = SegmentAttribution(
            segment_type=PromptSegmentType.INSTRUCTION,
            segment_index=0,
            segment_preview=original_instructions[:_SEGMENT_PREVIEW_LEN].replace("\n", " "),
            attribution_score=score_full - score_ablated,
            score_full=score_full,
            score_ablated=score_ablated,
        )
        self._logger.info(
            f"INSTRUCTION attribution: {result.attribution_score:.4f}"
        )
        return result

    def _ablate_demo(
        self,
        example: Any,
        predictor: Any,
        original_demos: list,
        demo_index: int,
        score_full: float,
    ) -> SegmentAttribution:
        """Remove one demo and re-score with the remaining demos."""
        ablated_demos = [d for i, d in enumerate(original_demos) if i != demo_index]
        predictor.demos = ablated_demos
        try:
            score_ablated = self._run_score(example, predictor, ablated_demos, is_ablation=True) or _ZERO_SCORE
        except Exception:
            raise
        finally:
            predictor.demos = original_demos
        result = SegmentAttribution(
            segment_type=PromptSegmentType.DEMO,
            segment_index=demo_index,
            segment_preview=str(original_demos[demo_index])[:_SEGMENT_PREVIEW_LEN].replace("\n", " "),
            attribution_score=score_full - score_ablated,
            score_full=score_full,
            score_ablated=score_ablated,
        )
        self._logger.info(
            f"DEMO[{demo_index}] attribution: "
            f"{result.attribution_score:.4f}  "
            f"preview='{result.segment_preview}'"
        )
        return result

    def _run_score(
        self,
        example: Any,
        predictor: Any,
        demos: list,
        is_ablation: bool = False,
    ) -> float | None:
        """
        Run inference and return the score.

        Returns None (baseline failure) or _ZERO_SCORE (ablation
        fallback) when the scorer returns INVALID_SCORE. During ablation,
        an unparseable output is treated as score=0.0 — the segment was so
        critical that removing it completely broke the model's output format.
        """
        predictor.demos = demos
        with dspy.context(cache=False):
            pred = self._llm(**example.inputs())
        score = self._scorer.extraction_metric(example, pred)
        if score == INVALID_SCORE:
            if is_ablation:
                self._logger.warning(
                    "Ablation produced unparseable output — "
                    "treating as score=0.0 (segment is critical)"
                )
                return _ZERO_SCORE
            return None
        return float(score)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        example_attributions: list[ExampleAttribution],
        n_demos: int,
    ) -> AttributionResult:
        instruction_scores = [
            ea.instruction_attribution for ea in example_attributions
        ]
        all_demo_scores = [
            score
            for ea in example_attributions
            for score in ea.demo_attributions
        ]

        # Per-demo-index average across all examples
        per_demo: dict[int, list[float]] = defaultdict(list)
        for ea in example_attributions:
            for seg in ea.segments:
                if seg.segment_type == PromptSegmentType.DEMO:
                    per_demo[seg.segment_index].append(seg.attribution_score)

        per_demo_avg = {
            idx: float(np.mean(scores))
            for idx, scores in per_demo.items()
        }

        result = AttributionResult(
            avg_instruction_attribution=float(np.mean(instruction_scores)),
            avg_demo_attribution=float(np.mean(all_demo_scores)) if all_demo_scores else _ZERO_SCORE,
            per_demo_avg_attribution=per_demo_avg,
            example_attributions=example_attributions,
            n_examples=len(example_attributions),
            n_demos=n_demos,
        )

        self._logger.info(
            f"Avg instruction attribution: {result.avg_instruction_attribution:.4f}  "
            f"(>0.1 used, <0.05 ignored)"
        )
        self._logger.info(
            f"Avg demo attribution:        {result.avg_demo_attribution:.4f}  "
            f"(>0.1 real work, <0.02 dead weight)"
        )
        for demo_idx, score in result.per_demo_avg_attribution.items():
            marker = _demo_marker(score)
            self._logger.info(f"  [{marker}] DEMO[{demo_idx}]: {score:.4f}")

        return result

    def _score(self) -> float:
        return self._SCORE
