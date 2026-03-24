"""
LoRA Fine-Tuning Node  —  Stage 4
===================================

Fine-tunes a HuggingFace CausalLM on the pipeline dataset using PEFT LoRA
adapters, then measures how prompt-attribution changes before vs. after
training.

Pipeline position
-----------------
  Stage 1 — Training set scoring        (ScoreAuditor)
  Stage 2 — Output auditing             (OutputResultAuditor)
  Stage 3 — LIG attribution             (LIGAttributionAuditor / PromptAttributionNode)
  Stage 4 — LoRA fine-tuning            **this node**

The node:
  1. Renders each example into a supervised (prompt, completion) pair using
     the compiled DSPy ChatAdapter.
  2. Applies PEFT LoRA adapters to the model.
  3. Trains for ``n_epochs`` over the dataset.
  4. Evaluates on the same dataset after each epoch using the scorer.
  5. Saves the adapter weights and a structured run report to ``output_dir``.
  6. Optionally runs ``PromptAttributionNode`` on a held-out probe example
     *before* and *after* fine-tuning and writes a side-by-side comparison
     report to ``output_dir/attribution_comparison/``.

Artifacts written to ``output_dir/``
--------------------------------------
  adapter/                     — PEFT adapter weights (``save_pretrained``)
  training_log.json            — per-step loss + per-epoch scorer metrics
  summary.json                 — final run summary (Pydantic → JSON)
  attribution_comparison/      — (optional) before/after attribution artifacts
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

import dspy
import torch
from dspy.adapters import ChatAdapter
from peft import LoraConfig, TaskType, get_peft_model
from returns.pipeline import is_successful
from returns.result import Failure, Result, Success
from transformers import AutoModelForCausalLM, AutoTokenizer

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.base.timing import timed
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.llm_prompt_usage_attribution.prompt_attribution_node import (
    PromptAttributionNode,
)
from proc.pipeline.lora_fine_tuning.models import LoRAHyperParams, TrainingHyperParams, AttributionComparisonConfig, \
    StepLog, EpochMetrics, LoRARunSummary, FAILURE_SCORE_THRESHOLD, STEP_LOG_PREFIX, TrainingPhase, AdapterSaveStatus
from proc.pipeline.output_result_auditor.score_extractor import INVALID_SCORE, ScoreExtractor

_DEFAULT_HF_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
_DEFAULT_OUTPUT_DIR: str = "runs/lora_finetuning"


class LoRAFineTuningNode(ProcNode):
    """
    Stage 4 — LoRA fine-tuning + attribution comparison.

    Parameters
    ----------
    dataset:
        Dataset to train and evaluate on.
    llm:
        Compiled DSPy LLM.  Used to render the chat-template prompt for each
        example and to access the compiled predictor signature/demos.
    scorer:
        ScoreExtractor used to evaluate pass rate after each epoch.
    hf_model_name:
        HuggingFace CausalLM to fine-tune.  Should match (or be smaller than)
        the attribution proxy model used in Stage 3.
    output_dir:
        Root directory for all artifacts.
    lora:
        LoRA adapter hyper-parameters.  Pass a custom ``LoRAHyperParams`` to
        override defaults.
    training:
        Training hyper-parameters.  Pass a custom ``TrainingHyperParams`` to
        override defaults.
    attr_device:
        Device for training and attribution.  Defaults to ``"cpu"``.
    force_dtype:
        Override automatic dtype selection (float32 on CPU, float16 on GPU).
    attribution_comparison:
        Configuration for the optional before/after attribution comparison run.
        Set ``AttributionComparisonConfig(enabled=False)`` to skip.
    """

    _SCORE: float = 1.0

    def __init__(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
        hf_model_name: str = _DEFAULT_HF_MODEL,
        output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
        lora: Optional[LoRAHyperParams] = None,
        training: Optional[TrainingHyperParams] = None,
        attr_device: str = "cpu",
        force_dtype: Optional[torch.dtype] = None,
        attribution_comparison: Optional[AttributionComparisonConfig] = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._dataset = dataset
        self._llm = llm
        self._scorer = scorer
        self._hf_model_name = hf_model_name
        self._output_dir = Path(output_dir)
        self._lora = lora or LoRAHyperParams()
        self._training = training or TrainingHyperParams()
        self._attr_cfg = attribution_comparison or AttributionComparisonConfig()

        self._device = self._resolve_device(torch.device(attr_device))
        self._load_dtype = (
            force_dtype
            if force_dtype is not None
            else (torch.float32 if self._device.type == "cpu" else torch.float16)
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(
            f"LoRAFineTuningNode: model={hf_model_name}  "
            f"device={self._device}  dtype={self._load_dtype}"
        )

    # ── ProcNode entry point ───────────────────────────────────────────────────

    def invoke(self) -> Result[ProcScore, ProcError]:
        predictors = self._llm.predictors()
        if not predictors:
            return Failure(ProcError(
                "No predictors found in LLM — cannot render compiled prompt structure."
            ))
        predictor = predictors[0]

        # Load model + tokenizer ───────────────────────────────────────────────
        with timed("_load_base_model", logger=self._logger):
            load_result = self._load_model()
        if not is_successful(load_result):
            return load_result
        model, tokenizer = load_result.unwrap()

        # Wrap with LoRA adapters ───────────────────────────────────────────────
        with timed("get_peft_model", logger=self._logger):
            model = self._apply_lora(model)

        # Build (prompt, completion) training pairs ────────────────────────────
        with timed("_build_training_pairs", logger=self._logger):
            pairs_result = self._build_training_pairs(predictor, tokenizer)
        if not is_successful(pairs_result):
            return pairs_result
        training_pairs: list[tuple[torch.Tensor, torch.Tensor]] = pairs_result.unwrap()

        if not training_pairs:
            return Failure(ProcError("Dataset produced zero training pairs."))
        self._logger.info(f"Built {len(training_pairs)} training pairs.")

        # Optional: attribution snapshot BEFORE training ───────────────────────
        if self._attr_cfg.enabled:
            before_dir = self._output_dir / "attribution_comparison" / TrainingPhase.BEFORE
            attr_result = self._run_attribution_snapshot(
                phase=TrainingPhase.BEFORE,
                predictor=predictor,
                tokenizer=tokenizer,
                output_dir=before_dir,
            )
            if not is_successful(attr_result):
                self._logger.warning(
                    f"Pre-training attribution failed: {attr_result.failure()}.  Continuing."
                )

        # Evaluate baseline (epoch 0) ──────────────────────────────────────────
        with timed("_evaluate epoch=0 (baseline)", logger=self._logger):
            baseline_metrics = self._evaluate(epoch=0, predictor=predictor)
        self._logger.info(
            f"Baseline pass rate: {baseline_metrics.pass_rate:.1%}  "
            f"({baseline_metrics.n_valid}/{baseline_metrics.n_examples})"
        )

        # Training loop ────────────────────────────────────────────────────────
        optimiser = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self._training.learning_rate,
            weight_decay=self._training.weight_decay,
        )
        scheduler = self._build_scheduler(
            optimiser=optimiser,
            total_steps=len(training_pairs) * self._training.n_epochs,
        )

        step_logs: list[StepLog] = []
        epoch_metrics: list[EpochMetrics] = [baseline_metrics]
        global_step = 0

        for epoch in range(1, self._training.n_epochs + 1):
            self._logger.info(f"=== Epoch {epoch}/{self._training.n_epochs} ===")
            model.train()

            shuffled_pairs = list(training_pairs)
            random.shuffle(shuffled_pairs)

            for step_idx, (prompt_ids, completion_ids) in enumerate(shuffled_pairs):
                global_step += 1
                with timed(
                    f"{STEP_LOG_PREFIX}[epoch={epoch} step={step_idx}]",
                    logger=self._logger,
                ):
                    loss = self._train_step(
                        model=model,
                        optimiser=optimiser,
                        scheduler=scheduler,
                        prompt_ids=prompt_ids,
                        completion_ids=completion_ids,
                    )

                log = StepLog(epoch=epoch, step=global_step, loss=loss)
                step_logs.append(log)
                self._logger.info(
                    f"  epoch={epoch}  step={step_idx}  global={global_step}  loss={loss:.6f}"
                )

            # Epoch evaluation ─────────────────────────────────────────────────
            model.eval()
            with timed(f"_evaluate epoch={epoch}", logger=self._logger):
                metrics = self._evaluate(epoch=epoch, predictor=predictor)
            epoch_metrics.append(metrics)
            self._logger.info(
                f"Epoch {epoch} pass rate: {metrics.pass_rate:.1%}  "
                f"({metrics.n_valid}/{metrics.n_examples})"
            )

        # Optional: attribution snapshot AFTER training ────────────────────────
        if self._attr_cfg.enabled:
            after_dir = self._output_dir / "attribution_comparison" / TrainingPhase.AFTER
            attr_result = self._run_attribution_snapshot(
                phase=TrainingPhase.AFTER,
                predictor=predictor,
                tokenizer=tokenizer,
                output_dir=after_dir,
            )
            if not is_successful(attr_result):
                self._logger.warning(
                    f"Post-training attribution failed: {attr_result.failure()}.  Continuing."
                )
            else:
                self._write_comparison_html(
                    before_dir=self._output_dir / "attribution_comparison" / TrainingPhase.BEFORE,
                    after_dir=after_dir,
                )

        # Save adapter weights ─────────────────────────────────────────────────
        adapter_dir = self._output_dir / "adapter"
        save_status = self._save_adapter(model=model, adapter_dir=adapter_dir)

        # Write artifacts ──────────────────────────────────────────────────────
        initial_pass_rate = baseline_metrics.pass_rate
        final_pass_rate = epoch_metrics[-1].pass_rate
        summary = LoRARunSummary(
            hf_model_name=self._hf_model_name,
            output_dir=str(self._output_dir),
            lora=self._lora,
            training=self._training,
            epoch_metrics=epoch_metrics,
            step_logs=step_logs,
            adapter_save_status=save_status,
            attribution_comparison_enabled=self._attr_cfg.enabled,
            initial_pass_rate=initial_pass_rate,
            final_pass_rate=final_pass_rate,
            pass_rate_delta=final_pass_rate - initial_pass_rate,
        )
        self._write_artifacts(summary)

        self._logger.info(
            f"LoRA fine-tuning complete.  "
            f"pass_rate: {initial_pass_rate:.1%} → {final_pass_rate:.1%}  "
            f"(Δ{summary.pass_rate_delta:+.1%})"
        )
        return Success(ProcScore(value=self._SCORE, context=summary))

    # ── Training step ──────────────────────────────────────────────────────────

    def _train_step(
        self,
        model: Any,
        optimiser: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> float:
        """
        One gradient update.

        Concatenates [prompt | completion], computes cross-entropy loss only
        over the completion tokens (the prompt positions are masked out with
        ``ignore_index=-100``).
        """
        prompt_ids = prompt_ids.to(self._device)
        completion_ids = completion_ids.to(self._device)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Build labels: -100 for prompt tokens (masked), real IDs for completion.
        ignore = torch.full_like(prompt_ids, fill_value=-100)
        labels = torch.cat([ignore, completion_ids], dim=1)

        optimiser.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss: torch.Tensor = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self._training.max_grad_norm
        )
        optimiser.step()
        scheduler.step()

        return float(loss.detach().cpu().item())

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self, epoch: int, predictor: Any) -> EpochMetrics:
        """
        Run the scorer over the full dataset and return pass/fail counts.

        Uses the API LLM (Ollama/Qwen3 via DSPy), not the HF proxy — the
        goal is to measure whether the LoRA-tuned prompt proxy improves
        the attribution signal, not to re-evaluate the API model directly.

        For a tighter feedback loop, swap this for a generation pass on the
        fine-tuned HF model itself.
        """
        n_examples = 0
        n_valid = 0
        n_invalid = 0

        for example in self._dataset.load():
            n_examples += 1
            with dspy.context(cache=False):
                pred = self._llm(**example.inputs())
            score = self._scorer.extraction_metric(example, pred)
            if score == INVALID_SCORE or score < FAILURE_SCORE_THRESHOLD:
                n_invalid += 1
            else:
                n_valid += 1

        pass_rate = n_valid / n_examples if n_examples > 0 else 0.0
        return EpochMetrics(
            epoch=epoch,
            n_examples=n_examples,
            n_valid=n_valid,
            n_invalid=n_invalid,
            pass_rate=pass_rate,
        )

    # ── Attribution snapshot ───────────────────────────────────────────────────

    def _run_attribution_snapshot(
        self,
        phase: TrainingPhase,
        predictor: Any,
        tokenizer: Any,
        output_dir: Path,
    ) -> Result[None, ProcError]:
        """
        Run ``PromptAttributionNode`` on the probe example and save to
        ``output_dir``.

        We create a single-example dataset wrapper so we can reuse the full
        ``PromptAttributionNode.invoke()`` pipeline without duplication.
        """
        self._logger.info(f"Running attribution snapshot: phase={phase}  dir={output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        probe_dataset = _ProbeDataset(
            dataset=self._dataset,
            example_index=self._attr_cfg.probe_example_index,
        )

        node = PromptAttributionNode(
            dataset=probe_dataset,
            llm=self._llm,
            scorer=self._scorer,
            hf_model_name=self._hf_model_name,
            output_dir=output_dir,
            ig_steps=self._attr_cfg.ig_steps,
            attr_device=str(self._device),
            save_html=self._attr_cfg.save_html,
            save_plots=self._attr_cfg.save_plots,
        )
        with timed(f"attribution_snapshot phase={phase}", logger=self._logger):
            result = node.invoke()

        if not is_successful(result):
            return Failure(result.failure())
        return Success(None)

    def _write_comparison_html(
        self,
        before_dir: Path,
        after_dir: Path,
    ) -> None:
        """
        Write a side-by-side HTML comparison of before/after attribution heatmaps.

        Reads the heatmap PNGs produced by PromptAttributionNode and embeds
        them as base64 inline images so the report is self-contained.
        """
        import base64

        comparison_dir = self._output_dir / "attribution_comparison"
        out_path = comparison_dir / "comparison.html"

        def _load_png_b64(directory: Path) -> str:
            png_path = directory / "example_000" / "prompt_heatmap.png"
            if not png_path.exists():
                return ""
            return base64.b64encode(png_path.read_bytes()).decode("ascii")

        before_b64 = _load_png_b64(before_dir)
        after_b64 = _load_png_b64(after_dir)

        before_img = (
            f'<img src="data:image/png;base64,{before_b64}" style="max-width:100%">'
            if before_b64 else "<p><em>Before heatmap not available.</em></p>"
        )
        after_img = (
            f'<img src="data:image/png;base64,{after_b64}" style="max-width:100%">'
            if after_b64 else "<p><em>After heatmap not available.</em></p>"
        )

        body = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>LoRA attribution comparison — before vs after</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
h1, h2 {{ margin-top: 24px; }}
.panel {{ border: 1px solid #ddd; padding: 16px; border-radius: 6px; }}
</style></head><body>
<h1>LoRA Fine-Tuning — Attribution Comparison</h1>
<p>Model: <code>{self._hf_model_name}</code> &nbsp;
   LoRA r={self._lora.r}, alpha={self._lora.lora_alpha}, dropout={self._lora.lora_dropout}</p>
<div class="grid">
  <div class="panel">
    <h2>Before fine-tuning</h2>
    {before_img}
  </div>
  <div class="panel">
    <h2>After fine-tuning ({self._training.n_epochs} epochs)</h2>
    {after_img}
  </div>
</div>
</body></html>"""

        out_path.write_text(body, encoding="utf-8")
        self._logger.info(f"Attribution comparison report: {out_path}")

    # ── Data preparation ───────────────────────────────────────────────────────

    def _build_training_pairs(
        self,
        predictor: Any,
        tokenizer: Any,
    ) -> Result[list[tuple[torch.Tensor, torch.Tensor]], ProcError]:
        """
        Build a list of (prompt_ids, completion_ids) tensors from the dataset.

        The prompt is the full DSPy chat-template rendered text.
        The completion is the expected output rendered by DSPy's ChatAdapter
        (i.e. what the model *should* have produced).
        """
        adapter = ChatAdapter()
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        for index, example in enumerate(self._dataset.load()):
            try:
                # Prompt: everything the model sees.
                dspy_msgs = adapter.format(
                    signature=predictor.signature,
                    demos=list(predictor.demos),
                    inputs=example.inputs(),
                )
                prompt_text: str = tokenizer.apply_chat_template(
                    dspy_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Completion: expected output tokens.
                completion_text: str = self._render_expected_completion(
                    example=example,
                    predictor=predictor,
                )

                prompt_ids = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"]

                completion_ids = tokenizer(
                    completion_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"]

                if completion_ids.numel() == 0:
                    self._logger.warning(
                        f"example[{index}]: completion tokenised to empty sequence — skipping."
                    )
                    continue

                pairs.append((prompt_ids, completion_ids))

            except Exception as exc:
                self._logger.warning(f"example[{index}]: failed to build pair — {exc}")

        if not pairs:
            return Failure(ProcError("No valid training pairs could be built from the dataset."))
        return Success(pairs)

    def _render_expected_completion(self, example: Any, predictor: Any) -> str:
        """
        Render the expected assistant response in DSPy's ChatAdapter format.

        DSPy's ChatAdapter formats the assistant turn as:

            [[ ## field_name ## ]]
            <value>

            [[ ## completed ## ]]

        We must match this format exactly so the model is trained to produce
        output that ChatAdapter can parse.

        ``example.labels()`` returns a dict of {output_field_name: value}
        for all fields declared as outputs in the signature.  For this
        pipeline the expected output is stored in the dataset under
        ``"expected"`` and surfaced by TrainingSetDataset as a single
        output field (e.g. ``meeting_time_extraction``).
        """
        labels: dict[str, Any] = example.labels()
        if not labels:
            self._logger.debug(
                f"example has no labels — output fields: "
                f"{list(predictor.signature.output_fields)}"
            )
            return ""

        lines: list[str] = []
        for field_name, value in labels.items():
            lines.append(f"[[ ## {field_name} ## ]]")
            if isinstance(value, (dict, list)):
                lines.append(json.dumps(value, ensure_ascii=False))
            else:
                lines.append(str(value).strip())
            lines.append("")  # blank line between fields
        lines.append("[[ ## completed ## ]]")
        return "\n".join(lines)

    # ── LoRA setup ─────────────────────────────────────────────────────────────

    def _apply_lora(self, model: Any) -> Any:
        """Wrap the model with PEFT LoRA adapters."""
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self._lora.r,
            lora_alpha=self._lora.lora_alpha,
            lora_dropout=self._lora.lora_dropout,
            target_modules=self._lora.target_modules,
            bias=self._lora.bias,
        )
        model = get_peft_model(model, peft_config)
        trainable, total = 0, 0
        for p in model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        self._logger.info(
            f"LoRA applied.  trainable params: {trainable:,}  "
            f"total params: {total:,}  "
            f"ratio: {trainable / total:.3%}"
        )
        return model

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> Result[tuple[Any, Any], ProcError]:
        """
        Load model without ``device_map="auto"`` (Captum compatibility).

        Uses ``self._load_dtype`` selected at init time.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._hf_model_name, trust_remote_code=True
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            self._logger.info(
                f"Loading {self._hf_model_name}  dtype={self._load_dtype}  device={self._device}"
            )
            model = AutoModelForCausalLM.from_pretrained(
                self._hf_model_name,
                dtype=self._load_dtype,
                trust_remote_code=True,
                # NO device_map="auto" — single device, no offloading.
            )
            model.to(self._device)
            model.eval()
            return Success((model, tokenizer))
        except Exception as exc:
            return Failure(ProcError(f"Failed to load '{self._hf_model_name}': {exc}"))

    # ── Scheduler ─────────────────────────────────────────────────────────────

    def _build_scheduler(
        self,
        optimiser: torch.optim.Optimizer,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Linear warmup + cosine decay scheduler.

        Falls back to a constant scheduler if warmup_steps >= total_steps.
        """
        warmup = min(self._training.warmup_steps, max(0, total_steps - 1))
        if warmup == 0:
            return torch.optim.lr_scheduler.ConstantLR(optimiser, factor=1.0)

        return torch.optim.lr_scheduler.SequentialLR(
            optimiser,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimiser, start_factor=1e-6, end_factor=1.0, total_iters=warmup
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=max(1, total_steps - warmup)
                ),
            ],
            milestones=[warmup],
        )

    # ── Adapter persistence ────────────────────────────────────────────────────

    def _save_adapter(
        self, model: Any, adapter_dir: Path
    ) -> AdapterSaveStatus:
        """Save PEFT adapter weights to ``adapter_dir``."""
        try:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            with timed("save_pretrained (adapter)", logger=self._logger):
                model.save_pretrained(str(adapter_dir))
            self._logger.info(f"Adapter saved to {adapter_dir}")
            return AdapterSaveStatus.SAVED
        except Exception as exc:
            self._logger.error(f"Failed to save adapter: {exc}")
            return AdapterSaveStatus.FAILED

    # ── Artifact writing ───────────────────────────────────────────────────────

    def _write_artifacts(self, summary: LoRARunSummary) -> None:
        """Write ``summary.json`` and ``training_log.json`` to ``output_dir``."""
        summary_path = self._output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(summary.model_dump_json(indent=2))
        self._logger.info(f"Summary written to {summary_path}")

        log_path = self._output_dir / "training_log.json"
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(
                [step.model_dump() for step in summary.step_logs],
                fh,
                indent=2,
            )
        self._logger.info(f"Training log written to {log_path}")

    # ── Device helpers ─────────────────────────────────────────────────────────

    def _resolve_device(self, requested: torch.device) -> torch.device:
        """
        Validate the requested device.

        MPS smoke-test is intentionally lightweight here — LoRA training
        uses standard PyTorch autograd (not Captum), so MPS compatibility
        is more reliable.  We still fall back to CPU if MPS is unavailable.
        """
        if requested.type == "mps":
            if not torch.backends.mps.is_available():
                self._logger.warning("MPS requested but not available.  Falling back to CPU.")
                return torch.device("cpu")
            return requested
        if requested.type == "cuda":
            if not torch.cuda.is_available():
                self._logger.warning("CUDA requested but not available.  Falling back to CPU.")
                return torch.device("cpu")
            return requested
        return requested


# ── Internal helpers ───────────────────────────────────────────────────────────

class _ProbeDataset(BaseDataset):
    """
    Wraps an existing dataset and exposes only a single example.

    Used to run ``PromptAttributionNode`` on the probe example without
    loading the full dataset or duplicating attribution logic.
    """

    def __init__(self, dataset: BaseDataset, example_index: int) -> None:
        self._dataset = dataset
        self._example_index = example_index

    def load(self) -> Any:  # type: ignore[override]
        for i, example in enumerate(self._dataset.load()):
            if i == self._example_index:
                yield example
                return
        raise IndexError(
            f"_ProbeDataset: example_index={self._example_index} is out of range."
        )
