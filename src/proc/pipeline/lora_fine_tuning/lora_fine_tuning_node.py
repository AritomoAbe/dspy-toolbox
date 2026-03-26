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
  2. Evaluates the raw base model on the dataset.
  3. Applies PEFT LoRA adapters to the model.
  4. Trains for ``n_epochs`` over the dataset.
  5. Evaluates after each epoch using the scorer.
  6. Saves the adapter weights and a structured run report to ``output_dir``.
  7. Optionally runs ``PromptAttributionNode`` on a held-out probe example
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

import base64
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional, TypeAlias

import torch
from dspy.adapters import ChatAdapter
from peft import LoraConfig, TaskType, get_peft_model
from returns.pipeline import is_successful
from returns.result import Failure, Result, Success
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.base.timing import timed
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.llm_prompt_usage_attribution.prompt_attribution_node import (
    PromptAttributionNode,
)
from proc.pipeline.lora_fine_tuning.models import (
    AdapterSaveStatus,
    AttributionComparisonConfig,
    EpochMetrics,
    FAILURE_SCORE_THRESHOLD,
    LoRAHyperParams,
    LoRARunSummary,
    STEP_LOG_PREFIX,
    StepLog,
)
from proc.pipeline.lora_fine_tuning.models import (
    TrainingHyperParams,
    TrainingLogArtifact,
    TrainingPhase,
)
from proc.pipeline.output_result_auditor.score_extractor import (
    INVALID_SCORE,
    ScoreExtractor,
)

_DEFAULT_HF_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
_DEFAULT_OUTPUT_DIR: str = "runs/lora_finetuning"
_DEFAULT_ATTR_DEVICE: str = "cpu"
_DEFAULT_CPU_DEVICE: str = "cpu"
_DEFAULT_CUDA_DEVICE: str = "cuda"
_DEFAULT_MPS_DEVICE: str = "mps"
_DEFAULT_SEED: int = 42
_IGNORE_INDEX: int = -100

_LABEL_LOAD_BASE_MODEL: str = "_load_base_model"
_LABEL_GET_PEFT_MODEL: str = "get_peft_model"
_LABEL_BUILD_TRAINING_PAIRS: str = "_build_training_pairs"
_LABEL_BASELINE_EVAL: str = "_evaluate epoch=0 (baseline)"
_LABEL_ATTRIBUTION_COMPARISON_DIR: str = "attribution_comparison"
_LABEL_ADAPTER_DIR: str = "adapter"
_LABEL_COMPARISON_HTML: str = "comparison.html"
_LABEL_PROMPT_HEATMAP: str = "prompt_heatmap.png"
_LABEL_SUMMARY_JSON: str = "summary.json"
_LABEL_TRAINING_LOG_JSON: str = "training_log.json"
_LABEL_UTF8: str = "utf-8"
_LABEL_ASCII: str = "ascii"
_LABEL_FORWARD_PASS: str = "_train_step forward"
_LABEL_GENERATE: str = "_evaluate generate"
_LABEL_ATTRIBUTION_SNAPSHOT: str = "attribution_snapshot phase="
_LABEL_COMPLETED_MARKER: str = "[[ ## completed ## ]]"
_WARMUP_START_FACTOR: float = 1e-6
_RETURN_TENSORS_PT: str = "pt"
_KEY_INPUT_IDS: str = "input_ids"
_KEY_ATTENTION_MASK: str = "attention_mask"

_TrainingPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]
_TrainingPairs: TypeAlias = list[_TrainingPair]


def _read_png_b64(directory: Path, logger: logging.Logger) -> str:
    """Find the first prompt heatmap under *directory* and return it base64-encoded."""
    candidates = sorted(directory.glob(f"**/{_LABEL_PROMPT_HEATMAP}"))
    if not candidates:
        logger.warning(
            f"_write_comparison_html: no {_LABEL_PROMPT_HEATMAP} found under {directory}"
        )
        return ""
    return base64.b64encode(candidates[0].read_bytes()).decode(_LABEL_ASCII)


class _CompletedMarkerStopping(StoppingCriteria):
    def __init__(self, marker_ids: torch.Tensor) -> None:
        self._marker_ids = marker_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> bool:
        marker_len = self._marker_ids.shape[0]
        if input_ids.shape[1] < marker_len:
            return False
        tail = input_ids[0, -marker_len:]
        return bool((tail == self._marker_ids).all())


class LoRAFineTuningNode(ProcNode):
    """
    Stage 4 — LoRA fine-tuning + attribution comparison.
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
        attr_device: str = _DEFAULT_ATTR_DEVICE,
        force_dtype: Optional[torch.dtype] = None,
        attribution_comparison: Optional[AttributionComparisonConfig] = None,
        seed: int = _DEFAULT_SEED,
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
        self._seed = seed

        self._device = self._resolve_device(torch.device(attr_device))
        if force_dtype is None:
            self._load_dtype: torch.dtype = (
                torch.float32 if self._device.type == _DEFAULT_CPU_DEVICE else torch.float16
            )
        else:
            self._load_dtype = force_dtype

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(
            f"LoRAFineTuningNode: model={hf_model_name}  "
            f"device={self._device}  dtype={self._load_dtype}"
        )

    def invoke(self) -> Result[ProcScore, ProcError]:
        self._set_seed(self._seed)

        predictors = self._llm.predictors()
        if not predictors:
            return Failure(ProcError(
                "No predictors found in LLM — cannot render compiled prompt structure."
            ))
        predictor = predictors[0]

        with timed(_LABEL_LOAD_BASE_MODEL, logger=self._logger):
            load_result = self._load_model()
        if not is_successful(load_result):
            return Failure(load_result.failure())
        model, tokenizer = load_result.unwrap()

        with timed(_LABEL_BUILD_TRAINING_PAIRS, logger=self._logger):
            pairs_result = self._build_training_pairs(predictor, tokenizer)
        if not is_successful(pairs_result):
            return Failure(pairs_result.failure())
        training_pairs = pairs_result.unwrap()

        if not training_pairs:
            return Failure(ProcError("Dataset produced zero training pairs."))
        self._logger.info(f"Built {len(training_pairs)} training pairs.")

        if self._attr_cfg.enabled:
            before_dir = (
                self._output_dir / _LABEL_ATTRIBUTION_COMPARISON_DIR / TrainingPhase.BEFORE
            )
            attr_result = self._run_attribution_snapshot(
                phase=TrainingPhase.BEFORE,
                output_dir=before_dir,
            )
            if not is_successful(attr_result):
                self._logger.warning(
                    f"Pre-training attribution failed: {attr_result.failure()}.  Continuing."
                )

        with timed(_LABEL_BASELINE_EVAL, logger=self._logger):
            baseline_metrics = self._evaluate(
                epoch=0,
                predictor=predictor,
                model=model,
                tokenizer=tokenizer,
            )
        self._logger.info(
            f"Baseline pass rate: {baseline_metrics.pass_rate:.1%}  "
            f"({baseline_metrics.n_valid}/{baseline_metrics.n_examples})"
        )

        with timed(_LABEL_GET_PEFT_MODEL, logger=self._logger):
            model = self._apply_lora(model)

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

            model.eval()
            with timed(f"_evaluate epoch={epoch}", logger=self._logger):
                metrics = self._evaluate(
                    epoch=epoch,
                    predictor=predictor,
                    model=model,
                    tokenizer=tokenizer,
                )
            epoch_metrics.append(metrics)
            self._logger.info(
                f"Epoch {epoch} pass rate: {metrics.pass_rate:.1%}  "
                f"({metrics.n_valid}/{metrics.n_examples})"
            )

        adapter_dir = self._output_dir / _LABEL_ADAPTER_DIR
        save_status = self._save_adapter(model=model, adapter_dir=adapter_dir)

        if self._attr_cfg.enabled:
            after_dir = (
                self._output_dir / _LABEL_ATTRIBUTION_COMPARISON_DIR / TrainingPhase.AFTER
            )
            attr_result = self._run_attribution_snapshot(
                phase=TrainingPhase.AFTER,
                output_dir=after_dir,
                adapter_dir=adapter_dir if save_status == AdapterSaveStatus.SAVED else None,
            )
            if is_successful(attr_result):
                self._write_comparison_html(
                    before_dir=(
                        self._output_dir / _LABEL_ATTRIBUTION_COMPARISON_DIR / TrainingPhase.BEFORE
                    ),
                    after_dir=after_dir,
                )
            else:
                self._logger.warning(
                    f"Post-training attribution failed: {attr_result.failure()}.  Continuing."
                )

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
            seed=self._seed,
        )
        self._write_artifacts(summary)

        self._logger.info(
            f"LoRA fine-tuning complete.  "
            f"pass_rate: {initial_pass_rate:.1%} → {final_pass_rate:.1%}  "
            f"(Δ{summary.pass_rate_delta:+.1%})"
        )
        return Success(ProcScore(value=self._SCORE, context=summary))

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
        ignore = torch.full_like(prompt_ids, fill_value=_IGNORE_INDEX)
        labels = torch.cat([ignore, completion_ids], dim=1)

        optimiser.zero_grad(set_to_none=True)
        with timed(_LABEL_FORWARD_PASS, logger=self._logger):
            outputs = model(input_ids=input_ids, labels=labels)
        loss: torch.Tensor = outputs.loss
        loss.backward()  # type: ignore[no-untyped-call]

        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=self._training.max_grad_norm,
        )
        optimiser.step()
        scheduler.step()

        return float(loss.detach().cpu().item())

    def _evaluate(
        self,
        epoch: int,
        predictor: Any,
        model: Any,
        tokenizer: Any,
    ) -> EpochMetrics:
        """
        Run the scorer over the full dataset and return pass/fail counts.
        """
        adapter = ChatAdapter()
        n_examples = 0
        n_valid = 0
        n_invalid = 0

        model.eval()
        for example in self._dataset.load():
            n_examples += 1
            try:
                passed = self._evaluate_example(
                    epoch, n_examples - 1, example, adapter, predictor, model, tokenizer,
                )
            except Exception as exc:
                self._logger.warning(f"_evaluate[epoch={epoch}]: example failed — {exc}")
                n_invalid += 1
                continue
            if passed:
                n_valid += 1
            else:
                n_invalid += 1

        pass_rate = n_valid / n_examples if n_examples > 0 else float()
        return EpochMetrics(
            epoch=epoch,
            n_examples=n_examples,
            n_valid=n_valid,
            n_invalid=n_invalid,
            pass_rate=pass_rate,
        )

    def _evaluate_example(
        self,
        epoch: int,
        example_idx: int,
        example: Any,
        adapter: ChatAdapter,
        predictor: Any,
        model: Any,
        tokenizer: Any,
    ) -> bool:
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
        prompt_encoding = tokenizer(
            prompt_text,
            return_tensors=_RETURN_TENSORS_PT,
            add_special_tokens=False,
        )
        prompt_ids = prompt_encoding[_KEY_INPUT_IDS].to(self._device)
        attention_mask = prompt_encoding[_KEY_ATTENTION_MASK].to(self._device)

        with torch.no_grad():
            with timed(
                f"{_LABEL_GENERATE}[epoch={epoch} example={example_idx}]",
                logger=self._logger,
            ):
                marker_ids = tokenizer(
                    _LABEL_COMPLETED_MARKER,
                    return_tensors=_RETURN_TENSORS_PT,
                    add_special_tokens=False,
                )[_KEY_INPUT_IDS][0].to(self._device)
                output_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self._training.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([_CompletedMarkerStopping(marker_ids)]),
                )
                generated_ids = output_ids[0, prompt_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred = adapter.parse(
            signature=predictor.signature,
            completion=generated_text,
        )
        score = self._scorer.extraction_metric(example, pred)
        return not (score == INVALID_SCORE or score < FAILURE_SCORE_THRESHOLD)

    def _run_attribution_snapshot(
        self,
        phase: TrainingPhase,
        output_dir: Path,
        adapter_dir: Optional[Path] = None,
    ) -> Result[None, ProcError]:
        """
        Run ``PromptAttributionNode`` on the probe example and save to
        ``output_dir``.  When *adapter_dir* is provided (AFTER phase), the node
        loads the saved LoRA adapter via ``peft_model_id`` so the comparison
        reflects the fine-tuned weights rather than the bare base model.
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
            peft_model_id=None if adapter_dir is None else str(adapter_dir),
        )

        with timed(f"{_LABEL_ATTRIBUTION_SNAPSHOT}{phase}", logger=self._logger):
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
        """
        comparison_dir = self._output_dir / _LABEL_ATTRIBUTION_COMPARISON_DIR
        comparison_dir.mkdir(parents=True, exist_ok=True)
        out_path = comparison_dir / _LABEL_COMPARISON_HTML

        before_b64 = _read_png_b64(before_dir, self._logger)
        after_b64 = _read_png_b64(after_dir, self._logger)

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

        out_path.write_text(body, encoding=_LABEL_UTF8)
        self._logger.info(f"Attribution comparison report: {out_path}")

    def _build_training_pairs(
        self,
        predictor: Any,
        tokenizer: Any,
    ) -> Result[_TrainingPairs, ProcError]:
        """
        Build a list of (prompt_ids, completion_ids) tensors from the dataset.
        """
        adapter = ChatAdapter()
        pairs: _TrainingPairs = []

        for index, example in enumerate(self._dataset.load()):
            try:
                pair = self._build_single_pair(index, example, adapter, predictor, tokenizer)
            except Exception as exc:
                self._logger.warning(f"example[{index}]: failed to build pair — {exc}")
                continue
            if pair is not None:
                pairs.append(pair)

        if pairs:
            return Success(pairs)
        return Failure(ProcError("No valid training pairs could be built from the dataset."))

    def _build_single_pair(
        self,
        index: int,
        example: Any,
        adapter: ChatAdapter,
        predictor: Any,
        tokenizer: Any,
    ) -> Optional[_TrainingPair]:
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
        completion_text: str = self._render_expected_completion(
            example=example,
            predictor=predictor,
        )
        prompt_ids = tokenizer(
            prompt_text,
            return_tensors=_RETURN_TENSORS_PT,
            add_special_tokens=False,
        )[_KEY_INPUT_IDS]
        completion_ids = tokenizer(
            completion_text,
            return_tensors=_RETURN_TENSORS_PT,
            add_special_tokens=False,
        )[_KEY_INPUT_IDS]
        if completion_ids.numel() == 0:
            self._logger.warning(
                f"example[{index}]: completion tokenised to empty sequence — skipping."
            )
            return None
        return prompt_ids, completion_ids

    def _render_expected_completion(self, example: Any, predictor: Any) -> str:
        """
        Render the expected assistant response in DSPy's ChatAdapter format.
        """
        labels = example.labels()
        if not labels:
            self._logger.debug(
                f"example has no labels — output fields: "
                f"{list(predictor.signature.output_fields)}"
            )
            return ""

        lines: list[str] = []
        output_field_names = list(predictor.signature.output_fields.keys())

        for field_name in output_field_names:
            if field_name not in labels:
                continue
            value = labels[field_name]
            lines.append(f"[[ ## {field_name} ## ]]")
            if isinstance(value, (dict, list)):
                lines.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
            else:
                lines.append(str(value).strip())
            lines.append("")
        lines.append(_LABEL_COMPLETED_MARKER)
        return "\n".join(lines)

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

    def _load_model(self) -> Result[tuple[Any, Any], ProcError]:
        """
        Load model without ``device_map="auto"`` (Captum compatibility).
        """
        try:
            return Success(self._load_model_impl())
        except Exception as exc:
            return Failure(ProcError(f"Failed to load '{self._hf_model_name}': {exc}"))

    def _load_model_impl(self) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(
            self._hf_model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self._logger.info(
            f"Loading {self._hf_model_name}  dtype={self._load_dtype}  device={self._device}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self._hf_model_name,
            dtype=self._load_dtype,  # 'torch_dtype' is deprecated in this transformers version
            trust_remote_code=True,
        )
        model.to(self._device)  # type: ignore[arg-type]
        model.eval()  # type: ignore[no-untyped-call]
        return model, tokenizer

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
            return torch.optim.lr_scheduler.ConstantLR(
                optimiser,
                factor=1.0,
                total_iters=1,
            )

        return torch.optim.lr_scheduler.SequentialLR(
            optimiser,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimiser,
                    start_factor=_WARMUP_START_FACTOR,
                    end_factor=1.0,
                    total_iters=warmup,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser,
                    T_max=max(1, total_steps - warmup),
                ),
            ],
            milestones=[warmup],
        )

    def _save_adapter(
        self,
        model: Any,
        adapter_dir: Path,
    ) -> AdapterSaveStatus:
        """Save PEFT adapter weights to ``adapter_dir``."""
        try:
            return self._do_save_adapter(model, adapter_dir)
        except Exception as exc:
            self._logger.error(f"Failed to save adapter: {exc}")
            return AdapterSaveStatus.FAILED

    def _do_save_adapter(self, model: Any, adapter_dir: Path) -> AdapterSaveStatus:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        with timed("save_pretrained (adapter)", logger=self._logger):
            model.save_pretrained(str(adapter_dir))
        self._logger.info(f"Adapter saved to {adapter_dir}")
        return AdapterSaveStatus.SAVED

    def _write_artifacts(self, summary: LoRARunSummary) -> None:
        """Write ``summary.json`` and ``training_log.json`` to ``output_dir``."""
        summary_path = self._output_dir / _LABEL_SUMMARY_JSON
        with open(summary_path, "w", encoding=_LABEL_UTF8) as fh:
            fh.write(summary.model_dump_json(indent=2))
        self._logger.info(f"Summary written to {summary_path}")

        log_path = self._output_dir / _LABEL_TRAINING_LOG_JSON
        training_log = TrainingLogArtifact(
            epoch_metrics=summary.epoch_metrics,
            step_logs=summary.step_logs,
        )
        with open(log_path, "w", encoding=_LABEL_UTF8) as fh:
            fh.write(training_log.model_dump_json(indent=2))
        self._logger.info(f"Training log written to {log_path}")

    def _resolve_device(self, requested: torch.device) -> torch.device:
        """
        Validate the requested device.
        """
        if requested.type == _DEFAULT_MPS_DEVICE:
            if not torch.backends.mps.is_available():
                self._logger.warning(
                    "MPS requested but not available.  Falling back to CPU."
                )
                return torch.device(_DEFAULT_CPU_DEVICE)
            return requested
        if requested.type == _DEFAULT_CUDA_DEVICE:
            if not torch.cuda.is_available():
                self._logger.warning(
                    "CUDA requested but not available.  Falling back to CPU."
                )
                return torch.device(_DEFAULT_CPU_DEVICE)
            return requested
        return requested

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class _ProbeDataset(BaseDataset):
    """
    Wraps an existing dataset and exposes only a single example.

    Used to run ``PromptAttributionNode`` on the probe example without
    loading the full dataset or duplicating attribution logic.
    """

    def __init__(self, dataset: BaseDataset, example_index: int) -> None:
        self._dataset = dataset
        self._example_index = example_index

    def load(self) -> list[Any]:
        for i, example in enumerate(self._dataset.load()):
            if i == self._example_index:
                return [example]
        raise IndexError(
            f"_ProbeDataset: example_index={self._example_index} is out of range."
        )
