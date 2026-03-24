from __future__ import annotations

import html
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from dspy.adapters import ChatAdapter
from returns.result import Failure, Result, Success
from transformers import AutoModelForCausalLM, AutoTokenizer

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.timing import timed
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor

_HF_DEFAULT_MODEL: str = 'Qwen/Qwen2.5-1.5B-Instruct'
_HF_DEFAULT_INTERNAL_BATCH: int = 1
_HF_DEFAULT_CHART_SCORE_THRESHOLD: float = 0.0


class HFAttributionBase(ProcNode):
    """
    Shared base for HuggingFace-backed Captum attribution nodes.

    Provides:
      - Common constructor parameters (dataset, llm, scorer, hf_model_name,
        internal_batch_size, attr_device, force_dtype)
      - Timestamped output directory creation (output_dir/<YYYY-MM-DDTHH-MM-SS>/)
      - Model loading with automatic dtype selection, pad_token_id fix, and timing
      - MPS device validation with autograd smoke-test
      - Predictor and prompt-text helpers for use in invoke()
      - Chart threshold filtering and HTML token span helpers shared by all
        subclass report generators

    Device / dtype strategy
    -----------------------
    Do NOT use device_map="auto": accelerate's AlignDevicesHook breaks Captum's
    interpolation loop. The model is always loaded onto a single device.

    dtype auto-selection:
        CPU      → float32 (stable gradients; no speed benefit from float16 on CPU)
        MPS/CUDA → float16 (hardware-accelerated, 2× memory saving)
        override → pass force_dtype to fix the dtype on any device
    """

    def __init__(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
        hf_model_name: str = _HF_DEFAULT_MODEL,
        internal_batch_size: int = _HF_DEFAULT_INTERNAL_BATCH,
        attr_device: str = 'cpu',
        force_dtype: torch.dtype | None = None,
        output_dir: str | Path = 'runs/attribution',
        save_html: bool = True,
        save_plots: bool = True,
        chart_score_threshold: float = _HF_DEFAULT_CHART_SCORE_THRESHOLD,
    ) -> None:
        self._logger = logging.getLogger(type(self).__module__)
        self._dataset = dataset
        self._llm = llm
        self._scorer = scorer
        self._hf_model_name = hf_model_name
        self._internal_batch_size = internal_batch_size
        self._device = self._validate_device(torch.device(attr_device))
        self._force_dtype = force_dtype
        if force_dtype is not None:
            self._load_dtype: torch.dtype = force_dtype
            self._logger.info(
                'dtype override: force_dtype=%s (auto would have chosen %s)',
                force_dtype,
                'float32' if self._device.type == 'cpu' else 'float16',
            )
        elif self._device.type == 'cpu':
            self._load_dtype = torch.float32
            self._logger.info(
                'dtype: float32 (CPU — no hardware fp16 units; '
                'float32 gives better gradient stability with no speed penalty)',
            )
        else:
            self._load_dtype = torch.float16
            self._logger.info('dtype: float16 (%s — hardware-accelerated fp16)', self._device.type)

        self._save_html = save_html
        self._save_plots = save_plots
        self._chart_score_threshold = chart_score_threshold
        run_ts = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        self._output_dir = Path(output_dir) / run_ts
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info('Output directory: %s', self._output_dir)

    def _load_model(self) -> Result[tuple[Any, Any], ProcError]:
        """
        Load the HuggingFace tokenizer and model without device_map="auto".

        The model is always loaded onto CPU first; the caller must call
        model.to(device) after this method returns. Calls model.eval() before
        returning. Sets pad_token_id from eos_token_id if missing.
        """
        try:
            with timed('_load_model tokenizer', logger=self._logger):
                tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name, trust_remote_code=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            with timed(f'_load_model {self._hf_model_name} ({self._load_dtype})', logger=self._logger):
                model = AutoModelForCausalLM.from_pretrained(
                    self._hf_model_name,
                    dtype=self._load_dtype,
                    trust_remote_code=True,
                )
            model.eval()
            return Success((model, tokenizer))
        except Exception as e:
            return Failure(ProcError(f"Failed to load '{self._hf_model_name}': {e}"))

    def _validate_device(self, requested: torch.device) -> torch.device:
        """
        Validate the requested device for Captum gradient computation.

        MPS (Apple Silicon GPU) gives 5-10× speedup for float16 forward passes.
        Performs an autograd smoke-test at init time and silently falls back to
        CPU if MPS is unavailable or autograd is broken in this PyTorch build.
        """
        if requested.type != 'mps':
            self._logger.info('Attribution device: %s', requested)
            return requested
        if not torch.backends.mps.is_available():
            self._logger.warning('MPS requested but not available. Falling back to CPU.')
            return torch.device('cpu')
        try:
            t = torch.ones((2, 2), device='mps', requires_grad=True)
            (t * t).sum().backward()
            del t
            torch.mps.empty_cache()
            self._logger.info(
                'MPS autograd validated — using MPS for attribution. '
                'Expected speedup: 5-10× over CPU for float16 forward passes.',
            )
            return requested
        except Exception as e:
            self._logger.warning('MPS autograd validation failed (%s). Falling back to CPU.', e)
            return torch.device('cpu')

    def _get_predictor(self) -> Result[Any, ProcError]:
        """Return the first DSPy predictor or a Failure if the LLM has no compiled predictors."""
        predictors = self._llm.predictors()
        if not predictors:
            return Failure(ProcError('No predictors found in LLM — cannot render compiled prompt.'))
        return Success(predictors[0])

    def _render_prompt(
        self,
        adapter: ChatAdapter,
        predictor: Any,
        example: Any,
        tokenizer: Any,
    ) -> str:
        """Render the compiled DSPy prompt as a chat-template string."""
        dspy_msgs = adapter.format(
            signature=predictor.signature,
            demos=list(predictor.demos),
            inputs=example.inputs(),
        )
        return tokenizer.apply_chat_template(  # type: ignore[no-any-return]
            dspy_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _apply_chart_threshold(self, scores: torch.Tensor) -> list[int]:
        """Return token indices whose normalized absolute score meets the threshold."""
        max_abs = float(scores.abs().max().item()) if scores.numel() else 1.0
        if self._chart_score_threshold <= 0.0 or max_abs == 0.0:
            return list(range(scores.shape[0]))
        return [
            i for i in range(scores.shape[0])
            if float(scores[i].abs().item()) / max_abs >= self._chart_score_threshold
        ]

    def _make_html_span(self, token_display: str, score: float, max_abs: float, region: str) -> str:
        """Return a color-coded HTML span for one token."""
        disp = html.escape(token_display)
        strength = min(1.0, abs(score) / max_abs) if max_abs > 0 else 0.0
        if score >= 0:
            color = f'rgba(0, 128, 0, {0.12 + 0.58 * strength:.3f})'
        else:
            color = f'rgba(200, 0, 0, {0.12 + 0.58 * strength:.3f})'
        return f'<span class="token" title="region={region} score={score:+.6f}" style="background:{color}">{disp}</span>'