"""
LIG Attribution Auditor  —  Stage 3
====================================

Panels
------
  Panel A — Token Saliency     (Captum LayerIntegratedGradients → per-token L2 norm)
  Panel B — Logit Lens         (forward hooks per decoder layer → target rank trajectory)
  Panel C — Segment Attribution (token saliency aggregated per DSPy prompt segment)

Device strategy (matches reference script)
------------------------------------------
  inference_device:  MPS > CUDA > CPU  (logit lens forward pass, fast)
  attr_device:       CPU               (Captum gradient passes, always CPU)

  CRITICAL: do NOT use device_map="auto". accelerate's AlignDevicesHook
  breaks Captum's interpolation loop and causes OOM on MPS. Load the model
  onto a single device with no offloading.
"""

import logging
import re
import textwrap
import time
from collections import defaultdict
from typing import Any

import torch
from captum.attr import LayerIntegratedGradients, IntegratedGradients
from dspy.adapters import ChatAdapter
from returns.pipeline import is_successful
from returns.result import Result, Success, Failure
from transformers import AutoModelForCausalLM, AutoTokenizer

import dspy

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.llm_prompt_usage_attribution.contexts import LIGAttributionContext
from proc.pipeline.llm_prompt_usage_attribution.models import LIGExampleResult, TokenSaliency, LayerProbe, \
    SegmentSaliency, LIGAttributionResult
from proc.pipeline.output_result_auditor.score_extractor import ScoreExtractor, INVALID_SCORE
from proc.base.timing import timed

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"

# 50 for exploration, 200 for final analysis (matches reference)
_DEFAULT_IG_STEPS: int = 200

# internal_batch_size=1 → minimum memory; increase if you have headroom
_DEFAULT_INTERNAL_BATCH: int = 1

# Top-N global tokens to surface in the aggregate result
_TOP_TOKEN_COUNT: int = 20

# Regex to split the rendered prompt into logical segments (matches reference Panel C)
_SEGMENT_SPLIT_PATTERN: str = r"(<\|im_start\|>|<\|im_end\|>|\[\[.*?\]\]|\n{2,})"

class LIGAttributionAuditor(ProcNode):
    """
    Stage 3 — Score How the LLM Uses the Prompt (Layer Integrated Gradients).

    For each example in the dataset:
      1. Render the compiled DSPy prompt via ChatAdapter (exact prompt the model sees)
      2. Run a fast logit-lens pass on inference_device to track per-layer rank
         of the prediction target  →  Panel B
      3. Move model to CPU and run Captum LayerIntegratedGradients to get
         per-token embedding attribution  →  Panel A
      4. Map token attributions back onto DSPy prompt segments  →  Panel C

    Parameters
    ----------
    dataset:          BaseDataset to evaluate.
    llm:              DSpyLLM — used for prediction and to access the compiled
                      predictor's signature and demos.
    scorer:           ScoreExtractor — validates each prediction.
    hf_model_name:    HuggingFace CausalLM to use as local attribution proxy.
                      Default: 'Qwen/Qwen2.5-1.5B-Instruct' (~3 GB, fits on CPU).
                      For higher accuracy use 'Qwen/Qwen2.5-7B-Instruct' with
                      attr_device='cpu' and ig_steps=200.
    ig_steps:         Captum integration steps. 50 = fast exploration,
                      200 = reliable final analysis.
    internal_batch_size: Captum internal batch size. 1 = minimum memory.
    attr_device:      Device for gradient passes. Always 'cpu' by default —
                      avoids MPS OOM and accelerate hook interference.
    """

    _SCORE: float = 1.0

    # Minimum recommended ig_steps per decoder layer.
    # Empirically: < 5 steps/layer → high convergence delta → unreliable attributions.
    _MIN_IG_STEPS_PER_LAYER: int = 10

    def __init__(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
        hf_model_name: str = _DEFAULT_MODEL,
        ig_steps: int = _DEFAULT_IG_STEPS,
        internal_batch_size: int = _DEFAULT_INTERNAL_BATCH,
        attr_device: str = "cpu",
        target_mode: str = "content",
    ) -> None:
        """
        Parameters
        ----------
        target_mode : str
            Controls which token is used as the LIG attribution target.

            "content" (Option 1, default) — skip format/whitespace tokens and
            target the first semantically meaningful token the model generates
            (e.g. '{' for JSON, a letter for structured text). This answers:
            "what prompt sections drive the model to produce content?"

            "top1" (Option 2) — use the model's raw top-1 predicted next token,
            including format tokens like '[['. This answers: "what drove the model
            to start responding in this format?" Useful for diagnosing whether the
            model is format-driven vs content-driven.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Using HuggingFace model: {hf_model_name}")

        if target_mode not in ("content", "top1"):
            raise ValueError(f"target_mode must be 'content' or 'top1', got {target_mode!r}")
        self._target_mode = target_mode

        self._dataset = dataset
        self._llm = llm
        self._scorer = scorer
        self._hf_model_name = hf_model_name
        self._ig_steps = ig_steps
        self._internal_batch_size = internal_batch_size

        # Device selection with MPS autograd validation.
        # MPS (Apple Silicon GPU) gives 5-10x speedup for float16 forward passes.
        # _validate_device tests MPS autograd at init time and falls back to CPU
        # silently if MPS is unavailable or autograd is broken in this PyTorch build.
        requested_device = torch.device(attr_device)
        self._device = self._validate_device(requested_device)
        self._inference_device = self._device  # kept for compatibility
        self._attr_device = self._device        # kept for compatibility
        # _model_dtype is set in invoke() after the model is loaded.
        # Storing it once prevents bugs where _logit_lens temporarily
        # mutates lm_head dtype and next(model.parameters()) returns float32.
        self._model_dtype: torch.dtype = torch.float16  # overwritten in invoke()

    # ── ProcNode entry point ───────────────────────────────────────────────────

    # ── Device validation ─────────────────────────────────────────────────────

    def _validate_device(self, requested: torch.device) -> torch.device:
        """
        Validate the requested device for Captum gradient computation.

        MPS (Apple Silicon GPU) gives 5-10x speedup for float16 forward passes.
        We smoke-test MPS autograd at init time and fall back to CPU silently if
        it fails — avoids surprises after waiting minutes for compilation.
        """
        if requested.type != "mps":
            self._logger.info(f"Attribution device: {requested}")
            return requested

        if not torch.backends.mps.is_available():
            self._logger.warning("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")

        try:
            _t = torch.ones(4, 4, requires_grad=True, device="mps")
            _loss = (_t * _t).sum()
            _loss.backward()
            del _t, _loss
            torch.mps.empty_cache()
            self._logger.info(
                "MPS autograd validated — using MPS for attribution. "
                "Expected speedup: 5-10x over CPU for float16 forward passes."
            )
            return requested
        except Exception as e:
            self._logger.warning(
                f"MPS autograd validation failed ({e}). Falling back to CPU."
            )
            return torch.device("cpu")

    def invoke(self) -> Result[ProcScore, ProcError]:
        predictors = self._llm.predictors()
        if not predictors:
            return Failure(ProcError(
                "No predictors found in LLM — cannot access compiled prompt structure"
            ))

        predictor = predictors[0]

        # Load HuggingFace proxy model ─────────────────────────────────────────
        self._logger.info(f"Loading HuggingFace proxy: {self._hf_model_name}")
        self._logger.info(
            f"inference_device={self._inference_device}  "
            f"attr_device={self._attr_device}"
        )

        with timed("_load_model", logger=self._logger):
            load_result = self._load_model()
        if not is_successful(load_result):
            return load_result

        hf_model, tokenizer = load_result.unwrap()

        # Move model to target device immediately after loading.
        # HuggingFace always loads onto CPU; without this, tensors sent to MPS
        # mismatch the CPU model → "Placeholder storage has not been allocated
        # on MPS device!"
        self._logger.info(f"Moving model to {self._device}")
        hf_model.to(self._device)

        # Store model dtype ONCE here — before _logit_lens temporarily mutates
        # lm_head to float32 for the logit projection. If we query
        # next(model.parameters()).dtype later, we may get float32 by mistake.
        self._model_dtype = next(hf_model.parameters()).dtype
        self._logger.info(f"Model dtype: {self._model_dtype}")

        # MPS JIT warmup — triggers Metal kernel compilation before timed passes.
        # Without this, the first real forward pass triggers compilation and can
        # take 3-5 minutes. After warmup, all subsequent passes use cached kernels.
        if self._device.type == "mps":
            self._logger.info(
                "MPS warmup — compiling Metal kernels (one-time cost, ~30-60s)..."
            )
            _wt0 = time.perf_counter()
            with torch.no_grad():
                _dummy = torch.zeros((1, 16), dtype=torch.long, device=self._device)
                _ = hf_model(input_ids=_dummy)
            torch.mps.empty_cache()
            self._logger.info(
                f"MPS warmup complete in {time.perf_counter() - _wt0:.1f}s — "
                "subsequent forward passes will be fast."
            )

        # Locate embed_tokens layer ────────────────────────────────────────────
        embed_layer = self._get_embed_layer(hf_model)
        if embed_layer is None:
            return Failure(ProcError(
                f"Cannot find embed_tokens in {self._hf_model_name}. "
                "Supported: Qwen2, LLaMA, Mistral, Phi, GPT-2, Falcon."
            ))

        # Identify number of decoder layers for logit lens ────────────────────
        decoder_layers = self._get_decoder_layers(hf_model)
        if decoder_layers is None:
            return Failure(ProcError(
                f"Cannot find decoder layers in {self._hf_model_name}."
            ))

        # Validate ig_steps against model depth ───────────────────────────────
        n_layers = len(decoder_layers)
        recommended_ig_steps = n_layers * self._MIN_IG_STEPS_PER_LAYER
        if self._ig_steps < recommended_ig_steps:
            self._logger.warning(
                f"ig_steps={self._ig_steps} may be too low for a {n_layers}-layer model. "
                f"Recommended minimum: {recommended_ig_steps} "
                f"({self._MIN_IG_STEPS_PER_LAYER} steps × {n_layers} layers). "
                f"Low ig_steps → high convergence delta → unreliable Panel A/C scores. "
                f"Pass ig_steps={recommended_ig_steps} or higher for reliable results."
            )
        else:
            self._logger.info(
                f"ig_steps={self._ig_steps} sufficient for {n_layers}-layer model "
                f"(recommended minimum: {recommended_ig_steps})"
            )

        adapter = ChatAdapter()
        example_results: list[LIGExampleResult] = []

        for index, example in enumerate(self._dataset.load()):
            self._logger.info(f"Processing example[{index}]")
            _example_t0 = time.perf_counter()

            # Generate prediction via API LLM ──────────────────────────────────
            with timed(f"example[{index}] DSpy execution", logger=self._logger):
                with dspy.context(cache=False):
                    pred = self._llm(**example.inputs())
                    self._logger.info(f"Result for example[{index}] is {pred}")

            score = self._scorer.extraction_metric(example, pred)
            self._logger.info(f"Result score for example[{index}] is {score}")
            if score == INVALID_SCORE:
                return Failure(ProcError(f"Cannot score example[{index}]"))

            # Render the compiled DSPy prompt (exact chat-template text) ───────
            dspy_msgs = adapter.format(
                signature=predictor.signature,
                demos=list(predictor.demos),
                inputs=example.inputs(),
            )
            prompt_text = tokenizer.apply_chat_template(
                dspy_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize ─────────────────────────────────────────────────────────
            inputs_enc = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs_enc["input_ids"]
            self._logger.info(f"Prompt token count: {input_ids.shape[1]}")  # add this

            # Universal safe crop: match whatever the model will see
            from proc.demos.meeting_invite.tuning.abe_gpt.gpt import block_size
            shape_before = input_ids.shape
            input_ids = input_ids[:, -block_size:]
            shape_after = input_ids.shape
            if shape_before != shape_after:
                self._logger.warning(f"Data was truncated. Shape before: {shape_before}, Shape after: {shape_after}")

            tokens_str = [tokenizer.decode([t]) for t in input_ids[0]]

            with timed(f"example[{index}] hf_model forward (target selection)", logger=self._logger):
                target_id, target_tok = self._resolve_target_token(
                    hf_model, tokenizer, input_ids
                )
            self._logger.info(
                f"Attribution target: {repr(target_tok)} (id={target_id})  "
                f"mode={self._target_mode}"
            )

            # Panel B — Logit lens (fast, on inference_device) ─────────────────
            hf_model.to(self._inference_device)
            input_ids_inf = input_ids.to(self._inference_device)

            with timed(f"example[{index}] _logit_lens", logger=self._logger):
                layer_probes, target_prob, target_rank = self._logit_lens(
                    hf_model, tokenizer, input_ids_inf,
                    target_id, decoder_layers,
                )

            # Panel A — Captum LIG (always on attr_device=CPU) ─────────────────
            hf_model.to(self._attr_device)
            input_ids_attr = input_ids.to(self._attr_device)

            with timed(f"example[{index}] _run_lig", logger=self._logger, heartbeat_interval=0):
                lig_result = self._run_lig(
                    hf_model, embed_layer, input_ids_attr, target_id
                )
            if not is_successful(lig_result):
                return lig_result

            token_attr_normalized, convergence_delta = lig_result.unwrap()

            # Panel C — Map token saliency → DSPy prompt segments ─────────────
            segment_labels = self._assign_segment_labels(
                tokens_str, predictor, tokenizer, prompt_text=prompt_text
            )
            token_saliencies = [
                TokenSaliency(
                    token=tok,
                    index=i,
                    saliency=token_attr_normalized[i],
                    raw_norm=token_attr_normalized[i],  # already L2-normed
                    segment_label=segment_labels[i],
                )
                for i, tok in enumerate(tokens_str)
            ]
            segment_saliencies = self._aggregate_segments(
                token_saliencies, prompt_text, tokenizer
            )

            top1_reached_layer = next(
                (p.layer_index for p in layer_probes if p.first_top1), -1
            )

            example_result = LIGExampleResult(
                example_index=index,
                target_text=f"{target_tok!r} (model top-1)",
                target_token=target_tok,
                target_prob=target_prob,
                target_rank=target_rank,
                convergence_delta=convergence_delta,
                top1_reached_layer=top1_reached_layer,
                token_saliencies=token_saliencies,
                layer_probes=layer_probes,
                segment_saliencies=segment_saliencies,
            )
            example_results.append(example_result)
            self._logger.info(
                "[timing] example[%d] total: %.3fs",
                index,
                time.perf_counter() - _example_t0,
            )
            self._log_example(example_result)

        result = self._aggregate(example_results)
        self._log_summary(result)

        return Success(ProcScore(
            value=self._score(),
            context=LIGAttributionContext(result),
        ))

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> Result[tuple, ProcError]:
        """
        Load without device_map="auto" — critical for Captum compatibility.
        accelerate's AlignDevicesHook intercepts every forward call to stream
        weights from disk, which breaks Captum's interpolation loop.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._hf_model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self._hf_model_name,
                # dtype=torch.float32,   # float32 → stable gradients on CPU
                dtype=torch.float16,  # float32 → stable gradients on CPU
                trust_remote_code=True,
                # NO device_map="auto" — single device, no offloading
            )
            model.eval()
            return Success((model, tokenizer))
        except Exception as e:
            return Failure(ProcError(
                f"Failed to load '{self._hf_model_name}': {e}"
            ))

    def _get_embed_layer(self, model: Any) -> torch.nn.Module | None:
        """Locate embed_tokens for common HuggingFace CausalLM architectures."""
        # Qwen2 / LLaMA / Mistral / Phi
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        # GPT-2
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte
        # Falcon
        if hasattr(model, "transformer") and hasattr(model.transformer, "word_embeddings"):
            return model.transformer.word_embeddings
        # Generic
        if hasattr(model, "get_input_embeddings"):
            return model.get_input_embeddings()
        return None

    def _get_decoder_layers(self, model: Any) -> list | None:
        """Locate decoder layer list for logit lens hooks."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        return None

    # ── Panel B — Logit lens ───────────────────────────────────────────────────

    def _logit_lens(
        self,
        model: Any,
        tokenizer: Any,
        input_ids: torch.Tensor,
        target_id: int,
        decoder_layers: list,
    ) -> tuple[list[LayerProbe], float, int]:
        """
        Register a forward hook on each decoder layer to capture the residual
        stream at the last token position. Project through norm + lm_head to
        get the probability of target_id at each layer.

        This is the same hook strategy as in the reference script.
        """
        residuals: dict[int, torch.Tensor] = {}
        hook_handles = []

        def make_hook(idx: int):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                residuals[idx] = hidden[:, -1, :].detach().float().cpu()
            return hook

        for i, layer in enumerate(decoder_layers):
            hook_handles.append(layer.register_forward_hook(make_hook(i)))

        with timed("_logit_lens hf_model forward", logger=self._logger):
            with torch.no_grad():
                final_out = model(input_ids=input_ids)

        for h in hook_handles:
            h.remove()

        # Final layer stats
        final_probs = final_out.logits[0, -1].float().softmax(dim=-1).cpu()
        target_prob = float(final_probs[target_id])
        target_rank = int((final_probs > final_probs[target_id]).sum()) + 1

        # Project residuals through norm + lm_head on CPU
        # (avoids MPS float precision issues, matches reference)
        # lm_head_cpu = model.lm_head.to("cpu")
        # norm_cpu = model.model.norm.to("cpu") if hasattr(model.model, "norm") else None

        # Cast to float32 explicitly — residuals are stored as float32 (.detach().float().cpu())
        # but the model may be loaded in float16. Always project in float32 for numerical stability.
        lm_head_cpu = model.lm_head.to("cpu").float()
        norm_cpu = model.model.norm.to("cpu").float() if hasattr(model.model, "norm") else None

        probes: list[LayerProbe] = []
        seen_top1 = False

        for i in range(len(decoder_layers)):
            resid = residuals[i].squeeze(0)
            if norm_cpu is not None:
                normed = norm_cpu(resid.unsqueeze(0))
            else:
                normed = resid.unsqueeze(0)
            logits_l = lm_head_cpu(normed).squeeze(0).float()
            probs_l = logits_l.softmax(dim=-1).detach()
            rank = int((probs_l > probs_l[target_id]).sum()) + 1
            prob = float(probs_l[target_id])

            is_first_top1 = (rank == 1 and not seen_top1)
            if is_first_top1:
                seen_top1 = True

            probes.append(LayerProbe(
                layer_index=i,
                target_prob=prob,
                target_rank=rank,
                first_top1=is_first_top1,
            ))

        # Restore lm_head + norm to inference device AND original dtype.
        # Use self._model_dtype (stored once at load time) rather than querying
        # next(model.parameters()).dtype which can return float32 if lm_head
        # is listed first and was temporarily cast above.
        model.lm_head.to(self._inference_device).to(self._model_dtype)
        if norm_cpu is not None:
            model.model.norm.to(self._inference_device).to(self._model_dtype)

        return probes, target_prob, target_rank

    # ── Panel A — Captum LayerIntegratedGradients ─────────────────────────────

    def _run_lig(
        self,
        model: Any,
        embed_layer: torch.nn.Module,
        input_ids: torch.Tensor,
        target_id: int,
    ) -> Result[tuple[list[float], float], ProcError]:
        """
        Run Captum IntegratedGradients in native model dtype (float16 on MPS/CPU).

        STRATEGY — pure native dtype, zero conversions:
          - Keep embeddings in self._model_dtype on self._device (MPS or CPU)
          - Captum interpolates between baseline/actual embeddings in model dtype
          - forward_fn passes embeddings directly to model — no dtype casting
          - Only the scalar output is cast to float32 for Captum gradient stability
          - Model stays on self._device throughout — no device switching

        MODEL DTYPE:
          Uses self._model_dtype stored once at load time. Never queries
          next(model.parameters()).dtype at runtime because _logit_lens temporarily
          casts lm_head to float32, which would return the wrong dtype here.

        FALLBACK:
          If MPS attribution fails mid-run (OOM or autograd error), retry the
          entire example on CPU in float32 for guaranteed correctness.

        attribution = L2-norm of per-token embedding gradient, normalized to [0,1]
        """

        def _run_on_device(device: torch.device) -> tuple[list[float], float]:
            """Run full IG attribution on the given device in native model dtype."""
            total_calls = self._ig_steps + 1
            call_counter = [0]
            t0 = [time.perf_counter()]

            model.to(device)
            ids = input_ids.to(device)

            # Use stored model dtype — reliable regardless of _logit_lens mutations
            dtype = self._model_dtype if device == self._device else torch.float32

            if device != self._device:
                # CPU fallback: use float32 for numerical stability
                model.float()
                self._logger.info("CPU fallback: model cast to float32")

            # Compute embeddings in model dtype — no float() cast.
            # Captum interpolates between these in the same dtype.
            with torch.no_grad():
                input_embeds    = embed_layer(ids)                  # (1,T,C) dtype
                baseline_embeds = embed_layer(torch.zeros_like(ids))  # (1,T,C) dtype

            self._logger.info(
                f"_run_lig: device={device.type}  "
                f"dtype={dtype}  "
                f"embed shape={input_embeds.shape}  "
                f"n_steps={self._ig_steps}"
            )

            def forward_fn(embeds: torch.Tensor) -> torch.Tensor:
                """
                embeds: (batch, T, C) in model dtype — Captum interpolated embeddings.
                No dtype casting — model and embeddings share the same dtype.
                Returns float32 scalar for Captum gradient stability.
                """
                call_counter[0] += 1
                step = call_counter[0]

                if step % max(1, total_calls // 20) == 0 or step == 1 or step == total_calls:
                    elapsed = time.perf_counter() - t0[0]
                    rate = step / elapsed if elapsed > 0 else 0.0
                    remaining = (total_calls - step) / rate if rate > 0 else 0.0
                    self._logger.info(
                        f"    LIG step {step:>4d}/{total_calls}  "
                        f"({step / total_calls * 100:5.1f}%)  "
                        f"elapsed={elapsed:6.1f}s  eta={remaining:6.1f}s  "
                        f"rate={rate:.2f} steps/s  "
                        f"device={device.type}  dtype={dtype}"
                    )

                with timed(f"forward_fn (step={step}/{total_calls})", logger=self._logger):
                    out = model(inputs_embeds=embeds)
                # Cast scalar output to float32 — Captum needs float32 for its
                # internal gradient accumulation even when embeddings are float16
                return out.logits[0, -1, target_id].unsqueeze(0).float()

            ig = IntegratedGradients(forward_fn)

            with timed(
                f"_run_lig ig.attribute "
                f"(n_steps={self._ig_steps}, device={device.type}, dtype={dtype})",
                logger=self._logger,
            ):
                attributions, delta = ig.attribute(
                    inputs=input_embeds,
                    baselines=baseline_embeds,
                    n_steps=self._ig_steps,
                    return_convergence_delta=True,
                    internal_batch_size=self._internal_batch_size,
                )

            # L2-norm over embedding dim, normalize to [0,1], move to CPU
            token_attr = attributions[0].float().norm(dim=-1).detach().cpu()
            token_attr = (token_attr / (token_attr.max() + 1e-9)).tolist()
            return token_attr, float(delta.item())

        try:
            try:
                token_attr, delta = _run_on_device(self._device)
                return Success((token_attr, delta))

            except Exception as primary_err:
                if self._device.type == "cpu":
                    raise  # already on CPU — no fallback

                # MPS failed (OOM or autograd error) — fall back to CPU float32
                self._logger.warning(
                    f"Attribution on {self._device.type} failed: {primary_err}. "
                    "Retrying on CPU with float32 — slower but guaranteed correct."
                )
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                token_attr, delta = _run_on_device(torch.device("cpu"))

                # Restore model to original device + dtype after CPU fallback
                model.to(self._device).to(self._model_dtype)
                return Success((token_attr, delta))

        except Exception as e:
            # Last-resort cleanup — ensure model is back on self._device
            try:
                model.to(self._device).to(self._model_dtype)
            except Exception:
                pass
            return Failure(ProcError(f"Captum LIG failed: {e}"))

    # ── Target token resolution ────────────────────────────────────────────────

    # Tokens that are pure format/structure and should be skipped in "content" mode.
    # These are DSPy output format markers, whitespace, and common chat-template tokens.
    _FORMAT_TOKENS: frozenset[str] = frozenset({
        "[[", "]]", "##", " ##", "## ", " ", "\n", "\t", "",
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    })

    def _resolve_target_token(
        self,
        model: Any,
        tokenizer: Any,
        input_ids: torch.Tensor,
        max_skip_steps: int = 3,
    ) -> tuple[int, str]:
        """
        Resolve the attribution target token.

        target_mode="top1" (Option 2):
            Return the model's raw top-1 predicted next token, including format
            tokens like '[['. Answers: "what drove the model to start responding
            in this format?"

        target_mode="content" (Option 1):
            Greedily decode up to max_skip_steps tokens, skipping pure format/
            whitespace tokens, and return the first semantically meaningful token.
            Answers: "what drove the model to produce actual content?"

        The distinction matters when DSPy output format tokens like '[[' are
        the top-1 prediction — they tell you about format compliance, not about
        whether the model understood the email content.
        """
        ids = input_ids.to(self._inference_device)

        with torch.no_grad():
            out = model(input_ids=ids)
            top1_id = int(out.logits[0, -1].argmax())
        top1_tok = tokenizer.decode([top1_id])

        if self._target_mode == "top1":
            return top1_id, top1_tok

        # "content" mode — skip format/whitespace tokens
        current_ids = ids
        for step in range(max_skip_steps):
            next_id = int(model(input_ids=current_ids).logits[0, -1].argmax())
            next_tok = tokenizer.decode([next_id])

            # Accept if not a pure format token
            if next_tok.strip() and next_tok not in self._FORMAT_TOKENS:
                self._logger.debug(
                    f"content mode: skipped {step} format token(s), "
                    f"settled on {repr(next_tok)} (id={next_id})"
                )
                return next_id, next_tok

            # Extend context with the generated format token and try again
            next_tensor = torch.tensor([[next_id]], device=self._inference_device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)

        # Fallback: top-1 of original prompt if all steps were format tokens
        self._logger.warning(
            f"content mode: all {max_skip_steps} generated tokens were format tokens, "
            f"falling back to raw top-1 {repr(top1_tok)}"
        )
        return top1_id, top1_tok

    # ── Panel C — Segment label assignment ────────────────────────────────────
    def _assign_segment_labels(
            self,
            tokens_str: list[str],
            predictor: Any,
            tokenizer: Any,
            prompt_text: str = "",
    ) -> list[str]:
        """
        Assign fine-grained segment labels by finding known DSPy prompt
        markers in prompt_text and labeling token ranges accordingly.

        Supports two chat template formats:
          - [SYSTEM] / [USER] / [ASSISTANT]  (AbeGPT/custom adapter)
          - <|im_start|>system / <|im_start|>user  (Qwen/OpenAI chat template)

        Segments produced:
            system_wrapper   — role tags ([SYSTEM], <|im_start|>system, etc.)
            instruction      — the task instructions / rules / JSON schema
            field_label      — [[ ## field_name ## ]] markers
            email_from       — sender address value
            email_to         — recipient address value
            email_body       — the email body value (most important for task)
            current_date     — the date value
            demo_N           — bootstrapped few-shot examples
            input            — anything not matched by the above
        """
        import re as _re
        from collections import Counter as _Counter

        labels = ["unknown"] * len(tokens_str)

        if not prompt_text:
            return ["input"] * len(tokens_str)

        # Build per-token character start offsets from tokens_str.
        token_char_starts = []
        pos = 0
        for tok in tokens_str:
            token_char_starts.append(pos)
            pos += len(tok)
        token_char_starts.append(pos)  # sentinel

        def char_to_token(char_pos: int) -> int:
            lo, hi = 0, len(token_char_starts) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if token_char_starts[mid] <= char_pos:
                    lo = mid + 1
                else:
                    hi = mid
            return max(0, lo - 1)

        def label_span(start_char: int, end_char: int, label: str) -> None:
            t_start = char_to_token(start_char)
            t_end = char_to_token(end_char)
            for i in range(t_start, min(t_end + 1, len(labels))):
                labels[i] = label

        def find_and_label(text: str, label: str, use_rfind: bool = False) -> bool:
            """Find text in prompt_text and label its token range."""
            if not text:
                return False
            # Try exact match, then stripped
            for candidate in [text, text.strip()]:
                idx = prompt_text.rfind(candidate) if use_rfind else prompt_text.find(candidate)
                if idx != -1:
                    label_span(idx, idx + len(candidate), label)
                    return True
            return False

        # ── 1. System/User/Assistant wrapper tags (both formats) ──────────────
        wrapper_tags = [
            "[SYSTEM]", "[USER]", "[ASSISTANT]",
            "<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant",
            "<|im_start|>", "<|im_end|>",
        ]
        for tag in wrapper_tags:
            start = 0
            while True:
                idx = prompt_text.find(tag, start)
                if idx == -1:
                    break
                label_span(idx, idx + len(tag), "system_wrapper")
                start = idx + len(tag)

        # ── 2. Field label markers [[ ## name ## ]] ───────────────────────────
        for m in _re.finditer(r'\[\[\s*##.*?##\s*\]\]', prompt_text, _re.DOTALL):
            label_span(m.start(), m.end(), "field_label")

        # ── 3. Task instruction block ─────────────────────────────────────────
        instr_text = predictor.signature.instructions or ""
        if instr_text and not find_and_label(instr_text, "instruction"):
            # Try matching via first substantial line
            first_line = next(
                (l.strip() for l in instr_text.splitlines() if len(l.strip()) > 20),
                ""
            )
            if first_line:
                idx = prompt_text.find(first_line)
                if idx != -1:
                    instr_end = idx + len(instr_text.strip())
                    label_span(idx, min(instr_end, len(prompt_text)), "instruction")
                    self._logger.debug("instruction matched via first-line heuristic")
                else:
                    self._logger.warning(
                        "instruction text not found in prompt — will be labeled as 'input'. "
                        "Check that predictor.signature.instructions matches the rendered prompt."
                    )

        # ── 4. Input field values ─────────────────────────────────────────────
        field_labels = {
            "email_from":   "email_from",
            "email_to":     "email_to",
            "email_body":   "email_body",
            "current_date": "current_date",
        }
        for field_name, seg_label in field_labels.items():
            marker = f"[[ ## {field_name} ## ]]"
            marker_idx = prompt_text.rfind(marker)
            if marker_idx == -1:
                continue
            value_start = marker_idx + len(marker)
            next_marker = _re.search(r'\[\[', prompt_text[value_start:])
            value_end = value_start + next_marker.start() if next_marker else len(prompt_text)
            value_text = prompt_text[value_start:value_end].strip()
            if value_text:
                find_and_label(value_text, seg_label, use_rfind=True)

        # ── 5. Few-shot demos ─────────────────────────────────────────────────
        for demo_idx, demo in enumerate(predictor.demos):
            demo_text = self._demo_to_text(demo, predictor.signature)
            find_and_label(demo_text, f"demo_{demo_idx}")

        # ── 6. Fill remaining unknowns as input ──────────────────────────────
        for i in range(len(labels)):
            if labels[i] == "unknown":
                labels[i] = "input"

        # ── 7. Log labeling summary ───────────────────────────────────────────
        label_counts = _Counter(labels)
        self._logger.debug(
            "segment label distribution: " +
            ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items()))
        )
        if label_counts.get("instruction", 0) == 0:
            self._logger.warning(
                "No tokens labeled as 'instruction'. "
                "Panel C instruction saliency will be missing."
            )

        return labels

    def _assign_segment_labels_old(
        self,
        tokens_str: list[str],
        predictor: Any,
        tokenizer: Any,
    ) -> list[str]:
        """
        Assign a DSPy segment label to each token position by tokenizing
        each segment (instruction, each demo, input) and matching token counts.

        Returns a list of labels, one per token, same length as tokens_str.
        """
        labels = ["unknown"] * len(tokens_str)
        cursor = 0

        def assign_range(start: int, end: int, label: str) -> None:
            for i in range(start, min(end, len(labels))):
                labels[i] = label

        # Instruction
        instr_text = predictor.signature.instructions or ""
        instr_ids = tokenizer(instr_text, add_special_tokens=False)["input_ids"]
        end = cursor + len(instr_ids)
        assign_range(cursor, end, "instruction")
        cursor = end

        # Each demo
        for demo_idx, demo in enumerate(predictor.demos):
            demo_text = self._demo_to_text(demo, predictor.signature)
            demo_ids = tokenizer(demo_text, add_special_tokens=False)["input_ids"]
            end = cursor + len(demo_ids)
            assign_range(cursor, end, f"demo_{demo_idx}")
            cursor = end

        # Remaining tokens = input
        assign_range(cursor, len(labels), "input")

        return labels

    def _demo_to_text(self, demo: Any, signature: Any) -> str:
        lines = []
        for field_name in list(signature.input_fields) + list(signature.output_fields):
            val = getattr(demo, field_name, None) or (
                demo.get(field_name, "") if hasattr(demo, "get") else ""
            )
            lines.append(f"{field_name}: {val}")
        return "\n".join(lines)

    def _aggregate_segments(
        self,
        token_saliencies: list[TokenSaliency],
        prompt_text: str,
        tokenizer: Any,
    ) -> list[SegmentSaliency]:
        """
        Group token saliencies by segment label and compute per-segment averages.
        Also uses the Panel C regex-split approach from the reference script to
        build human-readable segment previews.
        """
        by_label: dict[str, list[float]] = defaultdict(list)
        for ts in token_saliencies:
            by_label[ts.segment_label].append(ts.saliency)

        # Build text previews from the rendered prompt (Panel C approach)
        split_parts = [
            s.strip()
            for s in re.split(_SEGMENT_SPLIT_PATTERN, prompt_text)
            if s.strip()
        ]

        results: list[SegmentSaliency] = []
        for label, scores in sorted(by_label.items()):
            avg = sum(scores) / len(scores) if scores else 0.0
            preview_raw = next(
                (p for p in split_parts if label.replace("_", " ") in p.lower()),
                label,
            )
            preview = textwrap.shorten(preview_raw, width=60, placeholder="…")
            results.append(SegmentSaliency(
                label=label,
                text_preview=preview,
                avg_saliency=avg,
                token_count=len(scores),
            ))

        return sorted(results, key=lambda s: s.avg_saliency, reverse=True)

    # ── Aggregation across dataset ─────────────────────────────────────────────

    def _aggregate(self, example_results: list[LIGExampleResult]) -> LIGAttributionResult:
        input_sals = [
            s.avg_saliency
            for er in example_results
            for s in er.segment_saliencies
            if s.label == "input"
        ]
        instr_sals = [
            s.avg_saliency
            for er in example_results
            for s in er.segment_saliencies
            if s.label == "instruction"
        ]
        demo_sals = [
            s.avg_saliency
            for er in example_results
            for s in er.segment_saliencies
            if s.label.startswith("demo_")
        ]

        pct_top1 = sum(
            1 for er in example_results if er.top1_reached_layer >= 0
        ) / len(example_results)

        top1_layers = [
            er.top1_reached_layer
            for er in example_results
            if er.top1_reached_layer >= 0
        ]
        avg_top1_layer = sum(top1_layers) / len(top1_layers) if top1_layers else -1.0

        all_tokens = [
            ts
            for er in example_results
            for ts in er.token_saliencies
        ]
        top_tokens = sorted(all_tokens, key=lambda t: t.saliency, reverse=True)

        # Deduplicate top tokens by token string
        seen: set[str] = set()
        deduped: list[TokenSaliency] = []
        for t in top_tokens:
            if t.token not in seen:
                seen.add(t.token)
                deduped.append(t)
            if len(deduped) >= _TOP_TOKEN_COUNT:
                break

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        return LIGAttributionResult(
            avg_input_saliency=_mean(input_sals),
            avg_instruction_saliency=_mean(instr_sals),
            avg_demo_saliency=_mean(demo_sals),
            avg_convergence_delta=_mean([er.convergence_delta for er in example_results]),
            pct_examples_top1_reached=pct_top1,
            avg_top1_layer=avg_top1_layer,
            top_saliency_tokens=deduped,
            model_name=self._hf_model_name,
            n_examples=len(example_results),
            example_results=example_results,
        )

    # ── Logging ────────────────────────────────────────────────────────────────
    def _log_example(self, er: LIGExampleResult) -> None:
        # Delta interpretation
        delta = er.convergence_delta
        if abs(delta) < 0.05:
            delta_comment = "reliable"
        elif abs(delta) < 0.5:
            delta_comment = "borderline — consider increasing ig_steps"
        else:
            delta_comment = "unreliable — increase ig_steps to 200+"

        self._logger.info(
            f"Example {er.example_index} — target='{er.target_token}'  "
            f"rank=#{er.target_rank}  prob={er.target_prob:.4f}  "
            f"delta={er.convergence_delta:.4f} ({delta_comment})  "
            f"top1_layer={er.top1_reached_layer}"
        )

        self._logger.info("\n  Panel A — Top-5 salient tokens:")
        for ts in er.top5_tokens:
            self._logger.info(
                f"    [{ts.index:04d}] {repr(ts.token):20s}  "
                f"{ts.saliency:.3f}  ({ts.segment_label})"
            )

        # Panel B — Logit lens: per-layer rank trajectory
        self._logger.info("\n  Panel B — Logit lens (rank of target token per layer):")
        self._logger.info(
            "    rank = position of target token in the vocabulary probability distribution"
        )
        self._logger.info(
            "    rank=#1 means the model is most confident about this token at this layer."
        )
        self._logger.info(
            "    Ideal pattern: rank starts high (uncertain) and drops to #1 in middle layers"
        )
        self._logger.info(
            "    (reasoning happening). Committing at layer 0 = pattern matching, not reasoning."
        )
        self._logger.info(
            f"    {'layer':>5s}  {'rank':>6s}  {'prob':>7s}  note"
        )
        self._logger.info(f"    {'-' * 60}")

        n_layers = len(er.layer_probes)
        for probe in er.layer_probes:
            notes = []
            if probe.first_top1:
                if probe.layer_index == 0:
                    notes.append("⚠ rank-1 at first layer — pattern matching, not reasoning")
                elif probe.layer_index < n_layers // 2:
                    notes.append("✓ rank-1 reached early — confident decision")
                else:
                    notes.append("~ rank-1 reached late — decision made in deep layers")
            if probe.layer_index == n_layers - 1:
                if probe.target_rank == 1:
                    notes.append("✓ final layer confirms rank-1")
                else:
                    notes.append(f"⚠ final layer rank=#{probe.target_rank} — model uncertain at output")
            # Flag if rank is getting worse (increasing) across layers
            if probe.layer_index > 0:
                prev = er.layer_probes[probe.layer_index - 1]
                if probe.target_rank > prev.target_rank * 2:
                    notes.append("↑ rank worsening — later layers hurt this prediction")

            note_str = "  ".join(notes)
            self._logger.info(
                f"    {probe.layer_index:>5d}  {probe.target_rank:>6d}  "
                f"{probe.target_prob:>7.4f}  {note_str}"
            )

        # Panel B story — interpret the overall trajectory
        first_top1_layer = next((p.layer_index for p in er.layer_probes if p.first_top1), -1)
        final_rank = er.layer_probes[-1].target_rank if er.layer_probes else -1
        n_layers = len(er.layer_probes)

        self._logger.info("  Panel B — Trajectory interpretation:")
        if first_top1_layer == -1:
            self._logger.info(
                "    ✗ Target never reached rank-1 across all layers. "
                "The model is uncertain about this token even at the final layer. "
                "→ Prompt optimization tip: the model may not understand the output "
                "format — consider adding clearer output examples or restructuring "
                "the instruction to make the expected first token more explicit."
            )
        elif first_top1_layer == 0:
            self._logger.info(
                "    ⚠ Pattern matching detected: rank-1 achieved at layer 0. "
                "The model commits to this token immediately from the embedding layer, "
                "before any transformer reasoning occurs. This means the prediction "
                "is driven by surface-level token co-occurrence, not by understanding "
                "the prompt content. "
                "→ Prompt optimization tip: reformulate the instruction to require "
                "reasoning before outputting — e.g. add a 'think step by step' prefix "
                "or require a reasoning field before the structured output."
            )
        elif first_top1_layer < n_layers // 3:
            self._logger.info(
                f"    ✓ Early commitment: rank-1 reached at layer {first_top1_layer}/{n_layers}. "
                "The model resolves the target token in the early layers, suggesting "
                "strong learned associations between prompt structure and output format. "
                "This is healthy for format compliance but may indicate the model is "
                "deciding output format before fully processing the email content."
            )
        elif first_top1_layer < (2 * n_layers) // 3:
            self._logger.info(
                f"    ✓ Ideal reasoning pattern: rank-1 reached at layer {first_top1_layer}/{n_layers}. "
                "The model starts uncertain and resolves the target in the middle layers — "
                "this is the signature of genuine semantic reasoning where later transformer "
                "blocks integrate information from across the prompt. "
                "→ This is the desired behavior for a well-prompted extraction task."
            )
        else:
            self._logger.info(
                f"    ~ Late commitment: rank-1 reached at layer {first_top1_layer}/{n_layers}. "
                "The model only resolves the target in the final layers. This can mean "
                "the task is genuinely hard for this model, or the prompt provides "
                "conflicting signals that require deep processing to resolve. "
                "→ Prompt optimization tip: if this is consistent across examples, "
                "simplify the instruction or reduce the number of rules to reduce "
                "the cognitive load on later layers."
            )

        self._logger.info("\n  Panel C — Segment saliency (high → low):")
        self._logger.info(
            f"    {'segment':<20s}  {'avg_sal':>7s}  {'tokens':>7s}  {'ratio':>8s}  comment"
        )
        self._logger.info(f"    {'-' * 75}")

        _RATIO_HIGH = 0.003
        _RATIO_MED = 0.001
        _EXPECTED_HIGH = {"email_body", "instruction"}
        _EXPECTED_LOW = {"system_wrapper", "field_label", "email_to", "email_from"}

        for seg in er.segment_saliencies:
            n = seg.token_count
            ratio = seg.avg_saliency / n if n > 0 else 0.0

            if ratio >= _RATIO_HIGH:
                if seg.label in _EXPECTED_HIGH:
                    comment = "✓ model reads this"
                elif seg.label in _EXPECTED_LOW:
                    comment = "⚠ over-weighted — should not dominate"
                else:
                    comment = "✓ active"
            elif ratio >= _RATIO_MED:
                if seg.label in _EXPECTED_HIGH:
                    comment = "⚠ under-weighted — should be higher"
                else:
                    comment = "~ borderline"
            else:
                if seg.label in _EXPECTED_HIGH:
                    comment = "✗ ignored — prompt optimization needed"
                else:
                    comment = "~ ignored (ok if not critical)"

            self._logger.info(
                f"    {seg.label:<20s}  {seg.avg_saliency:>7.4f}  {n:>7d}  "
                f"{ratio:>8.5f}  {comment}"
            )

    def _log_summary(self, result: LIGAttributionResult) -> None:
        self._logger.info(
            f"\n{'─' * 64}\n"
            f"LIG Attribution Summary  [{result.model_name}]  "
            f"{result.n_examples} examples\n"
            f"{'─' * 64}"
        )
        self._logger.info(
            f"  Panel C — Segment saliency:\n"
            f"    input       : {result.avg_input_saliency:.4f}  (should be highest)\n"
            f"    instruction : {result.avg_instruction_saliency:.4f}  (>0.1 used, <0.05 ignored)\n"
            f"    demos       : {result.avg_demo_saliency:.4f}  (>0.1 real work, <0.02 dead weight)"
        )
        self._logger.info(
            f"  Panel B — Logit lens:\n"
            f"    target reached rank-1 in {result.pct_examples_top1_reached * 100:.0f}% of examples\n"
            f"    avg layer of first rank-1: {result.avg_top1_layer:.1f}"
        )
        self._logger.info(
            f"  Convergence delta: {result.avg_convergence_delta:.4f}  "
            f"({'reliable' if result.avg_convergence_delta < 0.05 else 'increase ig_steps'})"
        )
        self._logger.info("  Top salient tokens (Panel A, global):")
        for ts in result.top_saliency_tokens[:5]:
            self._logger.info(
                f"    {repr(ts.token):20s}  {ts.saliency:.3f}  ({ts.segment_label})"
            )

    def _score(self) -> float:
        return self._SCORE
