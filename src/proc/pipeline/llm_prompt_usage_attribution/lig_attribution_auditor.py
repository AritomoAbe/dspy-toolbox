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

Requirements
------------
  pip install dspy torch transformers captum
"""

import logging
import re
import textwrap
from collections import defaultdict
from typing import Any

import torch
from captum.attr import LayerIntegratedGradients
from dspy.adapters import ChatAdapter
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

    def __init__(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
        hf_model_name: str = _DEFAULT_MODEL,
        ig_steps: int = _DEFAULT_IG_STEPS,
        internal_batch_size: int = _DEFAULT_INTERNAL_BATCH,
        attr_device: str = "cpu",
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._dataset = dataset
        self._llm = llm
        self._scorer = scorer
        self._hf_model_name = hf_model_name
        self._ig_steps = ig_steps
        self._internal_batch_size = internal_batch_size
        self._attr_device = torch.device(attr_device)

        # Best available device for fast inference (logit lens)
        # if torch.backends.mps.is_available():
        #     self._inference_device = torch.device("mps")
        # elif torch.cuda.is_available():
        #     self._inference_device = torch.device("cuda")
        # else:
        #     self._inference_device = torch.device("cpu")

        # Single device for all passes — no switching between MPS and CPU.
        # Moving the model between devices requires holding both copies simultaneously
        # (~2x model size), which exhausts RAM and triggers SIGKILL on Apple Silicon.
        self._device = torch.device(attr_device)
        self._inference_device = self._device  # kept for compatibility
        self._attr_device = self._device  # kept for compatibility

    # ── ProcNode entry point ───────────────────────────────────────────────────

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

        load_result = self._load_model()
        if isinstance(load_result, Failure):
            return load_result

        hf_model, tokenizer = load_result.unwrap()

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

        adapter = ChatAdapter()
        example_results: list[LIGExampleResult] = []

        for index, example in enumerate(self._dataset.load()):
            self._logger.info(f"Processing example {index}")

            # Generate prediction via API LLM ──────────────────────────────────
            with dspy.context(cache=False):
                pred = self._llm(**example.inputs())

            score = self._scorer.extraction_metric(example, pred)
            if score == INVALID_SCORE:
                return Failure(ProcError(f"Cannot score example {index}"))

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

            inputs_enc = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs_enc["input_ids"]
            input_ids = input_ids[:, -256:]  # crop to GPT block_size

            # Universal safe crop: match whatever the model will see
            from proc.demos.meeting_invite.tuning.abe_gpt.gpt import block_size as _gpt_block_size
            input_ids = input_ids[:, -_gpt_block_size:]

            tokens_str = [tokenizer.decode([t]) for t in input_ids[0]]

            # Resolve attribution target token from prediction ──────────────────
            target_text = self._pred_to_text(pred)
            target_id, target_tok = self._resolve_target(
                tokenizer, hf_model, input_ids, target_text
            )

            # Panel B — Logit lens (fast, on inference_device) ─────────────────
            hf_model.to(self._inference_device)
            input_ids_inf = input_ids.to(self._inference_device)

            layer_probes, target_prob, target_rank = self._logit_lens(
                hf_model, tokenizer, input_ids_inf,
                target_id, decoder_layers,
            )

            # Panel A — Captum LIG (always on attr_device=CPU) ─────────────────
            hf_model.to(self._attr_device)
            input_ids_attr = input_ids.to(self._attr_device)

            lig_result = self._run_lig(
                hf_model, embed_layer, input_ids_attr, target_id
            )
            if isinstance(lig_result, Failure):
                return lig_result

            token_attr_normalized, convergence_delta = lig_result.unwrap()

            # Panel C — Map token saliency → DSPy prompt segments ─────────────
            segment_labels = self._assign_segment_labels(
                tokens_str, predictor, tokenizer
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
                target_text=target_text,
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
                torch_dtype=torch.float32,   # float32 → stable gradients on CPU
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
        lm_head_cpu = model.lm_head.to("cpu")
        norm_cpu = model.model.norm.to("cpu") if hasattr(model.model, "norm") else None

        probes: list[LayerProbe] = []
        seen_top1 = False

        for i in range(len(decoder_layers)):
            resid = residuals[i].squeeze(0)
            if norm_cpu is not None:
                normed = norm_cpu(resid.unsqueeze(0))
            else:
                normed = resid.unsqueeze(0)
            logits_l = lm_head_cpu(normed).squeeze(0).float()
            probs_l = logits_l.softmax(dim=-1)
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

        # Restore lm_head + norm to inference_device
        model.lm_head.to(self._inference_device)
        if norm_cpu is not None:
            model.model.norm.to(self._inference_device)

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
        Run Captum LayerIntegratedGradients on CPU.

        baseline = zero-id tensor (all pad tokens)
        attribution = L2-norm of per-token embedding gradient, normalized to [0,1]

        Matches the reference script exactly:
            token_attr = attributions[0].float().norm(dim=-1)
            token_attr = token_attr / (token_attr.max() + 1e-9)
        """
        try:
            def forward_fn(ids: torch.Tensor) -> torch.Tensor:
                out = model(input_ids=ids)
                return out.logits[0, -1, target_id].unsqueeze(0).float()

            lig = LayerIntegratedGradients(forward_fn, embed_layer)
            baseline_ids = torch.zeros_like(input_ids)

            attributions, delta = lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                n_steps=self._ig_steps,
                return_convergence_delta=True,
                internal_batch_size=self._internal_batch_size,
            )

            token_attr = attributions[0].float().norm(dim=-1).detach().cpu()
            token_attr = (token_attr / (token_attr.max() + 1e-9)).tolist()

            return Success((token_attr, float(delta.item())))

        except Exception as e:
            return Failure(ProcError(f"Captum LIG failed: {e}"))

    # ── Target token resolution ────────────────────────────────────────────────

    def _resolve_target(
        self,
        tokenizer: Any,
        model: Any,
        input_ids: torch.Tensor,
        target_text: str,
    ) -> tuple[int, str]:
        """
        Resolve the first token of target_text as the attribution target.
        Falls back to model top-1 if target_text doesn't tokenize cleanly.
        Matches the reference script's candidate resolution logic.
        """
        # Try " {word}" first (leading space), then bare word
        first_word = target_text.split()[0] if target_text.strip() else ""
        for candidate in [f" {first_word}", first_word]:
            enc = tokenizer.encode(candidate, add_special_tokens=False)
            if enc:
                return enc[0], tokenizer.decode([enc[0]])

        # Fallback: model top-1
        ids_inf = input_ids.to(self._inference_device)
        with torch.no_grad():
            out = model(input_ids=ids_inf)
            top1_id = int(out.logits[0, -1].argmax())
        return top1_id, tokenizer.decode([top1_id])

    # ── Panel C — Segment label assignment ────────────────────────────────────

    def _assign_segment_labels(
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

    # ── Prediction serialisation ───────────────────────────────────────────────

    def _pred_to_text(self, pred: Any) -> str:
        try:
            keys = list(pred.keys())
        except AttributeError:
            keys = [k for k in vars(pred) if not k.startswith("_")]

        for key in keys:
            try:
                val = pred[key]
            except (KeyError, TypeError):
                val = getattr(pred, key, None)
            if val is None:
                continue
            text = str(val).strip()
            if text:
                return text
        return ""

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log_example(self, er: LIGExampleResult) -> None:
        self._logger.info(
            f"Example {er.example_index} — target='{er.target_token}'  "
            f"rank=#{er.target_rank}  prob={er.target_prob:.4f}  "
            f"delta={er.convergence_delta:.4f}  "
            f"top1_layer={er.top1_reached_layer}"
        )
        self._logger.info("  Panel A — Top-5 salient tokens:")
        for ts in er.top5_tokens:
            self._logger.info(
                f"    [{ts.index:03d}] {repr(ts.token):20s}  "
                f"{ts.saliency:.3f}  ({ts.segment_label})"
            )
        self._logger.info("  Panel C — Segment saliency (high → low):")
        for seg in er.segment_saliencies:
            self._logger.info(
                f"    {seg.label:20s}  {seg.avg_saliency:.3f}  "
                f"({seg.token_count} tokens)"
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
