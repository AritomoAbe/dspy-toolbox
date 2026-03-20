"""
GPT → HuggingFace adapter for LIGAttributionAuditor
=====================================================
LIGAttributionAuditor expects a HuggingFace (model, tokenizer) pair.
This module provides two thin wrappers:

    GPTModelAdapter   — wraps GPTLanguageModel so it looks like a HF CausalLM:
                        .model.embed_tokens   (the embedding layer Captum needs)
                        .model.layers         (decoder layers for logit lens hooks)
                        .model.norm           (final layer norm)
                        .lm_head              (vocabulary projection)
                        .forward(input_ids)   (returns an object with .logits)

    GPTTokenizerAdapter — wraps the character-level encode/decode so it looks
                          like a HF AutoTokenizer:
                          .encode(text)
                          .decode(ids)
                          .apply_chat_template(messages)
                          .vocab_size

Usage in your demo script
--------------------------
    from gpt_hf_adapter import GPTModelAdapter, GPTTokenizerAdapter
    from gpt import GPTLanguageModel, encode, decode, vocab_size, device

    # Load your trained GPT
    gpt = GPTLanguageModel().to("cpu")
    gpt.load_state_dict(torch.load("gpt.pt", map_location="cpu"))

    # Wrap for LIGAttributionAuditor
    hf_model     = GPTModelAdapter(gpt)
    hf_tokenizer = GPTTokenizerAdapter(encode, decode, vocab_size)

    # Override _load_model in the auditor (see GptLIGAttributionAuditor below)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


# ── Logits container (mimics HuggingFace CausalLMOutput) ─────────────────────

@dataclass
class _CausalLMOutput:
    logits: torch.Tensor


# ── Model adapter ─────────────────────────────────────────────────────────────

class _InnerModel(nn.Module):
    """
    Exposes the attributes LIGAttributionAuditor looks for:
        .embed_tokens   — token embedding table (Captum attaches LIG here)
        .layers         — list of transformer Block modules (logit lens hooks)
        .norm           — final LayerNorm (used in logit lens projection)
    """

    def __init__(self, gpt: nn.Module):
        super().__init__()
        # These names must match what _get_embed_layer / _get_decoder_layers
        # look for in LIGAttributionAuditor:
        self.embed_tokens = gpt.token_embedding_table   # nn.Embedding
        self.layers       = list(gpt.blocks)            # list of Block modules
        self.norm         = gpt.ln_f                    # final LayerNorm
        self._gpt         = gpt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._gpt.blocks(x)


class GPTModelAdapter(nn.Module):
    """
    Makes GPTLanguageModel look like a HuggingFace AutoModelForCausalLM.

    LIGAttributionAuditor accesses:
        model.model.embed_tokens   <- token embedding (LIG layer)
        model.model.layers         <- decoder blocks (logit lens hooks)
        model.model.norm           <- final norm
        model.lm_head              <- vocab projection
        model(input_ids=...)       <- forward pass returning .logits

    The GPT's positional embedding is handled inside forward() to keep
    the interface identical to HuggingFace.
    """

    def __init__(self, gpt: nn.Module):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.model   = _InnerModel(gpt)
        self.lm_head = gpt.lm_head
        self._gpt    = gpt

        # Expose config attributes the auditor may read
        self.config  = type("Config", (), {
            "num_hidden_layers": len(list(gpt.blocks)),
        })()

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> _CausalLMOutput:
        # Crop to block_size — position_embedding_table only has block_size entries.
        # DSPy prompts can easily exceed 512 tokens; silently truncate from the left
        # (keep the most recent tokens, matching how generate() crops context).
        from proc.demos.meeting_invite.tuning.abe_gpt.gpt import block_size
        shape_before = input_ids.shape
        input_ids = input_ids[:, -block_size:]
        shape_after = input_ids.shape
        if shape_before != shape_after:
            self._logger.warning(f'Input shapes have been truncated: {shape_before} != {shape_after}')

        B, T = input_ids.shape
        device = input_ids.device

        tok_emb = self._gpt.token_embedding_table(input_ids)
        pos_ids = torch.arange(T, device=device)
        pos_emb = self._gpt.position_embedding_table(pos_ids)

        x       = tok_emb + pos_emb                                     # (B, T, C)
        x       = self._gpt.blocks(x)                                   # (B, T, C)
        x       = self._gpt.ln_f(x)                                     # (B, T, C)
        logits  = self._gpt.lm_head(x)                                  # (B, T, vocab)

        return _CausalLMOutput(logits=logits)

    def get_input_embeddings(self) -> nn.Embedding:
        return self._gpt.token_embedding_table


# ── Tokenizer adapter ──────────────────────────────────────────────────────────

class GPTTokenizerAdapter:
    """
    Makes the character-level tokenizer look like a HuggingFace AutoTokenizer.

    LIGAttributionAuditor calls:
        tokenizer(text, return_tensors="pt")  -> {"input_ids": tensor}
        tokenizer.decode([id, id, ...])       -> str
        tokenizer.encode(text, add_special_tokens=False) -> list[int]
        tokenizer.apply_chat_template(messages, ...)     -> str
        tokenizer.vocab_size                             -> int

    KEY DESIGN NOTE:
        The stored callables are named _encode_fn / _decode_fn (not _encode /
        _decode) to prevent any possibility of them shadowing or being confused
        with the public encode() / decode() methods. The previous naming caused
        a circular call: decode() -> self._decode(ids) where Python interpreted
        ids as `self` when _decode accidentally pointed to self.decode, causing:
            TypeError: decode() missing 1 required positional argument: 'ids'
    """

    def __init__(
        self,
        encode_fn: Callable[[str], list[int]],
        decode_fn: Callable[[list[int]], str],
        vocab_size: int,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._encode_fn = encode_fn   # always gpt.encode — never self.encode
        self._decode_fn = decode_fn   # always gpt.decode — never self.decode
        self.vocab_size = vocab_size

        # HF tokenizer attributes the auditor may reference
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.bos_token_id = 0

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
        **kwargs,
    ) -> dict:
        ids = self._safe_encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        return self._safe_encode(text)

    def decode(
        self,
        ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        ids = [int(i) for i in ids]
        try:
            return self._decode_fn(ids)
        except Exception as e:
            self._logger.warning(f'Decode failed for ids {ids[:5]}...: {e}')
            return ''

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        """
        The character-level GPT has no chat template — just concatenate
        all message contents. LIGAttributionAuditor uses this to build
        the prompt string it tokenizes for the logit lens + LIG passes.
        """
        parts = []
        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"[{role.upper()}] {content}")
        return "\n".join(parts)

    def _safe_encode(self, text: str) -> list[int]:
        try:
            result = self._encode_fn(text)
            return result if result else [0]
        except Exception as e:
            self._logger.warning(f'Encoding failed: {e}')
            return [0]

# ── GptLIGAttributionAuditor ───────────────────────────────────────────────────

class GptLIGAttributionAuditor:
    """
    Convenience factory that wires a trained GPTLanguageModel into
    LIGAttributionAuditor without downloading any HuggingFace model.

    Usage
    -----
        from gpt import GPTLanguageModel, encode, decode, vocab_size
        from gpt_hf_adapter import GptLIGAttributionAuditor

        gpt = GPTLanguageModel()
        gpt.load_state_dict(torch.load("gpt.pt", map_location="cpu"))

        auditor = GptLIGAttributionAuditor(
            dataset=dataset,
            llm=llm,
            scorer=scorer,
            gpt_model=gpt,
            encode_fn=encode,
            decode_fn=decode,
            vocab_size=vocab_size,
            ig_steps=50,
        )
        result = auditor.invoke()
    """

    def __new__(
        cls,
        gpt_model: nn.Module,
        encode_fn: Callable,
        decode_fn: Callable,
        vocab_size: int,
        **auditor_kwargs,
    ):
        from proc.pipeline.llm_prompt_usage_attribution.lig_attribution_auditor import (
            LIGAttributionAuditor,
        )
        from returns.result import Success

        hf_model     = GPTModelAdapter(gpt_model)
        hf_tokenizer = GPTTokenizerAdapter(encode_fn, decode_fn, vocab_size)

        instance = LIGAttributionAuditor(
            hf_model_name="custom-gpt-abe",
            **auditor_kwargs,
        )

        # Override _load_model to inject our pre-built adapter pair.
        # Supports both auditor versions:
        #   new: single self._device
        #   old: separate self._attr_device + self._inference_device
        def _load_model_override(self_inner):
            device = (
                getattr(self_inner, "_device", None)
                or getattr(self_inner, "_attr_device", None)
            )
            hf_model.eval()
            if device is not None:
                hf_model.to(device)
            return Success((hf_model, hf_tokenizer))

        import types
        instance._load_model = types.MethodType(_load_model_override, instance)

        return instance
