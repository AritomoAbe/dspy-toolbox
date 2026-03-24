"""
Unit tests for LIGAttributionAuditor._assign_segment_labels
============================================================

Tests cover:
  - AbeGPT format ([SYSTEM] / [USER])
  - Qwen chat template format (<|im_start|>system / <|im_start|>user)
  - Field value extraction (email_from, email_to, email_body, current_date)
  - Instruction matching — exact and first-line heuristic
  - Missing instruction warning
  - Empty prompt fallback
  - Label coverage (every token gets a label, no "unknown" left)
  - Demo labeling
"""

import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_auditor():
    """Build a minimal LIGAttributionAuditor with enough mocks to call the method."""
    # Avoid importing the full class (which triggers gpt.py / torch imports)
    # by patching at the module level. In your project, just import directly:
    #   from proc.pipeline.llm_prompt_usage_attribution.lig_attribution_auditor import LIGAttributionAuditor
    import sys, types

    # Build a minimal stand-in if the real import is unavailable in test env
    try:
        from proc.pipeline.llm_prompt_usage_attribution.token_attribution_auditor import (
            TokenAttributionAuditor,
        )
        auditor = object.__new__(TokenAttributionAuditor)
        import logging
        auditor._logger = logging.getLogger("test")
        return auditor
    except ImportError:
        pytest.skip("LIGAttributionAuditor not importable in this environment")


def _make_predictor(instruction: str, demos=None, field_names=None):
    """Build a mock DSPy predictor."""
    predictor = MagicMock()
    predictor.signature.instructions = instruction
    predictor.demos = demos or []
    # input_fields and output_fields used by _demo_to_text
    predictor.signature.input_fields = field_names or ["email_from", "email_to", "email_body", "current_date"]
    predictor.signature.output_fields = ["reasoning", "extracted_json"]
    return predictor


def _tokens_from_prompt(prompt: str) -> list[str]:
    """
    Simulate character-level tokenization: one token per character.
    This makes the char_to_token mapping trivially verifiable.
    """
    return list(prompt)


def _label_counts(labels: list[str]) -> dict:
    from collections import Counter
    return dict(Counter(labels))


# ── Fixtures ──────────────────────────────────────────────────────────────────

INSTRUCTION = (
    "You are a scheduling assistant. Extract structured scheduling data.\n"
    "Rules:\n- duration_minutes: default 30 if not stated.\n"
    "- urgency: today > this_week > next_week > flexible."
)

ABEGPT_PROMPT = (
    "[SYSTEM] Your input fields are:\n"
    "1. `email_from`\n"
    "2. `email_to`\n"
    "3. `email_body`\n"
    "4. `current_date`\n"
    f"{INSTRUCTION}\n"
    "[USER] [[ ## email_from ## ]]\n"
    "pedro@corp.mx\n"
    "[[ ## email_to ## ]]\n"
    "alex@abe-health.com\n"
    "[[ ## email_body ## ]]\n"
    "Hi Alex, can we meet today?\n"
    "[[ ## current_date ## ]]\n"
    "2024-03-11\n"
    "Respond with [[ ## reasoning ## ]] then [[ ## extracted_json ## ]]."
)

QWEN_PROMPT = (
    "<|im_start|>system\n"
    "Your input fields are:\n"
    "1. `email_from`\n"
    f"{INSTRUCTION}\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "[[ ## email_from ## ]]\n"
    "pedro@corp.mx\n"
    "[[ ## email_to ## ]]\n"
    "alex@abe-health.com\n"
    "[[ ## email_body ## ]]\n"
    "Hi Alex, can we meet today?\n"
    "[[ ## current_date ## ]]\n"
    "2024-03-11\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAssignSegmentLabels:

    def setup_method(self):
        self.auditor = _make_auditor()
        self.tokenizer = MagicMock()  # not used by the new implementation

    def _run(self, prompt: str, instruction: str = INSTRUCTION, demos=None) -> list[str]:
        tokens = _tokens_from_prompt(prompt)
        predictor = _make_predictor(instruction, demos=demos)
        return self.auditor._assign_segment_labels(
            tokens_str=tokens,
            predictor=predictor,
            tokenizer=self.tokenizer,
            prompt_text=prompt,
        )

    # ── Coverage ──────────────────────────────────────────────────────────────

    def test_no_unknown_labels_abegpt(self):
        labels = self._run(ABEGPT_PROMPT)
        assert "unknown" not in labels, "All tokens should be labeled — no 'unknown' remaining"

    def test_no_unknown_labels_qwen(self):
        labels = self._run(QWEN_PROMPT)
        assert "unknown" not in labels

    def test_label_count_matches_token_count(self):
        labels = self._run(ABEGPT_PROMPT)
        assert len(labels) == len(ABEGPT_PROMPT), (
            "One label per character (char-level tokenization)"
        )

    # ── Wrapper tag detection ─────────────────────────────────────────────────

    def test_system_wrapper_abegpt(self):
        labels = self._run(ABEGPT_PROMPT)
        counts = _label_counts(labels)
        assert counts.get("system_wrapper", 0) > 0, (
            "[SYSTEM] and [USER] tags should be labeled as system_wrapper"
        )

    def test_system_wrapper_qwen(self):
        labels = self._run(QWEN_PROMPT)
        counts = _label_counts(labels)
        assert counts.get("system_wrapper", 0) > 0, (
            "<|im_start|> and <|im_end|> tags should be labeled as system_wrapper"
        )

    def test_system_tag_position_abegpt(self):
        """[SYSTEM] at position 0 should be system_wrapper."""
        labels = self._run(ABEGPT_PROMPT)
        system_tag = "[SYSTEM]"
        idx = ABEGPT_PROMPT.find(system_tag)
        assert labels[idx] == "system_wrapper"
        assert labels[idx + len(system_tag) - 1] == "system_wrapper"

    def test_im_start_position_qwen(self):
        """<|im_start|> at position 0 should be system_wrapper."""
        labels = self._run(QWEN_PROMPT)
        tag = "<|im_start|>"
        idx = QWEN_PROMPT.find(tag)
        assert labels[idx] == "system_wrapper"

    # ── Instruction detection ─────────────────────────────────────────────────

    def test_instruction_labeled_abegpt(self):
        labels = self._run(ABEGPT_PROMPT)
        counts = _label_counts(labels)
        assert counts.get("instruction", 0) > 0, (
            "Instruction text should be labeled — check find_and_label matching"
        )

    def test_instruction_labeled_qwen(self):
        labels = self._run(QWEN_PROMPT)
        counts = _label_counts(labels)
        assert counts.get("instruction", 0) > 0

    def test_instruction_first_char_position(self):
        """The first character of the instruction should be labeled 'instruction'."""
        labels = self._run(ABEGPT_PROMPT)
        idx = ABEGPT_PROMPT.find(INSTRUCTION[:20])
        assert idx != -1
        assert labels[idx] == "instruction"

    def test_missing_instruction_falls_to_input(self):
        """If instruction doesn't appear in prompt, tokens become 'input', not 'unknown'."""
        # Include content that won't match any specific label — some freeform text
        # that sits between the field markers and isn't a field value
        prompt = (
            "[SYSTEM] Some preamble text that is not the instruction.\n"
            "[USER] [[ ## email_body ## ]]\n"
            "Hi!\n"
            "[[ ## current_date ## ]]\n"
            "2024-03-11\n"
            "Some trailing text here that is not a field value.\n"
        )
        labels = self._run(prompt, instruction="THIS INSTRUCTION IS NOT IN THE PROMPT")
        assert "unknown" not in labels, "No 'unknown' labels — all tokens should be labeled"
        counts = _label_counts(labels)
        # 'Some preamble text' and 'Some trailing text' have no specific label → 'input'
        assert counts.get("input", 0) > 0, (
            f"Expected some 'input' tokens for unmatched content. Got: {counts}"
        )

    def test_first_line_heuristic(self):
        """Instruction matching via first-line heuristic when exact match fails."""
        # Instruction with leading whitespace that won't match exactly
        instruction = "  " + INSTRUCTION  # leading spaces not in prompt
        labels = self._run(ABEGPT_PROMPT, instruction=instruction)
        counts = _label_counts(labels)
        # Should still find instruction via first-line heuristic
        assert counts.get("instruction", 0) > 0

    # ── Field label markers ───────────────────────────────────────────────────

    def test_field_labels_detected(self):
        labels = self._run(ABEGPT_PROMPT)
        counts = _label_counts(labels)
        assert counts.get("field_label", 0) > 0

    def test_field_label_position(self):
        """[[ ## email_from ## ]] should be labeled field_label."""
        labels = self._run(ABEGPT_PROMPT)
        marker = "[[ ## email_from ## ]]"
        idx = ABEGPT_PROMPT.find(marker)
        assert labels[idx] == "field_label"
        assert labels[idx + len(marker) - 1] == "field_label"

    # ── Field value extraction ────────────────────────────────────────────────

    def test_email_from_value_labeled(self):
        labels = self._run(ABEGPT_PROMPT)
        counts = _label_counts(labels)
        assert counts.get("email_from", 0) > 0, "pedro@corp.mx should be labeled email_from"

    def test_email_to_value_labeled(self):
        labels = self._run(ABEGPT_PROMPT)
        counts = _label_counts(labels)
        assert counts.get("email_to", 0) > 0

    def test_email_body_labeled(self):
        labels = self._run(ABEGPT_PROMPT)
        counts = _label_counts(labels)
        assert counts.get("email_body", 0) > 0

    def test_current_date_labeled(self):
        labels = self._run(ABEGPT_PROMPT)
        counts = _label_counts(labels)
        assert counts.get("current_date", 0) > 0

    def test_email_body_value_position(self):
        """'Hi Alex, can we meet today?' should be labeled email_body."""
        value = "Hi Alex, can we meet today?"
        labels = self._run(ABEGPT_PROMPT)
        idx = ABEGPT_PROMPT.rfind(value)
        assert idx != -1
        assert labels[idx] == "email_body", (
            f"Expected 'email_body' at char {idx}, got '{labels[idx]}'"
        )

    def test_rfind_uses_last_occurrence(self):
        """
        If the field value appears in both SYSTEM (instruction example) and USER turn,
        rfind should label the USER turn occurrence as the field value.
        """
        # Craft a prompt where 'pedro@corp.mx' appears twice
        prompt = (
            "[SYSTEM] Example: pedro@corp.mx sends to alex@abe-health.com\n"
            f"{INSTRUCTION}\n"
            "[USER] [[ ## email_from ## ]]\n"
            "pedro@corp.mx\n"
            "[[ ## email_to ## ]]\n"
            "alex@abe-health.com\n"
            "[[ ## email_body ## ]]\n"
            "Let us meet.\n"
            "[[ ## current_date ## ]]\n"
            "2024-03-11\n"
        )
        labels = self._run(prompt)
        # The last occurrence of 'pedro@corp.mx' should be email_from
        last_idx = prompt.rfind("pedro@corp.mx")
        assert labels[last_idx] == "email_from"

    # ── Empty / edge cases ────────────────────────────────────────────────────

    def test_empty_prompt_returns_all_input(self):
        tokens = list("some tokens here")
        predictor = _make_predictor(INSTRUCTION)
        labels = self.auditor._assign_segment_labels(
            tokens_str=tokens,
            predictor=predictor,
            tokenizer=self.tokenizer,
            prompt_text="",
        )
        assert all(l == "input" for l in labels)
        assert len(labels) == len(tokens)

    def test_empty_instruction(self):
        labels = self._run(ABEGPT_PROMPT, instruction="")
        assert "unknown" not in labels

    def test_no_demos_no_crash(self):
        labels = self._run(ABEGPT_PROMPT, demos=[])
        assert len(labels) == len(ABEGPT_PROMPT)

    # ── Demo labeling ─────────────────────────────────────────────────────────

    def test_demo_labeled(self):
        demo = MagicMock()
        demo.email_from = "alice@example.com"
        demo.email_to = "bob@example.com"
        demo.email_body = "Let us meet Friday."
        demo.current_date = "2024-01-01"
        demo.reasoning = "Sender is in UTC."
        demo.extracted_json = '{"urgency": "this_week"}'

        # Build a prompt that contains the demo text
        demo_text = (
            "email_from: alice@example.com\n"
            "email_to: bob@example.com\n"
            "email_body: Let us meet Friday.\n"
            "current_date: 2024-01-01\n"
            "reasoning: Sender is in UTC.\n"
            'extracted_json: {"urgency": "this_week"}'
        )
        prompt = f"[SYSTEM] {INSTRUCTION}\n[USER] {demo_text}\n[[ ## email_from ## ]]\npedro@corp.mx\n"

        # Mock _demo_to_text to return demo_text
        self.auditor._demo_to_text = MagicMock(return_value=demo_text)

        labels = self._run(prompt, demos=[demo])
        counts = _label_counts(labels)
        assert counts.get("demo_0", 0) > 0, "Demo text should be labeled demo_0"
