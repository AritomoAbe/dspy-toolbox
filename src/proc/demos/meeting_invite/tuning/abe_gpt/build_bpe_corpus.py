"""Build a plain-text corpus from an email JSONL dataset for BPE tokenizer training.

The script reads a JSONL file where each line is a JSON object. It extracts selected
fields, formats them into text examples, and writes one text file suitable for
training a BPE tokenizer.

Default dataset shape (detected from the provided sample):
- email_from: str
- email_to: str
- email_body: str
- current_date: str
- expected: dict (optional metadata; excluded by default)

Usage examples:
    python build_bpe_corpus.py trainset_emails_500.jsonl corpus.txt
    python build_bpe_corpus.py trainset_emails_500.jsonl corpus.txt --include-expected
    python build_bpe_corpus.py trainset_emails_500.jsonl corpus.txt --body-only

Recommended next step:
    Train your tokenizer on corpus.txt with a vocab size like 8k or 16k.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

EMAIL_START = "<|email_start|>"
EMAIL_END = "<|email_end|>"
FIELD_SEP = "\n"

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an email JSONL dataset into a plain-text corpus for BPE training."
    )
    parser.add_argument("input_jsonl", type=Path, help="Path to input JSONL dataset")
    parser.add_argument("output_txt", type=Path, help="Path to output corpus .txt file")
    parser.add_argument(
        "--include-expected",
        action="store_true",
        help="Include the 'expected' JSON payload in the corpus as text",
    )
    parser.add_argument(
        "--body-only",
        action="store_true",
        help="Write only email bodies, separated by blank lines and end markers",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase all text before writing the corpus",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate identical rendered examples",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit on the number of records to process",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)

    # Normalize line endings and strip trailing spaces on each line.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def render_record(record: dict[str, Any], include_expected: bool, body_only: bool) -> str:
    email_from = clean_text(record.get("email_from", ""))
    email_to = clean_text(record.get("email_to", ""))
    email_body = clean_text(record.get("email_body", ""))
    current_date = clean_text(record.get("current_date", ""))
    expected = record.get("expected")

    if body_only:
        parts = [EMAIL_START, email_body, EMAIL_END]
        return FIELD_SEP.join(part for part in parts if part)

    parts = [
        EMAIL_START,
        f"From: {email_from}" if email_from else "",
        f"To: {email_to}" if email_to else "",
        f"Date: {current_date}" if current_date else "",
        "",
        email_body,
    ]

    if include_expected and expected is not None:
        expected_text = clean_text(expected)
        if expected_text:
            parts.extend(["", "Expected:", expected_text])

    parts.append(EMAIL_END)
    return FIELD_SEP.join(part for part in parts if part != "")


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            yield obj


def main() -> int:
    args = parse_args()

    if not args.input_jsonl.exists():
        LOGGER.info(f"Input file not found: {args.input_jsonl}", file=sys.stderr)
        return 1

    rendered_examples: list[str] = []
    seen: set[str] = set()
    processed = 0

    for record in load_records(args.input_jsonl):
        rendered = render_record(
            record,
            include_expected=args.include_expected,
            body_only=args.body_only,
        )
        if args.lowercase:
            rendered = rendered.lower()
        if not rendered.strip():
            continue
        if args.dedupe:
            if rendered in seen:
                continue
            seen.add(rendered)
        rendered_examples.append(rendered)
        processed += 1
        if args.max_records is not None and processed >= args.max_records:
            break

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    corpus = "\n\n".join(rendered_examples).strip() + "\n"
    args.output_txt.write_text(corpus, encoding="utf-8")

    total_chars = len(corpus)
    total_lines = corpus.count("\n")
    LOGGER.info(f"Wrote {len(rendered_examples)} examples to {args.output_txt}")
    LOGGER.info(f"Characters: {total_chars:,}")
    LOGGER.info(f"Lines: {total_lines:,}")
    LOGGER.info(
        "Mode: "
        + ("body-only" if args.body_only else "structured email")
        + (" + expected" if args.include_expected else "")
        + (" + lowercase" if args.lowercase else "")
        + (" + dedupe" if args.dedupe else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
