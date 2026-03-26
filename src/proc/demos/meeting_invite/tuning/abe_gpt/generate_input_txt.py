"""
Convert email JSONL dataset → input.txt for GPT training
=========================================================

Formats each email as a structured training document so the character-level
GPT learns the vocabulary, patterns, and structure of scheduling emails AND
their extracted fields.

The format is deliberately simple and repetitive — this is what a character-
level model needs. Each record becomes a self-contained block separated by
a delimiter the model can learn to recognize.

Output format per example:
    ===
    FROM: frank@logistics.de
    TO: ceo@startup.com
    DATE: 2024-01-08
    ---
    Hi Jordan,

    I'd love to schedule a 1:1 in the coming weeks...
    ---
    TIMEZONE: America/Chicago
    DURATION: 60
    URGENCY: FLEXIBLE
    FLEXIBILITY: FLEXIBLE
    WINDOWS: []
    TOPIC: 1:1
    ===

Usage
-----
    python generate_input_txt.py
    python generate_input_txt.py --input emails.jsonl --output input.txt --repeats 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_RANDOM_SEED: int = 42
_MIN_CORPUS_CHARS: int = 100_000

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

LOGGER = logging.getLogger(__name__)


def format_windows(windows: list) -> str:
    """Render preferred_windows as a compact readable string."""
    if not windows:
        return "none"
    parts = []
    for w in windows:
        tokens = []
        day_of_week = w.get("day_of_week")
        if day_of_week:
            tokens.append(day_of_week)
        time_of_day = w.get("time_of_day")
        if time_of_day:
            tokens.append(time_of_day)
        iana_timezone = w.get("iana_timezone")
        if iana_timezone:
            tokens.append(f"({iana_timezone})")
        parts.append(" ".join(tokens))
    return " | ".join(parts)


def format_example(record: dict) -> str:
    """Convert one JSONL record into a training document block."""
    expected = record.get("expected", {})

    lines = [
        "===",
        f"FROM: {record.get('email_from', '')}",
        f"TO: {record.get('email_to', '')}",
        f"DATE: {record.get('current_date', '')}",
        "---",
        record.get("email_body", "").strip(),
        "---",
        f"TIMEZONE: {expected.get('sender_iana_timezone', 'UNKNOWN')}",
        f"DURATION: {expected.get('duration_minutes', '')}",
        f"URGENCY: {expected.get('urgency', '')}",
        f"FLEXIBILITY: {expected.get('flexibility', '')}",
        f"WINDOWS: {format_windows(expected.get('preferred_windows', []))}",
        f"TOPIC: {expected.get('meeting_topic', '')}",
        "===",
    ]
    return "\n".join(lines)


def generate(
    input_path: Path,
    output_path: Path,
    repeats: int = 1,
    shuffle: bool = True,
) -> None:
    # Load all records
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        LOGGER.error("ERROR: No records found in input file.")
        sys.exit(1)

    LOGGER.info(f"Loaded {len(records)} records from {input_path}")

    # Format all examples
    blocks = [format_example(r) for r in records]

    # Repeat the corpus — small datasets need multiple passes so the model
    # sees enough examples to learn patterns without memorising word-for-word.
    if shuffle and repeats > 1:
        import random
        random.seed(_RANDOM_SEED)

    corpus_blocks = []
    for i in range(repeats):
        if shuffle and repeats > 1:
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            corpus_blocks.extend(blocks[j] for j in indices)
        else:
            corpus_blocks.extend(blocks)

    joined = "\n\n".join(corpus_blocks)
    corpus = f"{joined}\n"

    output_path.write_text(corpus, encoding="utf-8")

    n_chars = len(corpus)
    n_tokens_approx = n_chars // 4  # rough char-level estimate
    LOGGER.info(f"Written {len(corpus_blocks)} blocks → {output_path}")
    LOGGER.info(f"  Characters : {n_chars:,}")
    LOGGER.info(f"  Tokens~    : {n_tokens_approx:,}  (rough estimate)")
    LOGGER.info(f"  Unique chars: {len(set(corpus))}")

    if n_chars < _MIN_CORPUS_CHARS:
        LOGGER.info(
            "\n  Corpus is small (<100K chars). Consider --repeats 5 or more\n"
            "     to give the character-level model enough signal to train on."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert email JSONL to GPT training corpus"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("emails.jsonl"),
        help="Path to the input JSONL file (default: emails.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("input_v0_0_0.txt"),
        help="Path to write input.txt (default: input.txt)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="How many times to repeat the corpus (default: 5). "
             "Small datasets need higher values.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle between repeats (order preserved)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        LOGGER.error(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    generate(
        input_path=args.input,
        output_path=args.output,
        repeats=args.repeats,
        shuffle=not args.no_shuffle,
    )


if __name__ == "__main__":
    main()
