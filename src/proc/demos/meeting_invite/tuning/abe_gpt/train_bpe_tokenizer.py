"""Train a BPE tokenizer from a plain-text email corpus.

Examples:
    python train_bpe_tokenizer.py email_corpus.txt out/tokenizer_8k --vocab-size 8000
    python train_bpe_tokenizer.py email_corpus.txt out/tokenizer_16k --vocab-size 16000

Outputs:
    - tokenizer.json           Full Hugging Face tokenizer definition
    - vocab.json               BPE vocab
    - merges.txt               BPE merges
    - special_tokens.json      Special tokens used during training
    - tokenizer_config.json    Small metadata/config file

Notes:
    - Uses ByteLevel BPE, which is a strong default for messy email text.
    - Keeps special email boundary tokens if they are present in the corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

_BOS_TOKEN: str = '<|bos|>'
_EOS_TOKEN: str = '<|eos|>'
_VOCAB_SIZE_8K: int = 8000
_VOCAB_SIZE_16K: int = 16000

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from email_corpus.txt")
    parser.add_argument("input_corpus", type=Path, help="Path to plain-text corpus, e.g. email_corpus.txt")
    parser.add_argument("output_dir", type=Path, help="Directory where tokenizer artifacts will be written")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=_VOCAB_SIZE_8K,
        choices=(_VOCAB_SIZE_8K, _VOCAB_SIZE_16K),
        help=f"Tokenizer vocabulary size (default: {_VOCAB_SIZE_8K})",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency for inclusion in vocab (default: 2)",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Print sample tokenization stats after training",
    )
    return parser.parse_args()


def require_tokenizers() -> None:
    try:
        import tokenizers  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "This script requires the 'tokenizers' package. Install it with:\n"
            "  pip install tokenizers"
        ) from exc


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input corpus not found: {path}")
    return path.read_text(encoding="utf-8")


def detect_special_tokens(text: str) -> list[str]:
    tokens: list[str] = ["[UNK]", "<|pad|>", _BOS_TOKEN, _EOS_TOKEN]
    for token in ("<|email_start|>", "<|email_end|>"):
        if token in text:
            tokens.append(token)
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for token in tokens:
        if token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped


def train_tokenizer(input_corpus: Path, output_dir: Path, vocab_size: int, min_frequency: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    text = read_text(input_corpus)
    special_tokens = detect_special_tokens(text)

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train([str(input_corpus)], trainer)

    bos = _BOS_TOKEN if _BOS_TOKEN in special_tokens else None
    eos = _EOS_TOKEN if _EOS_TOKEN in special_tokens else None
    if bos and eos:
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{bos} $A {eos}",
            pair=f"{bos} $A {eos} $B:1 {eos}:1",
            special_tokens=[
                (bos, tokenizer.token_to_id(bos)),
                (eos, tokenizer.token_to_id(eos)),
            ],
        )

    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    model = tokenizer.model
    try:
        model.save(str(output_dir), prefix="")
    except TypeError:
        model.save(str(output_dir))

    config = {
        "model_type": "ByteLevelBPE",
        "vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "input_corpus": str(input_corpus),
        "files": {
            "tokenizer_json": "tokenizer.json",
            "vocab": "vocab.json",
            "merges": "merges.txt",
        },
    }
    (output_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "special_tokens.json").write_text(json.dumps(special_tokens, indent=2), encoding="utf-8")

    return tokenizer, special_tokens


def chunk_text(text: str, max_len: int = 2000) -> Iterable[str]:
    start = 0
    while start < len(text):
        yield text[start:start + max_len]
        start += max_len


def print_stats(tokenizer, text: str) -> None:
    email_sample = (
        "<|email_start|>\nFrom: alice@example.com\nTo: bob@example.com\n"
        "Please review the attached deck.\n<|email_end|>"
    )
    pieces = [
        "Subject: Meeting tomorrow at 10\nCould we move it to 11?",
        email_sample,
        text[:1000] if text else "",
    ]
    LOGGER.info("\nSample tokenization stats:")
    for i, piece in enumerate(pieces, start=1):
        if not piece:
            continue
        enc = tokenizer.encode(piece)
        n_chars = len(piece)
        n_tokens = len(enc.ids)
        chars_per_token = n_chars / max(n_tokens, 1)
        LOGGER.info(
            f"  Sample {i}: chars={n_chars:4d}, tokens={n_tokens:4d}, "
            f"chars/token={chars_per_token:.2f}"
        )
        preview = enc.tokens[:24]
        LOGGER.info(f"    Tokens preview: {preview}")


def _log_training_summary(args: argparse.Namespace, tokenizer, special_tokens: list[str]) -> None:
    LOGGER.info("Tokenizer training complete.")
    LOGGER.info(f"Input corpus:        {args.input_corpus}")
    LOGGER.info(f"Output directory:    {args.output_dir}")
    LOGGER.info(f"Requested vocab:     {args.vocab_size}")
    LOGGER.info(f"Actual vocab:        {tokenizer.get_vocab_size()}")
    LOGGER.info(f"Min frequency:       {args.min_frequency}")
    LOGGER.info(f"Special tokens:      {special_tokens}")
    LOGGER.info("Artifacts written: tokenizer.json, vocab.json, merges.txt, tokenizer_config.json")


def main() -> None:
    args = parse_args()
    require_tokenizers()

    text = read_text(args.input_corpus)
    tokenizer, special_tokens = train_tokenizer(
        input_corpus=args.input_corpus,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    _log_training_summary(args, tokenizer, special_tokens)

    if args.show_stats:
        print_stats(tokenizer, text)


if __name__ == "__main__":
    main()
