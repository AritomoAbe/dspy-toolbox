"""
Simple GPT Implementation
Paper: "Attention Is All You Need" (Vaswani et al., 2017)

Architecture:
  - Character-level tokenization
  - Token + Positional Embeddings
  - Multi-Head Causal Self-Attention
  - Feed-Forward Blocks
  - Layer Norm + Residual Connections
  - Language Model Head
"""
import abc
import logging
import time
from abc import ABC
from pathlib import Path
from typing import Optional

from proc.base.timing import timed

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

LOGGER = logging.getLogger(__name__)

_N_EMBD_SMALL: int = 192
_N_EMBD_MEDIUM: int = 384
_BLOCK_SMALL: int = 512
_BLOCK_MEDIUM: int = 1024
_BLOCK_LARGE: int = 2048
_BATCH_LARGE: int = 16
_BATCH_SMALL: int = 4
_EVAL_INTERVAL_DEFAULT: int = 500
_SEED: int = 1337
_INIT_STD: float = 0.02
_TRAIN_SPLIT: float = 0.9
_PARAMS_SCALE: float = 1e6
_SAMPLE_MAX_TOKENS: int = 500

# ─────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────


class ModelConfig(ABC):
    @abc.abstractmethod
    def get_batch(self) -> int:
        ...

    @abc.abstractmethod
    def get_block_size(self) -> int:
        ...

    @abc.abstractmethod
    def get_version_post_fix(self) -> str:
        ...

    @abc.abstractmethod
    def get_n_embd(self) -> int:
        ...

    @abc.abstractmethod
    def get_eval_interval(self) -> int:
        ...


class ModelConfigV101(ModelConfig):

    def get_batch_size(self) -> int:
        ...

    def get_n_embd(self) -> int:
        return _N_EMBD_MEDIUM

    def get_batch(self) -> int:
        return _BATCH_LARGE

    def get_block_size(self) -> int:
        return _BLOCK_MEDIUM

    def get_version_post_fix(self) -> str:
        return "_v1_0_1"

    def get_eval_interval(self) -> int:
        return _EVAL_INTERVAL_DEFAULT


class ModelConfigV102(ModelConfig):

    def get_n_embd(self) -> int:
        return _N_EMBD_SMALL

    def get_batch(self) -> int:
        return _BATCH_SMALL

    def get_block_size(self) -> int:
        return _BLOCK_SMALL

    def get_version_post_fix(self) -> str:
        return "_v1_0_2"

    def get_eval_interval(self) -> int:
        return _EVAL_INTERVAL_DEFAULT


class ModelConfigV103(ModelConfig):

    def get_n_embd(self) -> int:
        return _N_EMBD_SMALL

    def get_batch(self) -> int:
        return _BATCH_SMALL

    def get_block_size(self) -> int:
        return _BLOCK_MEDIUM

    def get_version_post_fix(self) -> str:
        return "_v1_0_3"

    def get_eval_interval(self) -> int:
        return _EVAL_INTERVAL_DEFAULT


class ModelConfigV104(ModelConfig):

    def get_n_embd(self) -> int:
        return _N_EMBD_SMALL

    def get_batch(self) -> int:
        return _BATCH_SMALL

    def get_block_size(self) -> int:
        return _BLOCK_LARGE

    def get_version_post_fix(self) -> str:
        return "_v1_0_4"

    def get_eval_interval(self) -> int:
        return _EVAL_INTERVAL_DEFAULT


# params = ModelConfigV101()
# params = ModelConfigV102()
# params = ModelConfigV103()
params = ModelConfigV104()

VERSION_POST_FIX = params.get_version_post_fix()
batch_size = params.get_batch()                  # sequences processed in parallel
block_size = params.get_block_size()             # maximum context length (tokens)
max_iters = 5000    # training steps
eval_interval = params.get_eval_interval()     # how often to evaluate loss
learning_rate = 3e-4
device = 'mps'
eval_iters = 200
n_embd = params.get_n_embd()                 # embedding dimension
n_head = 6       # number of attention heads
n_layer = 6       # number of transformer blocks
dropout = 0.2

torch.manual_seed(_SEED)

# ─────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────
path = Path(__file__).parent / f"input{VERSION_POST_FIX}.txt"
try:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError as e:
    LOGGER.error(f'Can not load dataset from: {path}')
    raise e
LOGGER.info(f"Loaded dataset: {len(text):,} characters")

# ─────────────────────────────────────────
# Tokenizer (character-level)
# ─────────────────────────────────────────


class EncodeDecode(ABC):
    @abc.abstractmethod
    def encode(self, s: str) -> list[int]:
        ...

    @abc.abstractmethod
    def decode(self, tokens: list[int]) -> str:
        ...

    @abc.abstractmethod
    def get_vocab_size(self) -> int:
        ...


class EncodeDecodeV101(EncodeDecode):

    def __init__(self, chars) -> None:
        self._chars = chars
        self._stoi = {ch: i for i, ch in enumerate(chars)}  # string → int
        self._itos = {i: ch for i, ch in enumerate(chars)}  # int → string

    def encode(self, s: str) -> list[int]:
        result = list()
        for c in s:
            try:
                result.append(self._stoi[c])
            except KeyError as e:
                LOGGER.error(f"Encoding error: {e}")
        return result

    def decode(self, tokens: list[int]) -> str:
        return ''.join(self._itos[i] for i in tokens)

    def get_vocab_size(self) -> int:
        return len(self._chars)


class EncodeDecodeV102(EncodeDecode):

    def __init__(self) -> None:
        self._bpe = tiktoken.get_encoding("cl100k_base")

    def get_vocab_size(self) -> int:
        return self._bpe.n_vocab

    def encode(self, s: str) -> list[int]:
        return self._bpe.encode(s)

    def decode(self, tokens: list[int]) -> str:
        return self._bpe.decode(tokens)


class EncodeDecodeV103:
    """
    Expected directory layout (produced by train_bpe_tokenizer.py):
        tokenizer_8k/
          tokenizer.json
          vocab.json
          merges.txt
          special_tokens.json
          tokenizer_config.json
    """

    def __init__(self, tokenizer_dir: Optional[str] = None) -> None:
        if tokenizer_dir is None:
            tokenizer_dir = Path(__file__).parent / "tokenizer_8k"

        self._tokenizer_dir = Path(tokenizer_dir)
        self._tokenizer_path = self._tokenizer_dir / "tokenizer.json"

        if not self._tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {self._tokenizer_path}. "
                f"Expected a directory created by train_bpe_tokenizer.py"
            )

        self._bpe = Tokenizer.from_file(str(self._tokenizer_path))

    def get_vocab_size(self) -> int:
        return self._bpe.get_vocab_size()

    def encode(self, s: str) -> list[int]:
        return self._bpe.encode(s).ids

    def decode(self, tokens: list[int]) -> str:
        return self._bpe.decode(list(tokens))


# enc_dec = EncodeDecodeV101(sorted(set(text)))
# enc_dec = EncodeDecodeV102()
enc_dec = EncodeDecodeV103()

vocab_size = enc_dec.get_vocab_size()
encode = enc_dec.encode
decode = enc_dec.decode

# Train / validation split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(_TRAIN_SPLIT * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    """Sample a random batch of (input, target) token sequences."""
    src = train_data if split == 'train' else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i + block_size] for i in ix])
    y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: nn.Module) -> dict:
    """Evaluate mean loss over several batches for train and val splits."""
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ─────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────


class Head(nn.Module):
    """One head of causal (masked) self-attention."""

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular mask stored as a non-parameter buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Scaled dot-product attention scores
        scale = k.shape[-1] ** -0.5
        wei = q @ k.transpose(-2, -1) * scale  # (B, T, T)

        # Causal masking: tokens can only attend to earlier positions
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)       # (B, T, head_size)
        out = wei @ v             # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """h parallel attention heads whose outputs are concatenated and projected."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)   # output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → ReLU → Linear (4× wider hidden layer)."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: LayerNorm → MultiHeadAttn → LayerNorm → FFN (pre-norm variant)."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))    # residual + attention
        x = x + self.ffwd(self.ln2(x))  # residual + FFN
        return x


class GPTLanguageModel(nn.Module):
    """Character-level GPT: embedding → N transformer blocks → LM head."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)   # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=float(), std=_INIT_STD)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=float(), std=_INIT_STD)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                       # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))     # (T, C)
        x = tok_emb + pos_emb                                                        # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                                        # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Auto-regressively sample new tokens given a context."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]            # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]               # last time-step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────


def train():
    model = GPTLanguageModel().to(device)
    num_params = sum(p.numel() for p in model.parameters()) / _PARAMS_SCALE
    LOGGER.info(f"\nModel: {num_params:.2f}M parameters | device: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    cur_time = time.time()
    for step in range(max_iters):

        if step % 10 == 0:
            LOGGER.info(f"Step {step}: {max_iters}. Spend {time.time() - cur_time:.2f}s")
            cur_time = time.time()

        if step % eval_interval == 0 or step == max_iters - 1:
            with timed(f"estimate_loss (step={step})"):
                losses = estimate_loss(model)
            LOGGER.info(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Sample from the trained model
    LOGGER.info("\n─── Generated text ───")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with timed(f"model.generate (max_new_tokens={_SAMPLE_MAX_TOKENS})"):
        generated_ids = model.generate(context, max_new_tokens=_SAMPLE_MAX_TOKENS)[0].tolist()
    LOGGER.info(decode(generated_ids))

    torch.save(model.state_dict(), f"gpt{VERSION_POST_FIX}.pt")

    return model


if __name__ == '__main__':
    train()
