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
import logging
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

LOGGER = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────
batch_size    = 64      # sequences processed in parallel
block_size    = 256     # maximum context length (tokens)
max_iters     = 5000    # training steps
eval_interval = 500     # how often to evaluate loss
learning_rate = 3e-4
device        = 'mps'
eval_iters    = 200
n_embd        = 384     # embedding dimension
n_head        = 6       # number of attention heads
n_layer       = 6       # number of transformer blocks
dropout       = 0.2

torch.manual_seed(1337)

# ─────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DATA_PATH = _os.path.join(_HERE, 'input.txt')
try:
    with open(_DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
        LOGGER.info(f"Loaded dataset: {len(text):,} characters")
except FileNotFoundError:
    # Fallback: generate a small demo text
    text = (
        "To be, or not to be, that is the question: "
        "Whether 'tis nobler in the mind to suffer "
        "The slings and arrows of outrageous fortune, "
        "Or to take arms against a sea of troubles. " * 200
    )
    LOGGER.info("Using demo text (download input.txt for full Shakespeare dataset)")

# ─────────────────────────────────────────
# Tokenizer (character-level)
# ─────────────────────────────────────────
chars    = sorted(set(text))
vocab_size = len(chars)
stoi     = {ch: i for i, ch in enumerate(chars)}   # string → int
itos     = {i: ch for i, ch in enumerate(chars)}   # int → string
encode   = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train / validation split
data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]


def get_batch(split: str):
    """Sample a random batch of (input, target) token sequences."""
    src = train_data if split == 'train' else val_data
    ix  = torch.randint(len(src) - block_size, (batch_size,))
    x   = torch.stack([src[i:i + block_size]     for i in ix])
    y   = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
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
        self.key   = nn.Linear(n_embd, head_size, bias=False)
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
        wei   = q @ k.transpose(-2, -1) * scale  # (B, T, T)

        # Causal masking: tokens can only attend to earlier positions
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v   = self.value(x)       # (B, T, head_size)
        out = wei @ v             # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """h parallel attention heads whose outputs are concatenated and projected."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)   # output projection
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
        head_size  = n_embd // n_head
        self.sa    = MultiHeadAttention(n_head, head_size)
        self.ffwd  = FeedForward(n_embd)
        self.ln1   = nn.LayerNorm(n_embd)
        self.ln2   = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))    # residual + attention
        x = x + self.ffwd(self.ln2(x))  # residual + FFN
        return x


class GPTLanguageModel(nn.Module):
    """Character-level GPT: embedding → N transformer blocks → LM head."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)   # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                         # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x    = tok_emb + pos_emb                                          # (B, T, C)
        x    = self.blocks(x)
        x    = self.ln_f(x)
        logits = self.lm_head(x)                                          # (B, T, vocab_size)

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
            logits    = logits[:, -1, :]               # last time-step
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat((idx, idx_next), dim=1)
        return idx


# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────

def train():
    model     = GPTLanguageModel().to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    LOGGER.info(f"\nModel: {num_params:.2f}M parameters | device: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    cur_time = time.time()
    for step in range(max_iters):

        if step % 10 == 0:
            LOGGER.info(f"Step {step}: {max_iters}. Spend {time.time() - cur_time:.2f}s")
            cur_time = time.time()

        if step % eval_interval == 0 or step == max_iters - 1:
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
    LOGGER.info(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

    torch.save(model.state_dict(), "gpt.pt")

    return model


if __name__ == '__main__':
    train()
