"""
GPT → OpenAI-compatible API server
====================================
Wraps the custom GPTLanguageModel in a minimal FastAPI endpoint that speaks
the OpenAI /v1/chat/completions format. DSPy can then connect to it via:

    lm = dspy.LM(
        model="openai/gpt-abe",
        api_base="http://localhost:8000",
        api_key="none",
    )
    dspy.configure(lm=lm)

Run
---
    python gpt_server.py                         # starts on port 8000
    python gpt_server.py --checkpoint gpt.pt     # load a saved checkpoint
"""

import argparse
import logging
import time
from typing import Optional

from proc.base.timing import timed

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ── Import the GPT model definition ───────────────────────────────────────────
# Adjust this import to match where gpt.py lives in your project.
from gpt import (
    GPTLanguageModel,
    decode,
    device,
    block_size, enc_dec,
)

LOGGER = logging.getLogger(__name__)

# ── Load model ─────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Optional[str] = None) -> GPTLanguageModel:
    model = GPTLanguageModel().to(device)
    if checkpoint_path:
        with timed(f"load_model checkpoint={checkpoint_path}", logger=LOGGER):
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
        LOGGER.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        LOGGER.info("No checkpoint — using randomly initialized weights (for testing only)")
    model.eval()
    return model


# ── FastAPI app ────────────────────────────────────────────────────────────────

app  = FastAPI(title="GPT Abe API")
_model: Optional[GPTLanguageModel] = None


# ── OpenAI-compatible request/response schemas ────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-abe"
    messages: list[Message]
    max_tokens: int = 200
    temperature: float = 1.0

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-gpt"
    object: str = "chat.completion"
    created: int = 0
    model: str = "gpt-abe"
    choices: list[Choice]
    usage: Usage

def _parse_gpt_output(text: str) -> dict:
    """
    Parse the GPT's training-format output into a flat dict.

    The model generates text like:
        TIMEZONE: Europe/Paris
        DURATION: 60
        URGENCY: NEXT_WEEK
        FLEXIBILITY: FLEXIBLE
        WINDOWS: Tuesday lunchtime
        TOPIC: retrospective

    We extract these fields and return them as a dict ready for JSON serialisation.
    Unknown or missing fields fall back to safe defaults.
    """
    fields = {
        "sender_iana_timezone": "UNKNOWN",
        "duration_minutes": 30,
        "urgency": "FLEXIBLE",
        "flexibility": "FLEXIBLE",
        "preferred_windows": [],
        "meeting_topic": "",
    }

    key_map = {
        "TIMEZONE":    "sender_iana_timezone",
        "DURATION":    "duration_minutes",
        "URGENCY":     "urgency",
        "FLEXIBILITY": "flexibility",
        "WINDOWS":     "preferred_windows",
        "TOPIC":       "meeting_topic",
    }

    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        raw_key, _, raw_val = line.partition(":")
        raw_key = raw_key.strip().upper()
        raw_val = raw_val.strip()

        if raw_key not in key_map:
            continue

        field = key_map[raw_key]

        if field == "duration_minutes":
            try:
                fields[field] = int(raw_val)
            except ValueError:
                pass
        elif field == "preferred_windows":
            if raw_val.lower() in ("none", ""):
                fields[field] = []
            else:
                # Each window separated by " | ", e.g. "Tuesday lunchtime | Friday morning"
                windows = []
                for part in raw_val.split("|"):
                    part = part.strip()
                    if part:
                        tokens = part.split()
                        w: dict = {}
                        # Heuristic: first token looks like a day name
                        days = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
                        if tokens and tokens[0].lower() in days:
                            w["day_of_week"] = tokens[0].capitalize()
                            w["time_of_day"] = " ".join(tokens[1:]) if len(tokens) > 1 else ""
                        else:
                            w["time_of_day"] = part
                        # Timezone in parens e.g. "(Europe/Paris)"
                        import re as _re
                        tz_match = _re.search(r'\(([^)]+)\)', part)
                        if tz_match:
                            w["iana_timezone"] = tz_match.group(1)
                        windows.append(w)
                fields[field] = windows
        else:
            fields[field] = raw_val

    return fields


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@app.post("/chat/completions", response_model=ChatCompletionResponse)   # litellm omits /v1 prefix
def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Minimal OpenAI-compatible endpoint.

    Concatenates all message contents into a single prompt string,
    encodes with the character-level tokenizer, runs model.generate(),
    and returns the decoded continuation.
    """
    global _model

    # Flatten all messages into a single prompt
    prompt = "\n".join(m.content for m in request.messages)

    ids = enc_dec.encode(prompt)
    if not ids:
        LOGGER.error(f'Failed to encode prompt: {prompt}')
        ids = [0]   # fallback to first token

    context = torch.tensor([ids], dtype=torch.long, device=device)

    shape_before = context.shape
    # Crop to model's context window
    context = context[:, -block_size:]
    shape_after = context.shape
    if shape_before != shape_after:
        LOGGER.warning(f'Prompt context was truncated from {shape_before} to {shape_after}')

    # Generate
    with timed(f"model.generate (max_new_tokens={request.max_tokens})", logger=LOGGER):
        with torch.no_grad():
            output_ids = _model.generate(
                context,
                max_new_tokens=request.max_tokens,
            )

    # Decode only the newly generated tokens (skip the prompt)
    new_ids   = output_ids[0, context.shape[1]:].tolist()
    generated = decode(new_ids)

    # Parse the GPT's training-format output into structured fields
    extracted = _parse_gpt_output(generated)

    # Wrap in DSPy's expected response format.
    # ChatAdapter looks for a JSON object with the signature's output field names.
    # JSONAdapter looks for 'extracted_json' containing the actual payload.
    import json as _json
    dspy_response = _json.dumps({
        "reasoning": f"Generated by AbeGPT. Raw output: {generated[:120].strip()}",
        "extracted_json": _json.dumps(extracted),
    })

    n_prompt_tokens = context.shape[1]
    n_new_tokens    = len(new_ids)

    return ChatCompletionResponse(
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=dspy_response),
            )
        ],
        usage=Usage(
            prompt_tokens=n_prompt_tokens,
            completion_tokens=n_new_tokens,
            total_tokens=n_prompt_tokens + n_new_tokens,
        ),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": "gpt-abe"}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a saved model checkpoint (.pt file)")
    parser.add_argument("--port", type=int, default=8013)
    args = parser.parse_args()

    _model = load_model(args.checkpoint)
    print(f"\nGPT API running at http://localhost:{args.port}")
    print("Connect DSPy with:")
    print(f'  lm = dspy.LM("openai/gpt-abe", '
          f'api_base="http://localhost:{args.port}", api_key="none")\n')
    uvicorn.run(app, host="0.0.0.0", port=args.port)