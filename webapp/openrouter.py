"""
Standalone OpenRouter API helper.

Calls Llama 3.1 8B Instruct via OpenRouter's OpenAI-compatible API.
Reads the API key from the OPENROUTER__DEMO env var, falling back to
SECRETS/openrouter__mats.key file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import openai

KEY_PATH = Path(__file__).parent.parent / "SECRETS" / "openrouter__mats.key"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"


def _read_key() -> str:
    """Read OpenRouter API key from env var or file."""
    key = os.environ.get("OPENROUTER_DEMO", "").strip()
    if key:
        return key
    try:
        key = KEY_PATH.read_text().strip()
        if key:
            return key
    except FileNotFoundError:
        pass
    raise RuntimeError("No OpenRouter API key found in OPENROUTER_DEMO env var or file")


def query_openrouter(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: int = 42,
) -> str:
    """
    Send a chat completion request to OpenRouter and return the response text.
    """
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_read_key(),
    )

    # OpenRouter supports OpenAI-compatible params; top_k via extra_body
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        extra_body={"top_k": top_k},
    )

    return response.choices[0].message.content or ""


@dataclass
class MockOutput:
    """Mimics the shape that verify_outputs expects from a single output entry."""

    token_ids: list[int] = field(default_factory=list)


@dataclass
class MockRequestOutput:
    """Mimics vLLM RequestOutput with prompt_token_ids and outputs[0].token_ids."""

    prompt_token_ids: list[int] = field(default_factory=list)
    outputs: list[MockOutput] = field(default_factory=list)
