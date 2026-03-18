"""
FastAPI server wrapping the inference verification pipeline.

Provides endpoints for TEE-based generation, verification, and classification
of LLM outputs using Gumbel Likelihood Scores (GLS).
"""

import os
import sys
import json
import math
import time
import random
from pathlib import Path

FAUX_MODE = os.environ.get("FAUX_MODE", "").lower() in ("1", "true", "yes")

if not FAUX_MODE:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from ui import get_ui_html

if not FAUX_MODE:
    import torch
    from transformers import AutoTokenizer
    from inference_verification.generate import (
        GenerationConfig,
        load_prompts,
        generate_with_vllm,
    )
    from inference_verification.verify import (
        VerificationConfig,
        verify_outputs,
        classify_tokens,
    )
    from openrouter import MockRequestOutput, MockOutput

from openrouter import query_openrouter

app = FastAPI(
    title="Inference Verification API",
    description="Verifies LLM outputs for model weight exfiltration detection",
)


# --- Request / Response models ---

class VerifyRequest(BaseModel):
    """Request body for /verify endpoint."""
    n_prompts: int = Field(default=10, description="Number of prompts to sample from dataset")
    max_tokens: int = Field(default=100, description="Max tokens to generate per prompt")
    config: Optional[dict] = Field(default=None, description="Override generation/verification config")


class QueryRequest(BaseModel):
    """Request body for /query endpoint (OpenRouter only)."""
    prompt: str = Field(description="User prompt to send to OpenRouter")
    max_tokens: int = Field(default=100, description="Max tokens to generate")
    temperature: float = Field(default=1.0)
    top_k: int = Field(default=50)
    top_p: float = Field(default=0.95)
    seed: int = Field(default=42)


class QueryResponse(BaseModel):
    """Response from /query endpoint."""
    response_text: str
    prompt: str


class VerifyTextRequest(BaseModel):
    """Request body for /verify-text endpoint — verify already-queried text."""
    prompt: str = Field(description="Original prompt (needed for tokenization context)")
    response_text: str = Field(description="LLM response text to verify")
    temperature: float = Field(default=1.0)
    top_k: int = Field(default=50)
    top_p: float = Field(default=0.95)
    seed: int = Field(default=42)
    gls_threshold: float = Field(default=-5.0)
    logit_rank_threshold: int = Field(default=10)


class TokenResult(BaseModel):
    gls_score: Optional[float] = None
    logit_rank: int
    classification: str
    token_text: str = ""


class VerifyResponse(BaseModel):
    model_name: str
    seed: int
    n_prompts: int
    total_tokens: int
    num_safe: int
    num_suspicious: int
    num_dangerous: int
    safe_pct: float
    suspicious_pct: float
    dangerous_pct: float
    gls_threshold: float
    logit_rank_threshold: int
    tokens: list[TokenResult]
    generated_text: str = ""


# --- Default config ---

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_SEED = 42
DEFAULT_GLS_THRESHOLD = -5.0
DEFAULT_LOGIT_RANK_THRESHOLD = 10


# --- Faux mode helpers ---

FAUX_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In a world where artificial intelligence continues to evolve, "
    "researchers are exploring new ways to ensure safety and alignment. "
    "Large language models have shown remarkable capabilities in understanding "
    "and generating human-like text across many domains."
)

def _faux_response(
    text: str,
    *,
    seed: int = 42,
    n_prompts: int = 1,
    gls_threshold: float = DEFAULT_GLS_THRESHOLD,
    logit_rank_threshold: int = DEFAULT_LOGIT_RANK_THRESHOLD,
) -> VerifyResponse:
    """Generate a fake VerifyResponse with mostly-safe tokens."""
    rng = random.Random(seed)
    words = text.split()
    # Treat each word (with its leading space) as a "token" for display
    token_texts = []
    for i, w in enumerate(words):
        token_texts.append((" " + w) if i > 0 else w)

    tokens = []
    for t_text in token_texts:
        roll = rng.random()
        if roll < 0.80:
            gls = rng.uniform(0.5, 8.0)
            rank = rng.randint(0, 3)
            cls = "safe"
        elif roll < 0.93:
            gls = rng.uniform(-6.0, -3.0)
            rank = rng.randint(1, logit_rank_threshold)
            cls = "suspicious"
        else:
            gls = rng.uniform(-10.0, -5.0)
            rank = rng.randint(logit_rank_threshold + 1, 50)
            cls = "dangerous"
        tokens.append(TokenResult(
            gls_score=round(gls, 4),
            logit_rank=rank,
            classification=cls,
            token_text=t_text,
        ))

    total = len(tokens)
    num_safe = sum(1 for t in tokens if t.classification == "safe")
    num_suspicious = sum(1 for t in tokens if t.classification == "suspicious")
    num_dangerous = sum(1 for t in tokens if t.classification == "dangerous")

    return VerifyResponse(
        model_name=DEFAULT_MODEL + " (faux)",
        seed=seed,
        n_prompts=n_prompts,
        total_tokens=total,
        num_safe=num_safe,
        num_suspicious=num_suspicious,
        num_dangerous=num_dangerous,
        safe_pct=round(num_safe / total * 100, 2) if total > 0 else 0,
        suspicious_pct=round(num_suspicious / total * 100, 2) if total > 0 else 0,
        dangerous_pct=round(num_dangerous / total * 100, 2) if total > 0 else 0,
        gls_threshold=gls_threshold,
        logit_rank_threshold=logit_rank_threshold,
        tokens=tokens,
        generated_text=text,
    )


# --- Helpers (real mode only) ---

if not FAUX_MODE:
    def _build_verify_response(
        ver_cfg: VerificationConfig,
        verification_results: list[dict],
        gen_token_ids: list[int],
        tokenizer,
        *,
        model_name: str,
        seed: int,
        n_prompts: int,
    ) -> VerifyResponse:
        """Shared logic to build a VerifyResponse from verification results."""
        classification = classify_tokens(
            verification_results,
            gls_threshold=ver_cfg.gls_threshold,
            logit_rank_threshold=ver_cfg.logit_rank_threshold,
        )

        tokens = []
        for i, result in enumerate(verification_results):
            gls = result["sampled_gumbel_scores"]
            if isinstance(gls, torch.Tensor):
                gls = gls.item()
            token_text = tokenizer.decode([gen_token_ids[i]]) if i < len(gen_token_ids) else ""
            gls_val = float(gls)
            cls = classification["classifications"][i].value
            # First token is always unverifiable (no prior context), force safe
            if i == 0 and cls != "safe":
                cls = "safe"
                classification["num_safe"] += 1
                if classification["classifications"][i].value == "suspicious":
                    classification["num_suspicious"] -= 1
                elif classification["classifications"][i].value == "dangerous":
                    classification["num_dangerous"] -= 1
            tokens.append(TokenResult(
                gls_score=gls_val if math.isfinite(gls_val) else None,
                logit_rank=int(result["logit_rank"]),
                classification=cls,
                token_text=token_text,
            ))

        total = len(tokens)
        num_safe = classification["num_safe"]
        num_suspicious = classification["num_suspicious"]
        num_dangerous = classification["num_dangerous"]
        generated_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

        return VerifyResponse(
            model_name=model_name,
            seed=seed,
            n_prompts=n_prompts,
            total_tokens=total,
            num_safe=num_safe,
            num_suspicious=num_suspicious,
            num_dangerous=num_dangerous,
            safe_pct=round(num_safe / total * 100, 2) if total > 0 else 0,
            suspicious_pct=round(num_suspicious / total * 100, 2) if total > 0 else 0,
            dangerous_pct=round(num_dangerous / total * 100, 2) if total > 0 else 0,
            gls_threshold=ver_cfg.gls_threshold,
            logit_rank_threshold=ver_cfg.logit_rank_threshold,
            tokens=tokens,
            generated_text=generated_text,
        )

    def _collect_gen_token_ids(outputs) -> list[int]:
        """Collect all generated token IDs from vLLM outputs."""
        ids = []
        for output in outputs:
            out_ids = output.outputs[0].token_ids
            if isinstance(out_ids, torch.Tensor):
                ids.extend(out_ids.tolist())
            elif isinstance(out_ids, tuple):
                ids.extend(list(out_ids))
            else:
                ids.extend(list(out_ids))
        return ids


def _sse_event(data: dict, event: str | None = None) -> str:
    """Format a server-sent event."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data)}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


# --- Endpoints ---

@app.get("/ui", response_class=HTMLResponse)
def ui_page():
    return get_ui_html()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def get_config():
    return {
        "model_name": DEFAULT_MODEL,
        "seed": DEFAULT_SEED,
        "gls_threshold": DEFAULT_GLS_THRESHOLD,
        "logit_rank_threshold": DEFAULT_LOGIT_RANK_THRESHOLD,
        "gpu_available": torch.cuda.is_available() if not FAUX_MODE else False,
        "faux_mode": FAUX_MODE,
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    overrides = req.config or {}
    if FAUX_MODE:
        return _faux_response(
            FAUX_TEXT,
            seed=overrides.get("seed", DEFAULT_SEED),
            n_prompts=req.n_prompts,
            gls_threshold=overrides.get("gls_threshold", DEFAULT_GLS_THRESHOLD),
            logit_rank_threshold=overrides.get("logit_rank_threshold", DEFAULT_LOGIT_RANK_THRESHOLD),
        )

    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA not available")

    gen_cfg = GenerationConfig(
        model_name=overrides.get("model_name", DEFAULT_MODEL),
        n_prompts=req.n_prompts,
        max_tokens=req.max_tokens,
        temperature=overrides.get("temperature", 1.0),
        top_k=overrides.get("top_k", 50),
        top_p=overrides.get("top_p", 0.95),
        seed=overrides.get("seed", DEFAULT_SEED),
        gpu_memory_utilization=overrides.get("gpu_memory_utilization", 0.7),
    )

    ver_cfg = VerificationConfig(
        model_name=gen_cfg.model_name,
        temperature=gen_cfg.temperature,
        top_k=gen_cfg.top_k,
        top_p=gen_cfg.top_p,
        seed=gen_cfg.seed,
        gls_threshold=overrides.get("gls_threshold", DEFAULT_GLS_THRESHOLD),
        logit_rank_threshold=overrides.get("logit_rank_threshold", DEFAULT_LOGIT_RANK_THRESHOLD),
    )

    prompts = load_prompts(gen_cfg)
    outputs = generate_with_vllm(gen_cfg, prompts)
    verification_results = verify_outputs(ver_cfg, outputs)

    gen_token_ids = _collect_gen_token_ids(outputs)
    tokenizer = AutoTokenizer.from_pretrained(gen_cfg.model_name)

    return _build_verify_response(
        ver_cfg, verification_results, gen_token_ids, tokenizer,
        model_name=gen_cfg.model_name, seed=gen_cfg.seed, n_prompts=req.n_prompts,
    )


@app.post("/verify-stream")
def verify_stream(req: VerifyRequest):
    """SSE streaming version of /verify with progress stages."""

    def generate_faux():
        overrides = req.config or {}
        yield _sse_event({"stage": "generating", "detail": "Generating (faux)..."})
        time.sleep(1)
        yield _sse_event({"stage": "loading_model", "detail": "Loading model (faux)..."})
        time.sleep(0.8)
        yield _sse_event({"stage": "verifying", "detail": "Verifying (faux)..."})
        time.sleep(0.6)
        response = _faux_response(
            FAUX_TEXT,
            seed=overrides.get("seed", DEFAULT_SEED),
            n_prompts=req.n_prompts,
            gls_threshold=overrides.get("gls_threshold", DEFAULT_GLS_THRESHOLD),
            logit_rank_threshold=overrides.get("logit_rank_threshold", DEFAULT_LOGIT_RANK_THRESHOLD),
        )
        yield _sse_event({"stage": "done", "result": response.model_dump()})

    if FAUX_MODE:
        return StreamingResponse(generate_faux(), media_type="text/event-stream")

    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA not available")

    def generate():
      try:
        overrides = req.config or {}

        gen_cfg = GenerationConfig(
            model_name=overrides.get("model_name", DEFAULT_MODEL),
            n_prompts=req.n_prompts,
            max_tokens=req.max_tokens,
            temperature=overrides.get("temperature", 1.0),
            top_k=overrides.get("top_k", 50),
            top_p=overrides.get("top_p", 0.95),
            seed=overrides.get("seed", DEFAULT_SEED),
            gpu_memory_utilization=overrides.get("gpu_memory_utilization", 0.7),
        )

        ver_cfg = VerificationConfig(
            model_name=gen_cfg.model_name,
            temperature=gen_cfg.temperature,
            top_k=gen_cfg.top_k,
            top_p=gen_cfg.top_p,
            seed=gen_cfg.seed,
            gls_threshold=overrides.get("gls_threshold", DEFAULT_GLS_THRESHOLD),
            logit_rank_threshold=overrides.get("logit_rank_threshold", DEFAULT_LOGIT_RANK_THRESHOLD),
        )

        yield _sse_event({"stage": "generating", "detail": f"Generating with {gen_cfg.model_name}..."})

        prompts = load_prompts(gen_cfg)
        outputs = generate_with_vllm(gen_cfg, prompts)

        yield _sse_event({"stage": "loading_model", "detail": "Loading verification model..."})
        yield _sse_event({"stage": "verifying", "detail": "Running verification..."})

        verification_results = verify_outputs(ver_cfg, outputs)

        gen_token_ids = _collect_gen_token_ids(outputs)
        tokenizer = AutoTokenizer.from_pretrained(gen_cfg.model_name)

        response = _build_verify_response(
            ver_cfg, verification_results, gen_token_ids, tokenizer,
            model_name=gen_cfg.model_name, seed=gen_cfg.seed, n_prompts=req.n_prompts,
        )

        yield _sse_event({"stage": "done", "result": response.model_dump()})
      except Exception as e:
        import traceback
        traceback.print_exc()
        yield _sse_event({"stage": "error", "detail": f"{type(e).__name__}: {e}"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Query OpenRouter and return the response text (no verification)."""
    try:
        response_text = query_openrouter(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            seed=req.seed,
        )
    except Exception as e:
        response_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is fallback text because OpenRouter was unavailable. "
            "You can still verify this text against the local model to test the UI."
        )
        print(f"[WARN] OpenRouter failed ({e}), using fallback text")
    if not response_text:
        response_text = "Empty response fallback for UI testing."
    return QueryResponse(response_text=response_text, prompt=req.prompt)


@app.post("/verify-text", response_model=VerifyResponse)
def verify_text(req: VerifyTextRequest):
    """Verify already-queried text against the local model."""
    if FAUX_MODE:
        return _faux_response(
            req.response_text,
            seed=req.seed,
            gls_threshold=req.gls_threshold,
            logit_rank_threshold=req.logit_rank_threshold,
        )

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    prompt_token_ids = tokenizer.encode(req.prompt, add_special_tokens=True)
    response_token_ids = tokenizer.encode(req.response_text, add_special_tokens=False)

    if not response_token_ids:
        raise HTTPException(status_code=400, detail="Response text tokenized to empty sequence")

    mock_output = MockRequestOutput(
        prompt_token_ids=prompt_token_ids,
        outputs=[MockOutput(token_ids=response_token_ids)],
    )

    ver_cfg = VerificationConfig(
        model_name=DEFAULT_MODEL,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        seed=req.seed,
        gls_threshold=req.gls_threshold,
        logit_rank_threshold=req.logit_rank_threshold,
    )

    verification_results = verify_outputs(ver_cfg, [mock_output])

    return _build_verify_response(
        ver_cfg, verification_results, response_token_ids, tokenizer,
        model_name=DEFAULT_MODEL, seed=req.seed, n_prompts=1,
    )


@app.post("/verify-text-stream")
def verify_text_stream(req: VerifyTextRequest):
    """SSE streaming version of /verify-text with progress stages."""

    def generate_faux():
        yield _sse_event({"stage": "loading_model", "detail": "Loading model (faux)..."})
        time.sleep(0.5)
        yield _sse_event({"stage": "verifying", "detail": "Verifying (faux)..."})
        time.sleep(0.8)
        response = _faux_response(
            req.response_text,
            seed=req.seed,
            gls_threshold=req.gls_threshold,
            logit_rank_threshold=req.logit_rank_threshold,
        )
        yield _sse_event({"stage": "done", "result": response.model_dump()})

    if FAUX_MODE:
        return StreamingResponse(generate_faux(), media_type="text/event-stream")

    def generate():
      try:
        yield _sse_event({"stage": "loading_model", "detail": "Loading verification model..."})

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        prompt_token_ids = tokenizer.encode(req.prompt, add_special_tokens=True)
        response_token_ids = tokenizer.encode(req.response_text, add_special_tokens=False)

        if not response_token_ids:
            yield _sse_event({"stage": "error", "detail": "Response text tokenized to empty sequence"})
            return

        mock_output = MockRequestOutput(
            prompt_token_ids=prompt_token_ids,
            outputs=[MockOutput(token_ids=response_token_ids)],
        )

        ver_cfg = VerificationConfig(
            model_name=DEFAULT_MODEL,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            seed=req.seed,
            gls_threshold=req.gls_threshold,
            logit_rank_threshold=req.logit_rank_threshold,
        )

        yield _sse_event({"stage": "verifying", "detail": "Running verification..."})

        verification_results = verify_outputs(ver_cfg, [mock_output])

        response = _build_verify_response(
            ver_cfg, verification_results, response_token_ids, tokenizer,
            model_name=DEFAULT_MODEL, seed=req.seed, n_prompts=1,
        )

        yield _sse_event({"stage": "done", "result": response.model_dump()})
      except Exception as e:
        import traceback
        traceback.print_exc()
        yield _sse_event({"stage": "error", "detail": f"{type(e).__name__}: {e}"})

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
