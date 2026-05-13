"""
Token verification using GLS (and optionally CGS).

Loads generated outputs (RequestOutput pickle from generate.py), recomputes
logits via transformers, and writes one pkl per sigma:
  - all_prompts__sigma=<sigma>.pkl

Each per-sigma pkl is a list of per-token dicts with the keys consumed by
analysis/analyze_thresholds.py and analyze_two_step_classifier.py:
  - sampled_gumbel_scores: float
  - top_k_gumbel_scores: np.ndarray  (length = support_size)
  - sampled_support_idx: int  (rank of sampled token in support set; -1 if absent)
  - logit_rank: int           (sigma-independent; duplicated per file for convenience)
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import gc
import pickle

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import RequestOutput

from inference_verification.config import VerificationConfig
from inference_verification.manifest import (
    repo_root_from_module,
    utc_timestamp,
    write_verify_manifest,
)
from inference_verification.scoring_functions import (
    compute_convolved_gaussian_score,
    compute_gumbel_likelihood_score,
    compute_gumbel_likelihood_score_batch,
    draw_u,
    get_seed,
)

EPSILON = 1e-12


class TokenClassification(Enum):
    """Token classification categories."""

    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"


# ---------------------------------------------------------------------------
# vLLM-style logit filtering helpers (verbatim from vllm; needed because
# transformers gives raw logits and we have to recreate the sampling distribution
# exactly so GLS scores are comparable to what vllm produced at generation time).
# ---------------------------------------------------------------------------

def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Apply top-k mask to logits (from vLLM)."""
    assert len(logits.shape) == 2
    V = logits.shape[1]
    k = torch.minimum(k, torch.tensor([V], device=k.device, dtype=k.dtype))
    no_top_k_mask = k == logits.shape[1]
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = int(k.max().item())
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to logits (from vLLM)."""
    if p is None:
        if k is None:
            return logits
        return apply_top_k_only(logits, k)

    assert len(logits.shape) == 2

    if k is not None:
        V = logits.shape[1]
        k = torch.minimum(k, torch.tensor([V], device=k.device, dtype=k.dtype))

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None and (k > 0).all():
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def keep_one_token(scores: torch.Tensor, tok_idx: torch.Tensor) -> torch.Tensor:
    """Keep exactly one token (for greedy sampling)."""
    assert tok_idx.shape == scores.shape[:-1]
    out = torch.full_like(scores, float("-inf"))
    idx = tok_idx.unsqueeze(-1)
    values = torch.gather(scores, dim=-1, index=idx)
    out.scatter_(dim=-1, index=idx, src=values)
    return out


def get_probs(logits: torch.Tensor, temperature: float, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Compute probabilities from logits with temperature and top-k/top-p."""
    assert len(logits.shape) == 2

    if temperature > 0.0:
        x = logits / max(temperature, 1e-10)
    else:
        idx = torch.argmax(logits, dim=-1)
        x = keep_one_token(logits, idx)

    x = apply_top_k_top_p(x, top_k, top_p)
    probs = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
    return probs


def set_tokenizer_pad_token(tokenizer, model, model_name):
    """Set pad token if not already set."""
    if not tokenizer.pad_token and "llama" in model_name.lower():
        tokenizer.pad_token_id = (
            model.config.eos_token_id[0]
            if isinstance(model.config.eos_token_id, list)
            else model.config.eos_token_id
        )
    elif not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _as_list(x):
    """Convert tensor/tuple to list."""
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    return list(x)


def load_verification_model(model_name: str):
    """Load model and tokenizer for verification. Call once at startup."""
    print(f"Loading verification model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = set_tokenizer_pad_token(tokenizer, model, model_name)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Core verification: per-sigma, vocab-wide GLS scoring
# ---------------------------------------------------------------------------

def verify_outputs(
    cfg: VerificationConfig,
    outputs: list[RequestOutput],
    model=None,
    tokenizer=None,
) -> dict[float, list[dict]]:
    """Verify generated outputs and compute GLS scores per sigma.

    Returns ``{sigma: [token_dict, ...]}`` where each token_dict has keys
    described in the module docstring.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_verification_model(cfg.model_name)

    device = next(model.parameters()).device
    sigmas: list[float] = list(cfg.sigmas)
    results_by_sigma: dict[float, list[dict]] = {s: [] for s in sigmas}

    print(f"Verifying {len(outputs)} outputs across {len(sigmas)} sigma(s): {sigmas}")
    for output in tqdm(outputs, desc="Verifying"):
        prompt_ids = _as_list(output.prompt_token_ids)
        gen_ids = _as_list(output.outputs[0].token_ids)

        full_sequence = prompt_ids + gen_ids
        input_ids = torch.tensor([full_sequence], dtype=torch.long, device=device)

        with torch.no_grad():
            logits_BLV = model(input_ids=input_ids).logits

        logits_LV = logits_BLV.squeeze().float()
        top_k_tensor = torch.tensor([cfg.top_k], device=device) if cfg.top_k is not None else None
        top_p_tensor = torch.tensor([cfg.top_p], device=device)
        probs_LV = get_probs(logits_LV, cfg.temperature, top_k_tensor, top_p_tensor)

        # Unfiltered probabilities (only temperature) for support selection.
        if cfg.temperature > 0.0:
            unfiltered_logits_LV = logits_LV / max(cfg.temperature, 1e-10)
        else:
            unfiltered_logits_LV = logits_LV
        unfiltered_probs_LV = torch.nn.functional.softmax(
            unfiltered_logits_LV, dim=-1, dtype=torch.float32
        )

        # Initialize RNGs (Gumbel reseeded per prompt; CGS stateful across tokens).
        gumbel_gen = torch.Generator(device=device)
        gumbel_gen.manual_seed(cfg.seed)
        cgs_gen = torch.Generator(device=device)
        past_tokens: list[int] = []

        for j, sampled_token in enumerate(gen_ids):
            pos = len(prompt_ids) + j - 1
            logits_V = logits_LV[pos]

            # Logit rank (sigma-independent)
            sorted_indices = torch.argsort(logits_V, descending=True)
            logit_rank = (sorted_indices == sampled_token).nonzero(as_tuple=True)[0].item()

            # Draw exponential noise once per token (shared across sigmas — they
            # only change the noise scale applied analytically inside GLS).
            noise_V = torch.empty_like(logits_V)
            noise_V.exponential_(generator=gumbel_gen)

            # Support set: top-K tokens by unfiltered prob.
            unfiltered_probs_V = unfiltered_probs_LV[pos]
            support_size = min(cfg.support_size, unfiltered_probs_V.numel())
            support_indices = unfiltered_probs_V.topk(k=support_size).indices

            matches = torch.where(support_indices == sampled_token)[0]
            sampled_support_idx = matches[0].item() if len(matches) > 0 else -1

            # CGS bookkeeping (deterministic from seed + past tokens). We don't
            # currently store the CGS score; the sequence has to advance so the
            # generator stays consistent for any future use.
            cgs_seed = get_seed(cfg.seed, past_tokens)
            _u = draw_u(cgs_seed, cgs_gen)

            # Per-sigma scoring.
            for sigma in sigmas:
                claimed_token_score = compute_gumbel_likelihood_score(
                    logits_V=logits_V,
                    exponential_noise_V=noise_V,
                    temperature=cfg.temperature,
                    top_k=top_k_tensor,
                    top_p=top_p_tensor,
                    gold_idx=torch.tensor(sampled_token, device=device),
                    noise_sigma=sigma,
                    apply_top_k_top_p_fn=apply_top_k_top_p,
                    epsilon=EPSILON,
                )

                support_scores = compute_gumbel_likelihood_score_batch(
                    logits_V=logits_V,
                    exponential_noise_V=noise_V,
                    temperature=cfg.temperature,
                    top_k=top_k_tensor,
                    top_p=top_p_tensor,
                    gold_idx_list=support_indices,
                    noise_sigma=sigma,
                    apply_top_k_top_p_fn=apply_top_k_top_p,
                    epsilon=EPSILON,
                )

                results_by_sigma[sigma].append({
                    "sampled_gumbel_scores": float(claimed_token_score),
                    "top_k_gumbel_scores": support_scores.cpu().numpy(),
                    "sampled_support_idx": sampled_support_idx,
                    "logit_rank": logit_rank,
                })

            past_tokens.append(sampled_token)

    return results_by_sigma


def save_verification_results(
    results_by_sigma: dict[float, list[dict]],
    save_dir: str | Path,
) -> list[tuple[float, str]]:
    """Save one pkl per sigma. Returns ``[(sigma, filename), ...]``."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    written: list[tuple[float, str]] = []
    for sigma, results in results_by_sigma.items():
        filename = f"all_prompts__sigma={_format_sigma(sigma)}.pkl"
        out_file = save_path / filename
        with open(out_file, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved {len(results)} per-token results to {out_file}")
        written.append((sigma, filename))
    return written


def _format_sigma(sigma: float) -> str:
    """Format a sigma for use in filenames: trim trailing zeros, keep precision."""
    s = f"{sigma:g}"
    return s


def classify_tokens(
    verification_results: list[dict],
    gls_threshold: float,
    logit_rank_threshold: int,
) -> dict:
    """Classify tokens by GLS threshold + logit rank.

    Operates on a single-sigma list of dicts (the format saved per sigma).
    """
    num_safe = num_suspicious = num_dangerous = 0
    classifications = []

    for result in verification_results:
        gls_score = result["sampled_gumbel_scores"]
        logit_rank = result["logit_rank"]

        if gls_score > gls_threshold:
            classification = TokenClassification.SAFE
            num_safe += 1
        elif logit_rank > logit_rank_threshold:
            classification = TokenClassification.DANGEROUS
            num_dangerous += 1
        else:
            classification = TokenClassification.SUSPICIOUS
            num_suspicious += 1

        classifications.append(classification)

    return {
        "num_safe": num_safe,
        "num_suspicious": num_suspicious,
        "num_dangerous": num_dangerous,
        "classifications": classifications,
    }


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify generated tokens (multi-sigma GLS)")
    parser.add_argument("--input", type=str, required=True, help="Path to generated_outputs.pkl")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    parser.add_argument("--model", type=str, default=None, help="Model name (must match generation)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (must match generation)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k (must match generation)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (must match generation)")
    parser.add_argument("--seed", type=int, default=None, help="Seed (must match generation)")
    parser.add_argument("--sigmas", type=str, default=None,
                        help="Comma-separated list of sigma values (e.g. '0.01,1.0'). "
                             "Overrides --gumbel-sigma if both given.")
    parser.add_argument("--gumbel-sigma", type=float, default=None,
                        help="[Legacy] Single sigma value; equivalent to --sigmas <value>")
    parser.add_argument("--support-size", type=int, default=None,
                        help="Number of top tokens (by unfiltered prob) to score per position")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory")

    parser.add_argument("--classify", action="store_true",
                        help="Run classification after verification (per sigma)")
    parser.add_argument("--gls-threshold", type=float, default=None,
                        help="GLS threshold for classification")
    parser.add_argument("--logit-rank-threshold", type=int, default=None,
                        help="Logit rank threshold for classification")

    parser.add_argument("--row-name", type=str, default=None,
                        help="Experiments-table row name (recorded in manifest)")

    args = parser.parse_args()

    # Load config from YAML or defaults
    if args.config is not None:
        print(f"Loading configuration from {args.config}")
        cfg = VerificationConfig.from_yaml(args.config)
    else:
        cfg = VerificationConfig()

        if args.model is not None:
            cfg.model_name = args.model
        if args.temperature is not None:
            cfg.temperature = args.temperature
        if args.top_k is not None:
            cfg.top_k = args.top_k
        if args.top_p is not None:
            cfg.top_p = args.top_p
        if args.seed is not None:
            cfg.seed = args.seed
        if args.classify:
            cfg.classify = True

    # CLI overrides applied to either path.
    if args.sigmas is not None:
        cfg.sigmas = [float(s.strip()) for s in args.sigmas.split(",")]
    elif args.gumbel_sigma is not None:
        cfg.sigmas = [args.gumbel_sigma]
    if args.support_size is not None:
        cfg.support_size = args.support_size
    if args.gls_threshold is not None:
        cfg.gls_threshold = args.gls_threshold
    if args.logit_rank_threshold is not None:
        cfg.logit_rank_threshold = args.logit_rank_threshold

    # Save dir override
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    elif cfg.save_dir == "verification_results":
        cfg.save_dir = str(Path(args.input).parent)

    print("=" * 80)
    print("TOKEN VERIFICATION (multi-sigma GLS)")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Model: {cfg.model_name}")
    print(f"Temperature: {cfg.temperature}")
    print(f"Top-k: {cfg.top_k}")
    print(f"Top-p: {cfg.top_p}")
    print(f"Seed: {cfg.seed}")
    print(f"Sigmas: {cfg.sigmas}")
    print(f"Support size: {cfg.support_size}")
    print(f"Save dir: {cfg.save_dir}")
    print("=" * 80)

    # Load generated outputs
    print(f"\nLoading generated outputs from {args.input}...")
    with open(args.input, "rb") as f:
        outputs = pickle.load(f)
    print(f"Loaded {len(outputs)} generated outputs")

    # Verify
    started_at = utc_timestamp()
    started_at_perf = time.time()
    results_by_sigma = verify_outputs(cfg, outputs)
    duration = time.time() - started_at_perf
    print(
        f"Verified {sum(len(v) for v in results_by_sigma.values())} (token,sigma) entries "
        f"across {len(results_by_sigma)} sigma(s)"
    )

    # Save per-sigma pkls
    score_files = save_verification_results(results_by_sigma, cfg.save_dir)

    # Manifest (parent gen dir = parent of input pkl)
    parent_gen_dir = Path(args.input).resolve().parent
    write_verify_manifest(
        out_dir=cfg.save_dir,
        repo_root=repo_root_from_module(),
        row_name=args.row_name,
        cfg=cfg,
        parent_gen_dir=parent_gen_dir,
        score_files=score_files,
        started_at=started_at,
        duration_seconds=duration,
        exit_code=0,
    )

    print(f"\nDone! Per-sigma results saved to {cfg.save_dir}")

    # Optional classification per sigma
    if cfg.classify:
        print("\n" + "=" * 80)
        print("TOKEN CLASSIFICATION (per sigma)")
        print(f"GLS threshold: {cfg.gls_threshold}")
        print(f"Logit rank threshold: {cfg.logit_rank_threshold}")
        print("=" * 80)
        for sigma, results in results_by_sigma.items():
            cls = classify_tokens(
                verification_results=results,
                gls_threshold=cfg.gls_threshold,
                logit_rank_threshold=cfg.logit_rank_threshold,
            )
            total = len(results)
            print(
                f"  sigma={sigma}: total={total} "
                f"safe={cls['num_safe']} ({100 * cls['num_safe'] / total:.2f}%) "
                f"suspicious={cls['num_suspicious']} ({100 * cls['num_suspicious'] / total:.2f}%) "
                f"dangerous={cls['num_dangerous']} ({100 * cls['num_dangerous'] / total:.2f}%)"
            )


if __name__ == "__main__":
    main()
