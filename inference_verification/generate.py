"""
Text Generation using vLLM

This script handles generating text from prompts using vLLM with Gumbel-max sampling.
The generated sequences are saved for later verification.
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import torch
import vllm
from vllm import LLM, SamplingParams, RequestOutput
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Optional
import pickle
from dataclasses import dataclass, field
import gc
from datetime import datetime
import yaml

from inference_verification.manifest import (
    repo_root_from_module,
    utc_timestamp,
    write_generation_manifest,
)


# Repo-root-relative prompts directory: inference-verification/data/prompts/.
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"


def resolve_prompts_path(prompts_file: str) -> Path:
    """Resolve a prompts_file config value to an absolute path.

    Absolute paths are used as-is; relative paths resolve under PROMPTS_DIR.
    """
    p = Path(prompts_file)
    if p.is_absolute():
        return p
    return PROMPTS_DIR / p


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Generation settings
    n_prompts: int = 100
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: float = 0.95
    seed: int = 42

    # Prompts
    prompts_file: str = "prompts.json"
    max_ctx_len: int = 512

    # Save settings
    save_dir: str = "generated_outputs"

    # vLLM settings
    gpu_memory_utilization: float = 0.7
    max_model_len: Optional[int] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GenerationConfig":
        """Load configuration from YAML file.

        Accepts either a flat dict of fields or the legacy
        ``{model: {...}, generation_params: {...}}`` layout.
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if "model" in config_dict or "generation_params" in config_dict:
            model_config = config_dict.get("model", {})
            generation_config = config_dict.get("generation_params", {})
            merged_config = {**model_config, **generation_config}
        else:
            merged_config = config_dict

        return cls(**merged_config)


def load_prompts(cfg: GenerationConfig) -> list[list[int]]:
    """Load and tokenize prompts from cfg.prompts_file."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    prompts_path = resolve_prompts_path(cfg.prompts_file)
    with open(prompts_path) as f:
        conversations = json.load(f)

    tokenized_prompts = []
    unique_prompts = set()

    pbar = tqdm(total=cfg.n_prompts, desc="Loading prompts")
    for raw_prompt in conversations:
        if len(tokenized_prompts) >= cfg.n_prompts:
            break
        try:
            rendered_prompt = tokenizer.apply_chat_template(raw_prompt, tokenize=False, add_generation_prompt=True)
            tokenized_prompt = tokenizer.encode(rendered_prompt, add_special_tokens=False, return_tensors=None)

            if len(tokenized_prompt) <= cfg.max_ctx_len:
                if tuple(tokenized_prompt) not in unique_prompts:
                    unique_prompts.add(tuple(tokenized_prompt))
                    tokenized_prompts.append(tokenized_prompt)
                    pbar.update(1)
        except Exception as e:
            print(f"Warning: Failed to process prompt: {e}")

    pbar.close()
    del tokenizer
    return tokenized_prompts


def generate_with_vllm(cfg: GenerationConfig, prompts: list[list[int]]) -> list[RequestOutput]:
    """Generate sequences using vLLM with Gumbel-max sampling."""
    print(f"Loading vLLM model: {cfg.model_name}")
    llm_kwargs = {
        "model": cfg.model_name,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
    }
    if cfg.max_model_len is not None:
        llm_kwargs["max_model_len"] = cfg.max_model_len

    model = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        seed=cfg.seed,
    )

    print(f"Generating {len(prompts)} sequences...")
    # vLLM 0.17+ uses TokensPrompt dicts instead of prompt_token_ids kwarg
    token_prompts = [{"prompt_token_ids": p} for p in prompts]
    outputs = model.generate(token_prompts, sampling_params=sampling_params)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return outputs


def save_outputs(outputs: list[RequestOutput], save_dir: str) -> str:
    """Save generated outputs to pickle file."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    output_file = save_path / "generated_outputs.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(outputs, f)

    print(f"Saved {len(outputs)} generated outputs to {output_file}")
    return str(output_file)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate text using vLLM")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Optional overrides (only used if --config is not provided)
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--n-prompts", type=int, default=None, help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model sequence length")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Path to prompts JSON (relative paths resolve under data/prompts/)")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--row-name", type=str, default=None,
                        help="Experiments-table row name (recorded in manifest; set by run_experiments.py)")
    parser.add_argument("--notes", type=str, default=None,
                        help="Free-text notes recorded in the manifest")
    args = parser.parse_args()

    # Load config from YAML or use defaults
    if args.config is not None:
        print(f"Loading configuration from {args.config}")
        cfg = GenerationConfig.from_yaml(args.config)
    else:
        cfg = GenerationConfig()

        # Override config with command-line arguments
        if args.model is not None:
            cfg.model_name = args.model
        if args.n_prompts is not None:
            cfg.n_prompts = args.n_prompts
        if args.max_tokens is not None:
            cfg.max_tokens = args.max_tokens
        if args.temperature is not None:
            cfg.temperature = args.temperature
        if args.top_k is not None:
            cfg.top_k = args.top_k
        if args.top_p is not None:
            cfg.top_p = args.top_p
        if args.seed is not None:
            cfg.seed = args.seed
        if args.gpu_memory_utilization is not None:
            cfg.gpu_memory_utilization = args.gpu_memory_utilization

    # CLI overrides that apply to either path
    if args.max_model_len is not None:
        cfg.max_model_len = args.max_model_len
    if args.prompts_file is not None:
        cfg.prompts_file = args.prompts_file

    # Save dir override (applies whether using YAML or CLI args)
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    elif cfg.save_dir == "generated_outputs":
        # Create timestamped directory if using default
        datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.save_dir = f"generated_outputs/{datestr}"

    print("=" * 80)
    print("TEXT GENERATION WITH vLLM")
    print("=" * 80)
    print(f"Model: {cfg.model_name}")
    print(f"Prompts file: {cfg.prompts_file}")
    print(f"Prompts: {cfg.n_prompts}")
    print(f"Max tokens: {cfg.max_tokens}")
    print(f"Temperature: {cfg.temperature}")
    print(f"Top-k: {cfg.top_k}")
    print(f"Top-p: {cfg.top_p}")
    print(f"Seed: {cfg.seed}")
    print(f"Max model len: {cfg.max_model_len}")
    print(f"Save dir: {cfg.save_dir}")
    print("=" * 80)

    # Load prompts
    started_at = utc_timestamp()
    started_at_perf = time.time()
    prompts = load_prompts(cfg)
    print(f"Loaded {len(prompts)} prompts")

    # Generate
    outputs = generate_with_vllm(cfg, prompts)
    print(f"Generated {len(outputs)} outputs")

    # Save
    output_file = save_outputs(outputs, cfg.save_dir)

    # Manifest
    duration = time.time() - started_at_perf
    write_generation_manifest(
        out_dir=cfg.save_dir,
        repo_root=repo_root_from_module(),
        row_name=args.row_name,
        notes=args.notes,
        cfg=cfg,
        prompts_source_path=resolve_prompts_path(cfg.prompts_file),
        n_prompts_used=len(prompts),
        started_at=started_at,
        duration_seconds=duration,
        exit_code=0,
    )

    print("\nDone! Generated outputs saved to:", output_file)


if __name__ == "__main__":
    main()
