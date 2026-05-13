"""
Configuration dataclasses shared by generate.py, verify.py,
experiments_table.py, run_experiments.py, and run_analysis.py.

This module deliberately has no heavy runtime imports (no torch, no vllm,
no transformers) so it loads cleanly on macOS — which can't install vllm —
and other lightweight contexts (the runner, the table loader, the cross-row
analysis driver). Heavy imports stay inside generate.py / verify.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

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
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        if "model" in config_dict or "generation_params" in config_dict:
            model_config = config_dict.get("model", {})
            generation_config = config_dict.get("generation_params", {})
            merged_config = {**model_config, **generation_config}
        else:
            merged_config = config_dict

        # Drop fields that aren't on the dataclass (e.g. verify-only keys
        # appearing in a shared single-experiment YAML).
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        merged_config = {k: v for k, v in merged_config.items() if k in valid}

        return cls(**merged_config)


@dataclass
class VerificationConfig:
    """Configuration for token verification and classification."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Sampling parameters (must match generation)
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: float = 0.95
    seed: int = 42

    # Verification settings
    sigmas: list[float] = field(default_factory=lambda: [0.01, 1.0])
    support_size: int = 500     # Top-N tokens (by unfiltered prob) scored per position
    cgs_sigma: float = 0.01     # Gaussian std for CGS (currently unused)

    # Classification settings (recorded in manifest; not used for control flow here)
    classify: bool = False
    gls_threshold: float = -5.0
    logit_rank_threshold: int = 32

    # Save settings
    save_dir: str = "verification_results"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VerificationConfig":
        """Load config from YAML.

        Accepts either a flat dict of fields or the legacy
        ``{model: {...}, verification_params: {...}}`` layout.
        """
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        if "model" in config_dict or "verification_params" in config_dict:
            model_config = config_dict.get("model", {})
            verification_config = config_dict.get("verification_params", {})
            merged = {**model_config, **verification_config}
        else:
            merged = config_dict

        # Backwards-compat: legacy `gumbel_sigma: float` → sigmas: [float].
        if "gumbel_sigma" in merged and "sigmas" not in merged:
            merged["sigmas"] = [merged.pop("gumbel_sigma")]
        elif "gumbel_sigma" in merged:
            merged.pop("gumbel_sigma")

        # Drop fields not on the dataclass (e.g. n_prompts, prompts_file from a
        # shared single-experiment YAML).
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        merged = {k: v for k, v in merged.items() if k in valid}

        return cls(**merged)
