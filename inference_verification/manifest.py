"""
Manifest helpers shared by generate.py, verify.py, and run_experiments.py.

Each stage of an experiment writes a JSON manifest into its output directory
capturing what was run, on what code, and against what inputs. The runner
patches duration_seconds and exit_code into the manifest after the subprocess
returns. See dev-spec-e2e-experiment-loop-on-runpod.md §5 for schema details.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def utc_timestamp() -> str:
    """Return current UTC time as an ISO-8601 string with second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path | str, data: dict) -> None:
    """Atomic JSON write: write to <path>.tmp, fsync, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path: Path | str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file through sha256; safe for files larger than memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_file(src: Path | str, dst: Path | str) -> None:
    """Copy src → dst preserving content (used to snapshot the prompts file)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------

def _git(repo_root: Path, *args: str) -> Optional[str]:
    """Run a git command in repo_root; return stripped stdout or None on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except FileNotFoundError:
        return None


def _package_version(name: str) -> Optional[str]:
    """Return installed package version, or None if not importable."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def _gpu_info() -> dict:
    """Best-effort GPU info via torch.cuda. Empty dict if torch/CUDA unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        count = torch.cuda.device_count()
        if count == 0:
            return {}
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "count": count,
            "memory_gb": round(props.total_memory / (1024 ** 3), 1),
        }
    except Exception:
        return {}


def capture_environment(repo_root: Path | str, *, capture_diff: bool = True) -> dict:
    """Capture git state, package versions, and GPU info.

    If the working tree is dirty, the unified diff is included as the
    ``git_diff_text`` key (callers typically write it to ``<dir>/git.diff``
    and replace the field with the relative path).
    """
    repo_root = Path(repo_root)
    git_sha = _git(repo_root, "rev-parse", "HEAD")
    if git_sha is not None:
        # Detect dirty working tree (untracked + tracked changes).
        status = _git(repo_root, "status", "--porcelain")
        git_dirty = bool(status)
        diff_text = _git(repo_root, "diff", "HEAD") if (git_dirty and capture_diff) else None
    else:
        git_dirty = False
        diff_text = None

    env: dict[str, Any] = {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "git_diff": None,            # relative path written by caller if dirty
        "git_diff_text": diff_text,  # raw diff; caller pops this and writes git.diff
        "vllm_version": _package_version("vllm"),
        "transformers_version": _package_version("transformers"),
        "python_version": platform.python_version(),
        "gpu": _gpu_info(),
    }
    return env


def write_environment(env: dict, out_dir: Path) -> dict:
    """Persist git diff (if any) and return the env dict ready for the manifest.

    Mutates a copy of ``env``: pops ``git_diff_text`` and (when present) writes
    it to ``<out_dir>/git.diff``, setting ``git_diff`` to the relative path.
    """
    env = dict(env)
    diff_text = env.pop("git_diff_text", None)
    if diff_text:
        diff_path = out_dir / "git.diff"
        diff_path.write_text(diff_text)
        env["git_diff"] = "git.diff"
    return env


# ---------------------------------------------------------------------------
# Stage-specific writers
# ---------------------------------------------------------------------------

def write_generation_manifest(
    out_dir: Path | str,
    *,
    repo_root: Path | str,
    row_name: Optional[str],
    notes: Optional[str],
    cfg,                            # GenerationConfig
    prompts_source_path: Path | str,
    n_prompts_used: int,
    started_at: str,
    duration_seconds: Optional[float],
    exit_code: Optional[int],
    generated_outputs_filename: str = "generated_outputs.pkl",
    prompts_snapshot_filename: str = "prompts.snapshot.json",
) -> Path:
    """Write the generation-stage manifest. Returns the manifest path.

    Hashes the prompts source and snapshots it next to the manifest. Captures
    git/env state. Caller is expected to have already written the generated
    outputs pickle to ``out_dir / generated_outputs_filename``.
    """
    out_dir = Path(out_dir)
    repo_root = Path(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = out_dir / prompts_snapshot_filename
    snapshot_file(prompts_source_path, snapshot_path)
    prompts_sha = compute_sha256(prompts_source_path)
    prompts_rel = _path_relative_to(prompts_source_path, repo_root)

    env = write_environment(capture_environment(repo_root), out_dir)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": "generate",
        "row_name": row_name,
        "timestamp_utc": started_at,
        "duration_seconds": duration_seconds,
        "exit_code": exit_code,

        "model": cfg.model_name,
        "sampling": {
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "seed": cfg.seed,
        },
        "n_prompts_requested": cfg.n_prompts,
        "n_prompts_used": n_prompts_used,
        "max_tokens": cfg.max_tokens,
        "max_ctx_len": cfg.max_ctx_len,
        "max_model_len": cfg.max_model_len,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,

        "prompts": {
            "file": prompts_rel,
            "sha256": prompts_sha,
            "snapshot": prompts_snapshot_filename,
        },

        "outputs": {
            "generated_outputs": generated_outputs_filename,
        },

        "environment": env,

        "notes": notes,
    }

    manifest_path = out_dir / MANIFEST_FILENAME
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def write_verify_manifest(
    out_dir: Path | str,
    *,
    repo_root: Path | str,
    row_name: Optional[str],
    cfg,                            # VerificationConfig
    parent_gen_dir: Path | str,
    score_files: list[tuple[float, str]],   # [(sigma, filename), ...]
    started_at: str,
    duration_seconds: Optional[float],
    exit_code: Optional[int],
) -> Path:
    """Write the verification-stage manifest. Returns the manifest path."""
    out_dir = Path(out_dir)
    parent_gen_dir = Path(parent_gen_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parent_manifest_path = parent_gen_dir / MANIFEST_FILENAME
    parent_outputs_path = parent_gen_dir / "generated_outputs.pkl"

    parent = {
        "dir": _relative_to_safe(parent_gen_dir, out_dir),
        "manifest_sha256": compute_sha256(parent_manifest_path) if parent_manifest_path.exists() else None,
        "generated_outputs_sha256": compute_sha256(parent_outputs_path) if parent_outputs_path.exists() else None,
    }

    env_full = capture_environment(repo_root)
    env_full = write_environment(env_full, out_dir)
    # verify manifest carries a slimmer env (no vllm — verification doesn't use it)
    env = {
        "git_sha": env_full.get("git_sha"),
        "git_dirty": env_full.get("git_dirty"),
        "git_diff": env_full.get("git_diff"),
        "transformers_version": env_full.get("transformers_version"),
        "python_version": env_full.get("python_version"),
        "gpu": env_full.get("gpu"),
    }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": "verify",
        "row_name": row_name,
        "timestamp_utc": started_at,
        "duration_seconds": duration_seconds,
        "exit_code": exit_code,

        "parent_generation": parent,

        "model": cfg.model_name,
        "sampling": {
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "seed": cfg.seed,
        },

        "verification": {
            "sigmas": list(cfg.sigmas),
            "support_size": cfg.support_size,
            "classify": cfg.classify,
            "gls_threshold": cfg.gls_threshold,
            "logit_rank_threshold": cfg.logit_rank_threshold,
        },

        "outputs": {
            "scores": [{"sigma": s, "file": fn} for s, fn in score_files],
        },

        "environment": env,
    }

    manifest_path = out_dir / MANIFEST_FILENAME
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def write_analysis_manifest(
    out_dir: Path | str,
    *,
    repo_root: Path | str,
    row_name: Optional[str],
    scripts_run: list[str],
    started_at: str,
    duration_seconds: Optional[float],
    exit_code: Optional[int],
    artifacts: list[str],
) -> Path:
    """Write the analysis-stage manifest (lightweight)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root)

    git_sha = _git(repo_root, "rev-parse", "HEAD")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": "analyze",
        "row_name": row_name,
        "timestamp_utc": started_at,
        "duration_seconds": duration_seconds,
        "exit_code": exit_code,
        "scripts_run": list(scripts_run),
        "artifacts": list(artifacts),
        "environment": {
            "git_sha": git_sha,
            "python_version": platform.python_version(),
        },
    }

    manifest_path = out_dir / MANIFEST_FILENAME
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def patch_manifest(path: Path | str, **fields) -> None:
    """Read a manifest, update top-level fields, atomic-write back."""
    path = Path(path)
    data = read_json(path)
    data.update(fields)
    atomic_write_json(path, data)


# ---------------------------------------------------------------------------
# Internal path helpers
# ---------------------------------------------------------------------------

def _path_relative_to(path: Path | str, base: Path | str) -> str:
    """Return ``path`` relative to ``base`` if possible, else absolute string."""
    path = Path(path).resolve()
    base = Path(base).resolve()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _relative_to_safe(target: Path, start: Path) -> str:
    """``os.path.relpath`` with absolute fallback (no exception on different drives)."""
    try:
        return os.path.relpath(target, start)
    except ValueError:
        return str(target.resolve())


def repo_root_from_module() -> Path:
    """Return the inference-verification subrepo root (parent of this package).

    ``manifest.py`` lives at ``inference-verification/inference_verification/manifest.py``.
    The subrepo root is two parents up.
    """
    return Path(__file__).resolve().parent.parent
