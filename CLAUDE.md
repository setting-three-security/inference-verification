# CLAUDE.md — inference-verification

This file provides guidance to Claude Code (claude.ai/code) when working inside the `inference-verification/` subrepo. All commands assume this directory is the working directory.

## Project Overview

Core inference-verification library: tooling for detecting LLM model-weight exfiltration via Gumbel-Max Likelihood Scores (GLS) and Convolved Gaussian Scores (CGS). See the paper [Verifying LLM Inference to Detect Model Weight Exfiltration](https://arxiv.org/abs/2511.02620).

## Terminology Mapping (Paper vs Code)

- **IPT Likelihood Score (IPT-LS)** in the paper = **Convolved Gaussian Score (CGS)** in the code
- **Gumbel Max Likelihood Score (GM-LS)** in the paper = **Gumbel Likelihood Score (GLS)** in the code

## Architecture

The core pipeline has three stages: **generate** → **verify** → **analyze**, orchestrated end-to-end by an experiments-table runner.

### Generation (`inference_verification/generate.py`)
Uses vLLM with Gumbel-max sampling to generate token sequences. Loads prompts via `prompts_file` (relative paths resolve under `data/prompts/`). Saves `generated_outputs.pkl` and a generation `manifest.json` capturing model, sampling, env, and prompts hash.

### Verification (`inference_verification/verify.py`)
Loads a HuggingFace model (via `transformers`, not vLLM) to recompute logits for each generated sequence. Computes GLS scores per token across a configurable list of `sigmas` and a `support_size`-wide support set. Saves one pkl per sigma (`all_prompts__sigma=<value>.pkl`) plus a verify `manifest.json` with parent-generation linkage.

### Experiments-table runner (`inference_verification/run_experiments.py`)
Reads a YAML experiments table (`defaults` + per-row `experiments`), drives gen → verify → analyze for each row as subprocesses, and maintains a side-car `experiments/.index.json`. Idempotent on success; auto-retries failed/running rows. Flags: `--only`, `--skip-stages`, `--force`, `--dry-run`.

### Cross-row analysis (`inference_verification/run_analysis.py`)
Manually invoked after a session. Reads `.index.json`, gathers verify dirs of done rows, and writes combined plots under `experiments/_combined/<timestamp>/`. Never called by the runner.

### Experiments table loader (`inference_verification/experiments_table.py`)
Parses + validates the YAML table, merges `defaults` with per-row overrides, enforces unique row names, and rejects plural keys that would imply Cartesian expansion.

### Manifest helpers (`inference_verification/manifest.py`)
Shared helpers: env capture (git SHA/dirty/diff, package versions, GPU info), file hashing, atomic JSON write, prompt snapshotting, stage-specific manifest writers.

### Scoring Functions (`inference_verification/scoring_functions/`)
- `gumbel_likelihood_score.py` — GLS: log-probability that a competitor token beats the claimed token under a Gaussian noise model on logits
- `convolved_gaussian_score.py` — CGS: log-probability under Gaussian-perturbed CDF sampling; uses `xxhash` for deterministic seed derivation from token history

### Analysis (`inference_verification/analysis/`)
- `analyze_thresholds.py` — Pareto frontier plots (FPR vs extractable information). Discovers `all_prompts__sigma=*.pkl` and writes outputs under `<verify_dir>/analysis/`.
- `analyze_two_step_classifier.py` — Two-step classifier using GLS + logit rank.
- `plot_multi_model_comparison.py` — Combined plots; consumed by `run_analysis.py` via `plot_combined()`.

### Web Server (`webapp/api_server.py`)
FastAPI server exposing `/verify`, `/verify-stream`, `/query`, `/verify-text`, `/health`, `/config`, and `/ui` endpoints. Deployed inside a Tinfoil TEE container. Supports `FAUX_MODE` env var for running without GPU/model loading.

### Utilities (`utils/llm_api.py`)
API wrappers for Anthropic, OpenAI, and OpenRouter. API keys are loaded from a `SECRETS/` directory (gitignored).

## Output directory layout

The runner writes everything under `experiments/`:

```
experiments/
├── .index.json                                  # side-car run-state log (gitignored)
├── <row-name>__<timestamp>/                     # one generation run dir
│   ├── manifest.json
│   ├── config.yaml
│   ├── prompts.snapshot.json
│   ├── generated_outputs.pkl
│   ├── stdout.log
│   └── verifications/
│       └── sigma=<sigmas>_seed=<seed>__<timestamp>/   # one verify run dir
│           ├── manifest.json
│           ├── all_prompts__sigma=0.01.pkl
│           ├── all_prompts__sigma=1.0.pkl
│           ├── stdout.log
│           └── analysis/
│               ├── manifest.json
│               ├── stdout.log
│               ├── fpr_vs_bitrate.pkl
│               └── images/
└── _combined/<timestamp>/                       # cross-row plots from run_analysis.py
```

## Common Commands

All commands below run from inside `inference-verification/`.

### End-to-end smoke test (one row)
```bash
python -m inference_verification.run_experiments test.yaml
```

### Run an experiments sweep
```bash
python -m inference_verification.run_experiments experiments.yaml
python -m inference_verification.run_experiments experiments.yaml --only llama-8b-baseline
python -m inference_verification.run_experiments experiments.yaml --skip-stages analyze
python -m inference_verification.run_experiments experiments.yaml --dry-run
```

### Cross-row combined plots (after a session)
```bash
python -m inference_verification.run_analysis experiments.yaml --combined
python -m inference_verification.run_analysis experiments.yaml --combined --rows row-a,row-b
```

### Generate or verify directly (one-off)
```bash
python -m inference_verification.generate --config demonstration/config_example.yaml
python -m inference_verification.verify --input <gen_dir>/generated_outputs.pkl --config demonstration/config_example.yaml
```

### Run tests (external TEE integration tests)
```bash
pytest tests/ -m "not slow"           # fast health/config tests only
pytest tests/ --url https://your-url  # full suite against a deployment
```

### Pre-commit formatting
The repo has a `pre-commit` script that runs `yapf` on staged `.py` files.

## Development Workflow

Generation requires vLLM + CUDA (Linux-only). Local development is on macOS ARM. The end-to-end loop runs on a RunPod pod that the researcher SSHes into.

- **On a fresh pod**: provision (RunPod template with PyTorch image + network volume mounted at `/workspace/hf-cache`), then `./scripts/setup_pod.sh`.
- **Iterating from Mac**: `./scripts/sync_to_pod.sh` pushes the local working tree to the pod.
- **After a session**: `./scripts/sync_from_pod.sh` pulls `experiments/` (results + `.index.json`) back to the Mac. Then run `run_analysis.py --combined` locally.

`vllm` and `xformers` are gated to `sys_platform == 'linux'` in `pyproject.toml` so the project installs cleanly on macOS.

## Key Constraints

- **Sampling parameters must match between generation and verification** — temperature, top_k, top_p, and seed must be identical or verification scores will be invalid. The runner enforces this by constructing both configs from the same row.
- Generation uses **vLLM** (GPU-optimized batched inference); verification uses **transformers** `AutoModelForCausalLM` (single-sequence forward passes for logit extraction).
- Intermediate data is serialized as **pickle** files (`.pkl`), not JSON. Manifests are JSON.
- Each verify run produces one `all_prompts__sigma=<value>.pkl` per sigma (re-scoring an old gen with new sigmas only requires a new sibling verify dir, no re-generation).
- The TEE Docker image uses `requirements-tee.txt` (slim runtime deps only), not the full dev dependencies.

## Spec

The end-to-end RunPod loop is specified in `../AI_DOCS/dev-spec-e2e-experiment-loop-on-runpod.md`. The spec is the source of truth for the directory layout, manifest schemas, and re-run semantics.
