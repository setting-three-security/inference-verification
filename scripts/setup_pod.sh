#!/bin/bash
# Bootstrap a fresh RunPod pod for this project.
#
# Run from inside the inference-verification/ directory:
#   ./scripts/setup_pod.sh
#
# Recommended pod configuration (when provisioning the pod):
#   - Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1 (or current recommended PyTorch image)
#   - GPU:   H100 80GB (primary) or L4 24GB (smaller models)
#   - Network volume: mount at /workspace/hf-cache so HF model downloads
#     persist across pod terminations (Llama-3.1-70B is ~140GB).
#   - Env vars: set HF_TOKEN as a RunPod template environment variable.
#
# What this script does (in order):
#   1) Install project + the Linux-gated extras (vllm, xformers).
#   2) Set HUGGINGFACE_HUB_CACHE=/workspace/hf-cache.
#   3) Log in to Hugging Face via $HF_TOKEN (if set).
#
# Idempotent: safe to re-run after pod restart. ~5 min on first run.

set -euo pipefail

# Resolve the inference-verification subrepo root (parent of this script's dir).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

echo "================================================================="
echo "RunPod bootstrap"
echo "  REPO_ROOT: $REPO_ROOT"
echo "  HF cache:  /workspace/hf-cache"
echo "================================================================="

# 0) System packages. RunPod PyTorch images don't ship rsync by default,
# but sync_to_pod.sh / sync_from_pod.sh need it on both ends.
echo "[0/3] Installing system packages (rsync)..."
if command -v apt-get >/dev/null; then
  apt-get update -qq
  apt-get install -y --no-install-recommends rsync
fi

# 1) Install project and Linux-only extras.
echo "[1/3] Installing project deps..."
pip install --upgrade pip
pip install -e .
echo "[1/3] Installing vllm + xformers (Linux-gated)..."
pip install vllm xformers

# 2) HF cache on the network volume so model downloads persist.
echo "[2/3] Configuring HF cache at /workspace/hf-cache..."
mkdir -p /workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache

# Persist for future shells in this pod.
PROFILE_LINE='export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache'
if ! grep -qF "$PROFILE_LINE" "${HOME}/.bashrc" 2>/dev/null; then
  echo "$PROFILE_LINE" >> "${HOME}/.bashrc"
  echo "  added HUGGINGFACE_HUB_CACHE export to ~/.bashrc"
fi

# 3) HF login if a token is provided. Don't fail the whole bootstrap if not set —
# the user may be running only public models and not need HF auth.
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[3/3] Logging in to Hugging Face..."
  hf auth login --token "$HF_TOKEN" || {
    echo "  (hf auth login failed; continuing — gated models will 401)"
  }
else
  echo "[3/3] HF_TOKEN not set; skipping hf auth login."
  echo "  (Set HF_TOKEN as a RunPod template env var if you need gated models.)"
fi

echo "================================================================="
echo "Bootstrap complete."
echo
echo "Verify with:"
echo "  python -c 'import vllm, transformers, torch; print(vllm.__version__)'"
echo
echo "Then push your local working tree from your Mac:"
echo "  ./scripts/sync_to_pod.sh"
echo
echo "...and kick off an experiments table:"
echo "  python -m inference_verification.run_experiments test.yaml"
echo "================================================================="
