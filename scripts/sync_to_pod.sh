#!/bin/bash
# Push the local working tree to a RunPod pod.
#
# Run from inside the inference-verification/ directory:
#   ./scripts/sync_to_pod.sh
#
# Configure POD_HOST, POD_SSH_PORT, POD_PATH below or via env vars.
# Optionally source pod connection details from a gitignored .env.runpod:
#   export POD_HOST=root@xxxxx-xxxxxxxx.proxy.runpod.net
#   export POD_SSH_PORT=22
#
# This is one-way (Mac → pod). Use sync_from_pod.sh to pull experiment
# results back. Code is in git on the pod too (separate clone); rsync is
# for fast iteration on WIP changes.

set -euo pipefail

# Resolve the subrepo root (parent of this script's directory) and chdir there.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

# Optionally load .env.runpod for connection details (gitignored).
if [ -f .env.runpod ]; then
  # shellcheck disable=SC1091
  set -a; . ./.env.runpod; set +a
fi

POD_HOST="${POD_HOST:?set POD_HOST, e.g. root@xxxxx-xxxxxxxx.proxy.runpod.net (or in .env.runpod)}"
POD_SSH_PORT="${POD_SSH_PORT:-22}"
POD_PATH="${POD_PATH:-/workspace/inference-verification/}"

echo "================================================================="
echo "Syncing $REPO_ROOT/  →  ${POD_HOST}:${POD_PATH}"
echo "================================================================="

rsync -avz --partial --info=progress2 \
  -e "ssh -p ${POD_SSH_PORT}" \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  --exclude '*.egg-info' \
  --exclude 'experiments/' \
  --exclude 'output/' \
  --exclude 'generated_outputs/' \
  --exclude 'gumbel_cgs_analysis_results/' \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude 'SECRETS' \
  ./ "${POD_HOST}:${POD_PATH}"

echo "================================================================="
echo "Done. Now SSH into the pod and run:"
echo "  cd /workspace/inference-verification"
echo "  python -m inference_verification.run_experiments test.yaml"
echo "================================================================="
