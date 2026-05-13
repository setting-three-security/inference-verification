#!/bin/bash
# Pull experiment outputs from the RunPod pod down to the Mac.
#
# Run from inside the inference-verification/ directory:
#   ./scripts/sync_from_pod.sh
#
# Configure POD_HOST, POD_SSH_PORT, POD_PATH below or via env vars.
# Optionally source pod connection details from a gitignored .env.runpod:
#   export POD_HOST=root@xxxxx-xxxxxxxx.proxy.runpod.net
#   export POD_SSH_PORT=22
#
# One-way pod → Mac. The pod is authoritative for .index.json — rsync's
# default behavior of overwriting the Mac copy with the pod copy is what we
# want. Conflict-free by construction: timestamped dirs make every transfer
# purely additive on the Mac side.

set -euo pipefail

# Resolve the subrepo root and chdir there so relative LOCAL_PATH resolves
# against inference-verification/, not the user's CWD.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

if [ -f .env.runpod ]; then
  # shellcheck disable=SC1091
  set -a; . ./.env.runpod; set +a
fi

POD_HOST="${POD_HOST:?set POD_HOST, e.g. root@xxxxx-xxxxxxxx.proxy.runpod.net (or in .env.runpod)}"
POD_SSH_PORT="${POD_SSH_PORT:-22}"
POD_PATH="${POD_PATH:-/workspace/inference-verification/experiments/}"
LOCAL_PATH="${LOCAL_PATH:-./experiments/}"

echo "================================================================="
echo "Syncing ${POD_HOST}:${POD_PATH}  →  ${REPO_ROOT}/${LOCAL_PATH#./}"
echo "================================================================="

mkdir -p "${LOCAL_PATH}"

rsync -avz --partial --info=progress2 \
  -e "ssh -q -o LogLevel=ERROR -p ${POD_SSH_PORT} ${POD_SSH_KEY_PATH:+-i ${POD_SSH_KEY_PATH}}" \
  "${POD_HOST}:${POD_PATH}" \
  "${LOCAL_PATH}"

echo "================================================================="
echo "Done. Run cross-row analysis locally with:"
echo "  python -m inference_verification.run_analysis experiments.yaml --combined"
echo "================================================================="
