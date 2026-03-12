# =============================================================================
# Slim Dockerfile for inference verification API server
# =============================================================================
# Optimized for the Tinfoil TEE environment where disk (ramdisk) is limited.
#
#   1. Uses nvidia/cuda "runtime" instead of "devel" (~2GB smaller)
#      - "runtime" has just the shared libraries needed to run CUDA apps
#
#   2. Installs only runtime deps via requirements-tee.txt (~1.5GB smaller)
#      - Drops torchvision, xformers, wandb, matplotlib, pandas, scikit-learn,
#        pytest, ipykernel, bitsandbytes, accelerate, einops, jaxtyping, hydra
#      - See changes.md for per-package rationale
#
#   3. Installs the package without its pyproject.toml dependencies
#      - pyproject.toml has the full dev dep list; --no-deps avoids that
#
# Estimated image size: ~4.5GB (down from 8.4GB)
# =============================================================================

# "runtime" = CUDA shared libs only, no compilers/headers
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install only the slim runtime dependencies
COPY requirements-tee.txt .
RUN pip3 install --no-cache-dir -r requirements-tee.txt

COPY . /app
WORKDIR /app

# --no-deps: install the inference_verification package itself without
# re-pulling the full dependency list from pyproject.toml
RUN pip3 install --no-cache-dir --no-deps .

EXPOSE 8080

CMD ["python3", "webapp/api_server.py"]
