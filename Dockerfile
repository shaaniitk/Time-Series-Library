# 1. Start from the 'latest' tag, which we know works.
FROM rocm/pytorch:latest

# Keep this image generic; code will be mounted at runtime
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

# System build tools for packages like prophet, sktime, statsmodels, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-dev build-essential git curl cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtualenv inside the image
RUN python -m venv "$VENV_PATH" \
    && "$VENV_PATH/bin/pip" install --upgrade pip setuptools wheel

# Copy only requirement files to leverage Docker layer caching
WORKDIR /tmp
COPY requirements.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt

# Install all dependencies except torch/vision/audio and PyG compiled ops to avoid overwriting ROCm torch
RUN bash -lc "grep -vE '^(torch($|vision|audio)|torch-geom|torch-geometric|torch-scatter|torch-sparse|torch-cluster|torch-spline-conv)' /tmp/requirements.txt > /tmp/req-no-torch.txt" \
    && "$VENV_PATH/bin/pip" install --no-cache-dir -r /tmp/req-no-torch.txt

# Optional: install pure-Python torch_geometric if imports are expected without compiled ops
# Install pure-Python torch_geometric (HeteroData support)
RUN "$VENV_PATH/bin/pip" install --no-cache-dir torch_geometric

# Add entrypoint that ensures venv is active and can conditionally sync deps on container start
COPY scripts/docker/entrypoint.sh /usr/local/bin/tsl-entrypoint
RUN chmod +x /usr/local/bin/tsl-entrypoint

# Runtime workdir is where the repo will be mounted
WORKDIR /app/Time-Series-Library

ENTRYPOINT ["/usr/local/bin/tsl-entrypoint"]
CMD ["bash"]