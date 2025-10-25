#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=${VENV_PATH:-/opt/venv}
REQ_FILE=${REQ_FILE:-/app/Time-Series-Library/requirements.txt}
CONSTRAINTS_FILE=${CONSTRAINTS_FILE:-/app/Time-Series-Library/constraints.txt}
REQ_FILTERED=/tmp/req-no-torch.txt
REQ_HASH_FILE="$VENV_PATH/.requirements.hash"

ensure_venv() {
  if [[ ! -d "$VENV_PATH" || ! -x "$VENV_PATH/bin/python" ]]; then
    echo "[entrypoint] Creating virtualenv at $VENV_PATH"
    python -m venv "$VENV_PATH"
    "$VENV_PATH/bin/pip" install --upgrade pip setuptools wheel
  fi
}

filter_requirements() {
  if [[ -f "$REQ_FILE" ]]; then
    # Filter out torch family and PyG compiled deps to avoid overriding ROCm torch in base image
    grep -vE '^(torch($|vision|audio)|torch-geom|torch-geometric|torch-scatter|torch-sparse|torch-cluster|torch-spline-conv)' "$REQ_FILE" > "$REQ_FILTERED" || true
  else
    # No requirements present in mounted repo
    : > "$REQ_FILTERED"
  fi
}

maybe_install_requirements() {
  # If no requirements, nothing to do
  if [[ ! -s "$REQ_FILTERED" ]]; then
    return 0
  fi

  # Compute a hash of filtered requirements + optional constraints to detect change
  local current_hash
  if [[ -f "$CONSTRAINTS_FILE" ]]; then
    current_hash=$(cat "$REQ_FILTERED" "$CONSTRAINTS_FILE" | sha256sum | awk '{print $1}')
  else
    current_hash=$(cat "$REQ_FILTERED" | sha256sum | awk '{print $1}')
  fi

  local previous_hash=""
  if [[ -f "$REQ_HASH_FILE" ]]; then
    previous_hash=$(cat "$REQ_HASH_FILE")
  fi

  if [[ "$current_hash" != "$previous_hash" ]]; then
    echo "[entrypoint] Installing/updating Python packages (detected requirements change)"
    if [[ -f "$CONSTRAINTS_FILE" ]]; then
      "$VENV_PATH/bin/pip" install --no-cache-dir -r "$REQ_FILTERED" -c "$CONSTRAINTS_FILE"
    else
      "$VENV_PATH/bin/pip" install --no-cache-dir -r "$REQ_FILTERED"
    fi
    echo "$current_hash" > "$REQ_HASH_FILE"
  else
    echo "[entrypoint] Requirements unchanged; skipping installs"
  fi
}

activate_venv() {
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
}

print_gpu_info() {
  python - <<'PY'
import torch
print("[entrypoint] torch:", torch.__version__)
print("[entrypoint] HIP version:", getattr(torch.version, "hip", None))
print("[entrypoint] CUDA available:", torch.cuda.is_available())
print("[entrypoint] GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[entrypoint] Device 0:", torch.cuda.get_device_name(0))
PY
}

ensure_venv
filter_requirements
maybe_install_requirements
activate_venv
print_gpu_info || true

# Execute provided command within venv
exec "$@"
