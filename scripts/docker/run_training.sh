#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-tsl-rocm:latest}
CONTAINER_NAME=${CONTAINER_NAME:-tsl-dev}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$HOME/Documents/workspace}
PROJECT_DIR=${PROJECT_DIR:-$WORKSPACE_ROOT/Time-Series-Library}
RUN_CMD=${RUN_CMD:-python scripts/train/train_celestial_production.py}

# Build image if missing
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "[run_training] Building image $IMAGE_NAME"
  docker build -t "$IMAGE_NAME" -f "$PROJECT_DIR/Dockerfile" "$PROJECT_DIR"
fi

# Run training with GPU device mappings and repo mounted
exec docker run -it --rm \
  --name "$CONTAINER_NAME" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v "$WORKSPACE_ROOT":/app \
  -w /app/Time-Series-Library \
  "$IMAGE_NAME" \
  bash -lc "$RUN_CMD"
