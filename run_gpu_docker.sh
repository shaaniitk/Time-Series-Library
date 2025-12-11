#!/bin/bash

# Definition of colors
GREEN='\033[0;32m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'

echo -e "${GREEN}=== Celestial GPU Training via Docker ===${NC}"

# 1. Build the image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t celestial-pgat-gpu .

if [ $? -ne 0 ]; then
    echo "Docker build failed. Please check the logs."
    exit 1
fi

# 2. Run the container
echo -e "${YELLOW}Starting Container...${NC}"
echo "Mounting current directory to /app/Time-Series-Library"

# Start Docker with:
# - GPU devices (/dev/kfd, /dev/dri)
# - Security setting for ROCm (seccomp=unconfined)
# - Shared memory (ipc=host)
# - Environment variable for AMD Rembrandt (GFX 10.3.0)
# - Volume mount
# - Interactive mode

docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    --ipc=host \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    -e AMD_SERIALIZE_KERNEL=1 \
    -v "$(pwd):/app/Time-Series-Library" \
    celestial-pgat-gpu \
    bash -c "
        echo '--- Inside Container ---'
        echo 'Checking GPU...'
        python -c 'import torch; print(\"CUDA Available:\", torch.cuda.is_available()); print(\"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")'
        
        echo 'Starting Training...'
        python scripts/train/train_celestial_enhanced_pgat.py --config configs/celestial_enhanced_pgat.yaml
    "
