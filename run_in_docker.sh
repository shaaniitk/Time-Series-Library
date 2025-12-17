#!/bin/bash

# Configuration Defaults
TYPE="nvidia"
IMAGE_NAME="celestial-pgat-gpu"
DOCKERFILE="Dockerfile.gpu"
CONTAINER_NAME="celestial_training_gpu"

# Help function
show_help() {
    echo "Usage: ./run_in_docker.sh [options] [command]"
    echo ""
    echo "Options:"
    echo "  --rocm        Use AMD ROCm configuration"
    echo "  --nvidia      Use NVIDIA CUDA configuration (default)"
    echo "  --build       Force rebuild of the Docker image"
    echo "  --interactive Run in interactive mode (bash shell)"
    echo "  --clean       Remove the container after exit"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_in_docker.sh --rocm --build           # Build AMD image"
    echo "  ./run_in_docker.sh --rocm python train.py   # Run with AMD GPU"
}

# Parse args
BUILD=false
INTERACTIVE=false
CLEAN=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rocm)
            TYPE="rocm"
            IMAGE_NAME="celestial-pgat-rocm"
            DOCKERFILE="Dockerfile.rocm"
            shift
            ;;
        --nvidia)
            TYPE="nvidia"
            IMAGE_NAME="celestial-pgat-gpu"
            DOCKERFILE="Dockerfile.gpu"
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$@"
                break
            fi
            ;;
    esac
done

# Build image if requested or if it doesn't exist
if [ "$BUILD" = true ] || [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "🔨 Building Docker image '$IMAGE_NAME' from $DOCKERFILE..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE .
    if [ $? -ne 0 ]; then
        echo "❌ Build failed."
        exit 1
    fi
    echo "✅ Build complete."
fi

# Prepare run flags
COMMON_FLAGS="--shm-size=8g -v $(pwd):/workspace -e PYTHONUNBUFFERED=1"
if [ "$CLEAN" = true ]; then
    COMMON_FLAGS="$COMMON_FLAGS --rm"
fi

if [ "$TYPE" == "rocm" ]; then
    # AMD ROCm flags
    RUN_FLAGS="$COMMON_FLAGS --device=/dev/kfd --device=/dev/dri --group-add video"
    echo "🔹 Using AMD ROCm runtime"
else
    # NVIDIA flags
    RUN_FLAGS="$COMMON_FLAGS --gpus all"
    echo "🔹 Using NVIDIA CUDA runtime"
fi

if [ -z "$COMMAND" ] || [ "$INTERACTIVE" = true ]; then
    echo "🚀 Starting interactive container ($TYPE)..."
    docker run $RUN_FLAGS -it $IMAGE_NAME /bin/bash
else
    echo "🚀 Running command in container ($TYPE)..."
    docker run $RUN_FLAGS $IMAGE_NAME $COMMAND
fi
