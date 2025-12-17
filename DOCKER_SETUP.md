# Docker GPU Training Setup

This project supports training specifically on NVIDIA GPUs via Docker.

## Prerequisites
- **Docker** installed
- **NVIDIA Driver** installed
- **NVIDIA Container Toolkit** installed (for `--gpus all` support)

## Files
- `Dockerfile.gpu`: Defines the environment (PyTorch 2.1, CUDA 12.1).
- `run_in_docker.sh`: Helper script to build and run the container.

## How to Run

### 1. Build and Enter Shell (Recommended)

**For NVIDIA Users:**
```bash
./run_in_docker.sh --nvidia --build
```

**For AMD (ROCm) Users:**
```bash
./run_in_docker.sh --rocm --build
```

### 2. Run Training Directly
Pass your command as usual:
```bash
# NVIDIA
./run_in_docker.sh --nvidia python scripts/train/train_celestial_enhanced_pgat.py ...

# AMD
./run_in_docker.sh --rocm python scripts/train/train_celestial_enhanced_pgat.py ...
```

### 3. Force Rebuild
```bash
./run_in_docker.sh --rocm --build
```

## GPU Check
Inside the container:
```bash
python -c "import torch; print(f'CUDA (ROCm) available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```

> Values of `torch.cuda.is_available()` will return `True` for ROCm if properly configured, as PyTorch maps ROCm to the CUDA API.
