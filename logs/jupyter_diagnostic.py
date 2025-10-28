import sys
import torch
import os

print("=== Jupyter vs Script Environment Diagnostic ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch location: {torch.__file__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")

print(f"Current working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

# Check if we're in the right virtual environment
venv_path = os.environ.get('VIRTUAL_ENV', 'Not in virtual environment')
print(f"Virtual environment: {venv_path}")