#!/usr/bin/env python3
"""Debug script to test imports."""

import sys
import os

print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:5]}")

try:
    from utils.logger import logger
    print("✓ utils.logger import successful")
except ImportError as e:
    print(f"✗ utils.logger import failed: {e}")

try:
    from configs.modular_components import ModularComponent
    print("✓ configs.modular_components import successful")
except ImportError as e:
    print(f"✗ configs.modular_components import failed: {e}")

try:
    from models.modular_autoformer import ModularAutoformer
    print("✓ models.modular_autoformer import successful")
except ImportError as e:
    print(f"✗ models.modular_autoformer import failed: {e}")