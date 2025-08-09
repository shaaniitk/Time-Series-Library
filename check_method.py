#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.EnhancedAutoformer import EnhancedAutoformer
print(f"Methods: {[m for m in dir(EnhancedAutoformer) if not m.startswith('_')]}")
print(f"Has get_auxiliary_loss: {hasattr(EnhancedAutoformer, 'get_auxiliary_loss')}")
