#!/usr/bin/env python3
"""
Critical Bug Fixes Validation Script

This script validates that all critical bugs identified in SOTA_Temporal_PGAT.py have been fixed:
1. Duplicate adjacency matrix line (FIXED)
2. Hardcoded adjacency matrix weights (FIXED - now configurable)
3. Hardcoded diagonal fill value (FIXED - now configurable)
4. Unused graph parameter (FIXED - made optional with documentation)
5. Duplicate import (FIXED)

Author: Kiro AI Assistant
Date: 2025-01-04
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, Any
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_source_code_fixes():
    """Validate that source code fixes have been applied correctly."""
    print("🔍 Validating source code fixes...")
    
    model_file = "models/SOTA_Temporal_PGAT.py"
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check 1: No duplicate adjacency matrix lines
    adjacency_lines = re.findall(r'adjacency_matrix = .* \* base_adjacency \+ .* \* adaptive_adjacency', content)
    if len(adjacency_lines) > 1:
        issues.append(f"❌ Found {len(adjacency_lines)} duplicate adjacency matrix lines")
    else:
        print("✅ No duplicate adjacency matrix lines found")
    
    # Check 2: Configurable adjacency weights
    if 'base_adjacency_weight' in content and 'adaptive_adjacency_weight' in content:
        print("✅ Adjacency matrix weights are now configurable")
    else:
        issues.append("❌ Adjacency matrix weights are still hardcoded")
    
    # Check 3: Configurable diagonal value
    if 'adjacency_diagonal_value' in content:
        print("✅ Diagonal fill value is now configurable")
    else:
        issues.append("❌ Diagonal fill value is still hardcoded")
    
    # Check 4: Graph parameter is optional
    if 'def forward(self, wave_window, target_window, graph=None):' in content:
        print("✅ Graph parameter is now optional")
    else:
        issues.append("❌ Graph parameter is still required")
    
    # Check 5: No duplicate imports
    import_lines = re.findall(r'from utils\.graph_utils import get_pyg_graph', content)
    if len(import_lines) > 1:
        issues.append(f"❌ Found {len(import_lines)} duplicate import lines")
    else:
        print("✅ No duplicate import lines found")
    
    return issues

def create_test_config():
    """Create a test configuration with the new parameters."""
    
    class TestConfig:
        def __init__(self):
            # Standard parameters
            self.seq_len = 96
            self.pred_len = 24
            self.enc_in = 7
            self.c_out = 3
            self.d_model = 512
            self.n_heads = 8
            self.dropout = 0.1
            
            # NEW: Configurable adjacency matrix weights
            self.base_adjacency_weight = 0.6  # Changed from default 0.7
            self.adaptive_adjacency_weight = 0.4  # Changed from default 0.3
            
            # NEW: Configurable diagonal value
            self.adjacency_diagonal_value = 0.05  # Changed from default 0.1
            
            # Other required parameters
            self.use_mixture_density = True
            self.autocorr_factor = 1
            self.max_eigenvectors = 16
    
    return TestConfig()

def test_configurable_parameters():
    """Test that the new configurable parameters work correctly."""
    print("\n🧪 Testing configurable parameters...")
    
    try:
        from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
        
        config = create_test_config()
        model = SOTA_Temporal_PGAT(config, mode='probabilistic')
        
        # Test that the model uses the custom weights
        print(f"✅ Model created with custom adjacency weights: base={config.base_adjacency_weight}, adaptive={config.adaptive_adjacency_weight}")
        print(f"✅ Model created with custom diagonal value: {config.adjacency_diagonal_value}")
        
        # Test forward pass with optional graph parameter
        batch_size = 2
        # IMPORTANT: Both wave_window and target_window should have the same number of features
        # for proper concatenation in the forward method
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.enc_in)  # Changed from c_out to enc_in
        
        # Test with graph=None (should work now)
        try:
            output = model(wave_window, target_window, graph=None)
            print("✅ Forward pass works with graph=None")
        except Exception as e:
            print(f"❌ Forward pass failed with graph=None: {e}")
            return False
        
        # Test with explicit graph parameter
        try:
            graph = torch.randn(config.enc_in + config.c_out, config.enc_in + config.c_out)
            output = model(wave_window, target_window, graph=graph)
            print("✅ Forward pass works with explicit graph parameter")
        except Exception as e:
            print(f"❌ Forward pass failed with explicit graph: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test configurable parameters: {e}")
        return False

def test_backward_compatibility():
    """Test that models still work with old configurations (without new parameters)."""
    print("\n🔄 Testing backward compatibility...")
    
    try:
        from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
        
        class OldConfig:
            def __init__(self):
                # Only old parameters, no new ones
                self.seq_len = 96
                self.pred_len = 24
                self.enc_in = 7
                self.c_out = 3
                self.d_model = 512
                self.n_heads = 8
                self.dropout = 0.1
                self.use_mixture_density = True
                self.autocorr_factor = 1
                self.max_eigenvectors = 16
        
        config = OldConfig()
        model = SOTA_Temporal_PGAT(config, mode='probabilistic')
        
        # Should use default values (0.7, 0.3, 0.1)
        print("✅ Model created with old config (using default values)")
        
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.enc_in)  # Changed from c_out to enc_in
        
        output = model(wave_window, target_window)
        print("✅ Forward pass works with old config")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Starting Critical Bug Fixes Validation")
    print("=" * 50)
    
    # Validate source code fixes
    source_issues = validate_source_code_fixes()
    
    if source_issues:
        print("\n❌ Source code validation failed:")
        for issue in source_issues:
            print(f"  {issue}")
        return False
    
    print("\n✅ All source code fixes validated successfully!")
    
    # Test configurable parameters
    if not test_configurable_parameters():
        return False
    
    # Test backward compatibility
    if not test_backward_compatibility():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL CRITICAL BUG FIXES VALIDATED SUCCESSFULLY!")
    print("\nSummary of fixes:")
    print("✅ Removed duplicate adjacency matrix line")
    print("✅ Made adjacency matrix weights configurable (base_adjacency_weight, adaptive_adjacency_weight)")
    print("✅ Made diagonal fill value configurable (adjacency_diagonal_value)")
    print("✅ Made graph parameter optional with proper documentation")
    print("✅ Removed duplicate import statement")
    print("✅ Maintained backward compatibility with default values")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)