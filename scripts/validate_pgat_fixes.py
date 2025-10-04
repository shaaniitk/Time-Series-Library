#!/usr/bin/env python3
"""Validation script to ensure all PGAT fixes are working correctly."""

import torch
import torch.nn as nn
import sys
import os
import ast
import inspect

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_duplicate_methods():
    """Check for duplicate method definitions."""
    print("🔍 Checking for duplicate methods...")
    
    # Read the file and parse it
    with open('models/SOTA_Temporal_PGAT.py', 'r') as f:
        content = f.read()
    
    # Parse the AST
    tree = ast.parse(content)
    
    method_names = {}
    duplicates = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name in method_names:
                duplicates.append(node.name)
                print(f"   ❌ Duplicate method found: {node.name}")
            else:
                method_names[node.name] = node.lineno
    
    if not duplicates:
        print("   ✅ No duplicate methods found")
        return True
    return False

def check_unused_imports():
    """Check for unused imports."""
    print("\n🔍 Checking for unused imports...")
    
    with open('models/SOTA_Temporal_PGAT.py', 'r') as f:
        content = f.read()
    
    # Check for specific unused imports that were mentioned
    unused_imports = [
        'GraphTransformerLayer',
        'JointSpatioTemporalEncoding', 
        'HierarchicalGraphPositionalEncoding'
    ]
    
    found_unused = []
    for import_name in unused_imports:
        if import_name in content:
            # Check if it's actually used (not just imported)
            import_lines = [line for line in content.split('\n') if f'import {import_name}' in line or f'{import_name}' in line]
            usage_lines = [line for line in content.split('\n') if import_name in line and 'import' not in line]
            
            if len(import_lines) > 0 and len(usage_lines) <= 1:  # Only import line, no usage
                found_unused.append(import_name)
                print(f"   ❌ Unused import found: {import_name}")
    
    if not found_unused:
        print("   ✅ No unused imports found")
        return True
    return False

def check_unreachable_code():
    """Check for unreachable code."""
    print("\n🔍 Checking for unreachable code...")
    
    with open('models/SOTA_Temporal_PGAT.py', 'r') as f:
        lines = f.readlines()
    
    # Look for return statements followed by more code in the same function
    unreachable_found = False
    in_function = False
    function_name = ""
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if stripped.startswith('def '):
            in_function = True
            function_name = stripped.split('(')[0].replace('def ', '')
        elif in_function and (stripped == '' or not line.startswith(' ')):
            in_function = False
        elif in_function and stripped.startswith('return '):
            # Check if there's more code after this return in the same function
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line == '' or next_line.startswith('#'):
                    continue
                elif not lines[j].startswith(' '):  # New function/class
                    break
                elif next_line and not next_line.startswith('def ') and not next_line.startswith('class '):
                    print(f"   ❌ Unreachable code found in {function_name} at line {j+1}: {next_line}")
                    unreachable_found = True
                    break
    
    if not unreachable_found:
        print("   ✅ No unreachable code found")
        return True
    return False

def test_model_functionality():
    """Test that the model still works correctly after fixes."""
    print("\n🔍 Testing model functionality...")
    
    try:
        from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
        
        # Create test config
        class TestConfig:
            def __init__(self):
                self.enc_in = 10
                self.c_out = 3
                self.seq_len = 48
                self.pred_len = 12
                self.d_model = 256
                self.n_heads = 4
                self.dropout = 0.1
                self.enable_memory_optimization = True
                self.use_mixture_density = True
                self.use_dynamic_edge_weights = True
                self.use_autocorr_attention = True
                self.enable_dynamic_graph = True
                self.enable_structural_pos_encoding = True
                self.enable_graph_positional_encoding = True
                self.enable_graph_attention = True
                self.mdn_components = 3
                self.max_eigenvectors = 8
                self.autocorr_factor = 1
                self.use_adaptive_temporal = True
        
        config = TestConfig()
        
        # Test model initialization
        model = SOTA_Temporal_PGAT(config, mode='probabilistic')
        print("   ✅ Model initialization successful")
        
        # Test forward pass
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.c_out)
        graph = torch.randn(10, 10)
        
        with torch.no_grad():
            output = model(wave_window, target_window, graph)
        
        expected_shape = (batch_size, config.pred_len, config.c_out)
        if output.shape == expected_shape:
            print(f"   ✅ Forward pass successful: {output.shape}")
        else:
            print(f"   ❌ Forward pass shape mismatch: {output.shape} vs {expected_shape}")
            return False
        
        # Test memory stats
        stats = model.get_memory_stats()
        print(f"   ✅ Memory stats working: {stats['model_parameters']:,} parameters")
        
        # Test configuration methods
        model.configure_for_training()
        model.configure_for_inference()
        print("   ✅ Configuration methods working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dynamic_graph_logic():
    """Check that dynamic graph logic is properly implemented."""
    print("\n🔍 Checking dynamic graph logic...")
    
    with open('models/SOTA_Temporal_PGAT.py', 'r') as f:
        content = f.read()
    
    # Check for the fixed logic
    if "0.7 * base_adjacency + 0.3 * adaptive_adjacency" in content:
        print("   ✅ Dynamic graph combination logic implemented")
        return True
    else:
        print("   ❌ Dynamic graph combination logic not found")
        return False

def main():
    """Run all validation checks."""
    print("=" * 60)
    print("PGAT FIXES VALIDATION")
    print("=" * 60)
    
    checks = [
        ("Duplicate Methods", check_duplicate_methods),
        ("Unused Imports", check_unused_imports),
        ("Unreachable Code", check_unreachable_code),
        ("Dynamic Graph Logic", check_dynamic_graph_logic),
        ("Model Functionality", test_model_functionality),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:.<30} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! PGAT fixes are working correctly.")
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)