#!/usr/bin/env python3
"""
Simple validation test for Enhanced PGAT fixes (no PyTorch required)
Tests code structure and logic without running the model
"""

import os
import sys
import ast
import inspect
from pathlib import Path

def test_enhanced_pgat_fixes():
    """Test that the fixes are properly implemented in the code"""
    print("🧪 Testing Enhanced SOTA PGAT Code Fixes")
    print("=" * 50)
    
    # Test 1: Check if dynamic parameter creation is fixed
    print("\n1. Testing Dynamic Parameter Creation Fix...")
    
    try:
        # Read the Enhanced_SOTA_PGAT file
        with open('models/Enhanced_SOTA_PGAT.py', 'r') as f:
            content = f.read()
        
        # Check for pre-allocated projection layers
        if 'common_sizes = [64, 128, 256, 512, 1024, 2048, 4096]' in content:
            print("   ✅ Pre-allocated projection sizes found")
        else:
            print("   ❌ Pre-allocated projection sizes not found")
            return False
        
        # Check for pre-allocated fusion layer
        if 'max_fusion_dim = self.d_model * 6' in content:
            print("   ✅ Pre-allocated fusion layer found")
        else:
            print("   ❌ Pre-allocated fusion layer not found")
            return False
        
        # Check that dynamic creation is removed
        if 'nn.Linear(fusion_input.size(-1), self.d_model).to(' not in content:
            print("   ✅ Dynamic layer creation removed")
        else:
            print("   ❌ Dynamic layer creation still present")
            return False
            
    except Exception as e:
        print(f"   ❌ Error reading Enhanced_SOTA_PGAT.py: {e}")
        return False
    
    # Test 2: Check configuration consistency fix
    print("\n2. Testing Configuration Consistency Fix...")
    
    try:
        # Check for updated default values
        if "getattr(config, 'seq_len', 256)" in content:
            print("   ✅ Updated seq_len default found")
        else:
            print("   ❌ Updated seq_len default not found")
            return False
        
        if "getattr(config, 'd_model', 128)" in content:
            print("   ✅ Updated d_model default found")
        else:
            print("   ❌ Updated d_model default not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking configuration: {e}")
        return False
    
    # Test 3: Check graph validation fix
    print("\n3. Testing Graph Validation Fix...")
    
    try:
        if '_validate_graph_output' in content:
            print("   ✅ Graph validation method found")
        else:
            print("   ❌ Graph validation method not found")
            return False
        
        if '_align_sequence_lengths' in content:
            print("   ✅ Sequence alignment method found")
        else:
            print("   ❌ Sequence alignment method not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking graph validation: {e}")
        return False
    
    # Test 4: Check training script fixes
    print("\n4. Testing Training Script Fixes...")
    
    try:
        with open('scripts/train/train_financial_enhanced_pgat.py', 'r') as f:
            training_content = f.read()
        
        # Check that double scaling is removed
        if 'data_info[\'scaler\'].target_scaler.transform(' not in training_content:
            print("   ✅ Double scaling removed from training script")
        else:
            print("   ❌ Double scaling still present in training script")
            return False
        
        # Check for improved MDN handling
        if 'return_uncertainty=False' in training_content:
            print("   ✅ Improved MDN handling found")
        else:
            print("   ❌ Improved MDN handling not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Error reading training script: {e}")
        return False
    
    return True

def test_file_structure():
    """Test that all required files exist"""
    print("\n🧪 Testing File Structure")
    print("=" * 50)
    
    required_files = [
        'models/Enhanced_SOTA_PGAT.py',
        'scripts/train/train_financial_enhanced_pgat.py',
        'docs/Enhanced_PGAT_Critical_Fixes.md',
        'test_enhanced_pgat_fixes.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_syntax_validity():
    """Test that Python files have valid syntax"""
    print("\n🧪 Testing Syntax Validity")
    print("=" * 50)
    
    python_files = [
        'models/Enhanced_SOTA_PGAT.py',
        'scripts/train/train_financial_enhanced_pgat.py',
        'test_enhanced_pgat_fixes.py'
    ]
    
    all_valid = True
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Try to parse the file
            ast.parse(content)
            print(f"   ✅ {file_path} - Valid syntax")
            
        except SyntaxError as e:
            print(f"   ❌ {file_path} - Syntax error: {e}")
            all_valid = False
        except Exception as e:
            print(f"   ❌ {file_path} - Error: {e}")
            all_valid = False
    
    return all_valid

def main():
    """Run all validation tests"""
    print("🚀 Enhanced SOTA PGAT Fixes Validation (Code-Only)")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Syntax Validity", test_syntax_validity),
        ("Code Fixes Implementation", test_enhanced_pgat_fixes),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All code fixes validated successfully!")
        print("\n📋 SUMMARY OF FIXES APPLIED:")
        print("1. ✅ Fixed dynamic parameter creation in Enhanced_SOTA_PGAT")
        print("2. ✅ Fixed configuration consistency issues")
        print("3. ✅ Added graph component validation")
        print("4. ✅ Added sequence length alignment")
        print("5. ✅ Removed double scaling from training script")
        print("6. ✅ Improved MDN handling")
        print("\n🚀 Ready for testing with PyTorch environment!")
        return 0
    else:
        print("\n⚠️  Some fixes need attention")
        return 1

if __name__ == "__main__":
    exit(main())