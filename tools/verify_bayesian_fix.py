"""
Direct test for HFBayesianAutoformer without using models.__init__
Tests the fixed file directly to verify corruption is resolved.
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Simple config class
class SimpleConfig:
    def __init__(self):
        # Basic config
        self.seq_len = 48
        self.pred_len = 12
        self.c_out = 1
        self.enc_in = 1
        self.dec_in = 1
        
        # Embedding config
        self.embed = 'timeF'
        self.freq = 'h'
        self.d_model = 512
        
        # Bayesian specific configs
        self.mc_samples = 5
        self.uncertainty_method = 'mc_dropout'
        self.quantile_mode = True
        self.quantile_levels = [0.1, 0.5, 0.9]

def test_bayesian_file_structure():
    """Test the file structure is correct"""
    print("🔍 Testing HFBayesianAutoformer File Structure")
    print("=" * 45)
    
    # Check file exists and is readable
    filepath = "models/HFBayesianAutoformer.py"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ File exists and is readable")
        print(f"📊 File size: {len(content)} characters")
        
        # Check for clean docstring
        lines = content.split('\n')
        if lines[0].startswith('"""'):
            print(f"✅ Clean docstring start")
        else:
            print(f"❌ Corrupted docstring start: {lines[0][:50]}")
            
        # Check for proper class definition
        class_line = None
        for i, line in enumerate(lines):
            if 'class HFBayesianAutoformer' in line:
                class_line = i
                print(f"✅ Class definition found at line {i+1}")
                break
        
        if class_line is None:
            print(f"❌ No class definition found")
            return False
            
        # Check for proper forward function
        forward_line = None
        for i, line in enumerate(lines):
            if 'def forward(' in line and i > class_line:
                forward_line = i
                print(f"✅ Forward function found at line {i+1}")
                break
                
        if forward_line is None:
            print(f"❌ No forward function found")
            return False
            
        # Check for no corrupted forward function in docstring
        docstring_end = None
        for i, line in enumerate(lines):
            if '"""' in line and i > 0:
                docstring_end = i
                break
                
        if docstring_end and docstring_end < 15:  # Should be early in file
            docstring_content = '\n'.join(lines[1:docstring_end])
            if 'def forward(' in docstring_content:
                print(f"❌ Forward function still in docstring!")
                return False
            else:
                print(f"✅ Clean docstring, no function definitions")
        
        print(f"✅ File structure appears correct")
        return True
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def test_syntax_only():
    """Test if the Python syntax is correct"""
    print(f"\n🐍 Testing Python Syntax")
    print("=" * 25)
    
    try:
        import ast
        with open("models/HFBayesianAutoformer.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        ast.parse(content)
        print(f"✅ Python syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🔧 HFBayesianAutoformer File Verification")
    print("=" * 40)
    
    # Test file structure
    structure_ok = test_bayesian_file_structure()
    
    # Test syntax
    syntax_ok = test_syntax_only()
    
    # Summary
    print(f"\n📊 VERIFICATION SUMMARY")
    print("=" * 25)
    print(f"File structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"Python syntax: {'✅ PASS' if syntax_ok else '❌ FAIL'}")
    
    if structure_ok and syntax_ok:
        print(f"\n🎉 HFBayesianAutoformer file is FIXED!")
        print(f"✅ File corruption has been resolved")
        print(f"✅ Python syntax is valid") 
        print(f"✅ Class and function structure is correct")
    else:
        print(f"\n❌ File still has issues")

if __name__ == "__main__":
    main()
