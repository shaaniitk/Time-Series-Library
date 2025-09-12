#!/usr/bin/env python3
"""
Advanced Debug Test Script

This script tests different debugging scenarios to identify why breakpoints aren't working.

Usage:
1. Run with VS Code debugger (F5)
2. Run with embedded debugpy
3. Run with pdb
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_with_debugpy():
    """Test using debugpy directly."""
    try:
        import debugpy
        print(f"✅ debugpy version: {debugpy.__version__}")
        
        # Start debugpy server
        if not debugpy.is_client_connected():
            print("🔍 Starting debugpy server on port 5678...")
            debugpy.listen(5678)
            print("⏳ Waiting for debugger to attach...")
            print("   In VS Code: Run 'Python: Attach' configuration")
            debugpy.wait_for_client()
            print("✅ Debugger attached!")
        
        # This should trigger a breakpoint
        x = 10
        y = 20
        result = x + y  # <- Set breakpoint here when using attach mode
        print(f"Result: {result}")
        
        return True
        
    except ImportError:
        print("❌ debugpy not installed. Install with: pip install debugpy")
        return False
    except Exception as e:
        print(f"❌ debugpy error: {e}")
        return False

def test_with_pdb():
    """Test using Python's built-in pdb debugger."""
    print("🔍 Testing with pdb (Python debugger)...")
    
    import pdb
    
    x = 5
    y = 10
    
    # This will create an interactive debugging session
    print("About to enter pdb debugger...")
    print("Commands: 'n' (next), 'c' (continue), 'l' (list), 'q' (quit)")
    
    pdb.set_trace()  # This creates a breakpoint
    
    result = x + y
    print(f"PDB Result: {result}")
    
    return True

def test_basic_execution():
    """Test basic script execution without debugging."""
    print("🔍 Testing basic execution...")
    
    # Test project imports
    try:
        from layers.modular.core.registry import component_registry
        print(f"✅ Registry import successful: {type(component_registry)}")
    except Exception as e:
        print(f"❌ Registry import failed: {e}")
    
    # Test basic math
    numbers = [1, 2, 3, 4, 5]
    total = sum(numbers)
    average = total / len(numbers)
    
    print(f"Numbers: {numbers}")
    print(f"Total: {total}")
    print(f"Average: {average}")
    
    return True

def diagnose_environment():
    """Diagnose the Python environment."""
    print("🔍 Environment Diagnosis")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    
    print("\nPython path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    print("\nEnvironment variables:")
    for key in ['PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV']:
        value = os.environ.get(key, 'Not set')
        print(f"  {key}: {value}")
    
    # Check for debugpy
    try:
        import debugpy
        print(f"\n✅ debugpy available: {debugpy.__version__}")
    except ImportError:
        print("\n❌ debugpy not available")
    
    return True

def main():
    """Main function with multiple test scenarios."""
    print("🚀 Advanced Debug Test Starting...")
    print("=" * 60)
    
    # Diagnose environment first
    diagnose_environment()
    
    print("\n" + "=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'debugpy':
            print("🔧 Running in debugpy mode...")
            test_with_debugpy()
        elif mode == 'pdb':
            print("🔧 Running in pdb mode...")
            test_with_pdb()
        elif mode == 'basic':
            print("🔧 Running in basic mode...")
            test_basic_execution()
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: debugpy, pdb, basic")
            return 1
    else:
        print("🔧 Running all tests...")
        
        # Test 1: Basic execution
        print("\n📋 Test 1: Basic Execution")
        test_basic_execution()
        
        # Test 2: This line should be where you set a breakpoint in VS Code
        print("\n📋 Test 2: Breakpoint Test")
        print("👆 Set a breakpoint on the next line and run with F5")
        
        breakpoint_test_var = "This is where the breakpoint should stop"  # <- BREAKPOINT HERE
        print(f"Breakpoint variable: {breakpoint_test_var}")
        
        print("\n🎉 All tests completed!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())