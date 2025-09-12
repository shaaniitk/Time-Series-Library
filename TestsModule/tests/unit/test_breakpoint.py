#!/usr/bin/env python3
"""
Simple test script to verify breakpoints are working in VS Code debugger.

Instructions:
1. Set a breakpoint on line 15 (the print statement)
2. Run this script using F5 with "Python: Current File" configuration
3. The debugger should stop at the breakpoint
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🔍 Testing breakpoint functionality...")  # <- Set breakpoint here

def test_function():
    """Simple test function to step through."""
    x = 10
    y = 20
    result = x + y  # <- Another good place for a breakpoint
    return result

def test_import():
    """Test importing project modules."""
    try:
        from layers.modular.core.registry import component_registry
        print(f"✅ Successfully imported component_registry: {type(component_registry)}")
        return True
    except Exception as e:
        print(f"❌ Failed to import: {e}")
        return False

def main():
    print("Starting breakpoint test...")
    
    # Test basic functionality
    result = test_function()
    print(f"Function result: {result}")
    
    # Test project imports
    import_success = test_import()
    
    if import_success:
        print("🎉 All tests passed! Breakpoints should be working.")
    else:
        print("⚠️ Import test failed, but breakpoints should still work.")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())