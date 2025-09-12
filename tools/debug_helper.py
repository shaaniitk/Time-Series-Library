#!/usr/bin/env python3
"""
Debug Helper Script for Time Series Library

Usage:
    python debug_helper.py test <test_path>           # Run specific test
    python debug_helper.py module <module_path>       # Import and test module
    python debug_helper.py registry                   # Debug registry issues
    python debug_helper.py attention <attention_type> # Test attention components

Examples:
    python debug_helper.py test TestsModule/tests/modular/algorithmic/test_algorithmic_sophistication_modular.py::test_fourier_restored_capabilities
    python debug_helper.py module utils_algorithm_adapters
    python debug_helper.py registry
    python debug_helper.py attention fourier
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_test(test_path):
    """Run a specific test with detailed output."""
    import subprocess
    
    print(f"🔍 Debugging test: {test_path}")
    print("=" * 60)
    
    cmd = [
        sys.executable, "-m", "pytest", 
        test_path, 
        "-v", "-s", "--tb=long", "--capture=no"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def debug_module(module_path):
    """Import and inspect a module."""
    print(f"🔍 Debugging module: {module_path}")
    print("=" * 60)
    
    try:
        # Try to import the module
        if '.' in module_path:
            module = __import__(module_path, fromlist=[''])
        else:
            module = __import__(module_path)
        
        print(f"✅ Successfully imported {module_path}")
        print(f"📍 Module location: {getattr(module, '__file__', 'Built-in')}")
        
        # List public attributes
        attrs = [attr for attr in dir(module) if not attr.startswith('_')]
        if attrs:
            print(f"📋 Public attributes: {', '.join(attrs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to import {module_path}: {e}")
        print("📋 Traceback:")
        traceback.print_exc()
        return False

def debug_registry():
    """Debug registry-related issues."""
    print("🔍 Debugging registry system")
    print("=" * 60)
    
    try:
        # Test core registry imports
        from layers.modular.core.registry import component_registry, ComponentFamily
        print("✅ Successfully imported component_registry and ComponentFamily")
        
        # Check registry state
        print(f"📋 Registry type: {type(component_registry)}")
        
        # Try to list registered components
        if hasattr(component_registry, '_components'):
            components = component_registry._components
            print(f"📋 Registered components: {len(components)} total")
            for family, comps in components.items():
                print(f"  - {family}: {list(comps.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Registry debug failed: {e}")
        traceback.print_exc()
        return False

def debug_attention(attention_type):
    """Debug specific attention component."""
    print(f"🔍 Debugging attention type: {attention_type}")
    print("=" * 60)
    
    try:
        # Import registry
        from layers.modular.core.registry import component_registry, ComponentFamily
        
        # Map attention types to component names
        attention_map = {
            'fourier': 'restored_fourier_attention',
            'autocorr': 'restored_autocorrelation_attention', 
            'meta': 'restored_meta_learning_attention'
        }
        
        component_name = attention_map.get(attention_type, attention_type)
        
        # Try to create the component
        config = {
            'd_model': 64,
            'seq_len': 48, 
            'num_heads': 4,
            'dropout': 0.0
        }
        
        print(f"📋 Creating {component_name} with config: {config}")
        
        component = component_registry.create(
            component_name, 
            ComponentFamily.ATTENTION, 
            **config
        )
        
        print(f"✅ Successfully created {component_name}")
        print(f"📋 Component type: {type(component)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Attention debug failed: {e}")
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    
    command = sys.argv[1].lower()
    
    if command == 'test' and len(sys.argv) >= 3:
        success = debug_test(sys.argv[2])
    elif command == 'module' and len(sys.argv) >= 3:
        success = debug_module(sys.argv[2])
    elif command == 'registry':
        success = debug_registry()
    elif command == 'attention' and len(sys.argv) >= 3:
        success = debug_attention(sys.argv[2])
    else:
        print(__doc__)
        return 1
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())