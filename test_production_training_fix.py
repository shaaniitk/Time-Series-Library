#!/usr/bin/env python3
"""
Test script to verify the production training fixes work
"""

import sys
import os
import pickle

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_config_pickling():
    """Test that SimpleConfig can be pickled (needed for multiprocessing)"""
    print("Testing SimpleConfig Pickling...")
    
    try:
        from scripts.train.train_celestial_production import SimpleConfig
        
        # Create a simple config
        config_data = {
            'model': 'Celestial_Enhanced_PGAT',
            'd_model': 780,
            'seq_len': 500,
            'pred_len': 20,
            'num_workers': 0
        }
        
        config = SimpleConfig(config_data)
        
        # Test basic access
        print(f"✅ Basic access: model = {config.model}")
        print(f"✅ Basic access: d_model = {config.d_model}")
        
        # Test pickling
        pickled_data = pickle.dumps(config)
        unpickled_config = pickle.loads(pickled_data)
        
        print(f"✅ Pickling works: unpickled model = {unpickled_config.model}")
        print(f"✅ Pickling works: unpickled d_model = {unpickled_config.d_model}")
        
        # Test that unpickled config works the same
        assert config.model == unpickled_config.model
        assert config.d_model == unpickled_config.d_model
        
        print("✅ SimpleConfig pickling test passed!")
        return True
        
    except Exception as e:
        print(f"❌ SimpleConfig pickling test failed: {e}")
        return False

def test_config_loading():
    """Test loading the updated production config"""
    print("\nTesting Updated Production Config...")
    
    try:
        import yaml
        
        config_path = "configs/celestial_production_deep_ultimate.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check key fixes
        fixes = {
            'num_workers': (config.get('num_workers'), 0, "Single-threaded data loading"),
            'enc_in': (config.get('enc_in'), 113, "Correct input features"),
            'num_input_waves': (config.get('num_input_waves'), 113, "Correct input waves"),
            'd_model': (config.get('d_model'), 780, "Dimension compatible with 13 celestial bodies"),
            'pin_memory': (config.get('pin_memory'), False, "Disabled for CPU compatibility"),
        }
        
        print("Configuration Fixes:")
        all_good = True
        for key, (actual, expected, description) in fixes.items():
            status = "✅" if actual == expected else "❌"
            print(f"  {status} {key}: {actual} (expected {expected}) - {description}")
            if actual != expected:
                all_good = False
        
        if all_good:
            print("✅ All configuration fixes applied correctly!")
        else:
            print("❌ Some configuration fixes are missing!")
            
        return all_good
        
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PRODUCTION TRAINING FIX VALIDATION")
    print("=" * 60)
    
    success = True
    
    # Test SimpleConfig pickling
    success &= test_simple_config_pickling()
    
    # Test config loading
    success &= test_config_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL FIXES VALIDATED!")
        print("✅ SimpleConfig pickling works (no more recursion)")
        print("✅ Configuration updated for single-threaded operation")
        print("✅ Input dimensions corrected (113 features)")
        print("✅ Multiprocessing issues resolved")
        print("\n🚀 Ready to run production training:")
        print("python scripts/train/train_celestial_production.py --config configs/celestial_production_deep_ultimate.yaml")
    else:
        print("❌ SOME FIXES FAILED!")
        print("Please check the implementation before running production training")
    print("=" * 60)