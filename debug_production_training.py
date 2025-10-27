#!/usr/bin/env python3
"""
Debug script to test production training with a simple config
"""

import sys
from pathlib import Path
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.train.train_celestial_production import train_celestial_pgat_production

def test_simple_config():
    """Test with the simplest possible config"""
    config_path = "results/production_workflow_tests/01_Baseline.yaml"
    
    print(f"🔍 Testing production training with: {config_path}")
    print("=" * 60)
    
    try:
        success = train_celestial_pgat_production(config_path)
        print(f"\n📊 Result: {success}")
        return success
        
    except Exception as e:
        print(f"\n💥 Exception caught: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_config()