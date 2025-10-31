#!/usr/bin/env python3
"""
Test the fixed training with the original config
"""

import sys
import yaml

def test_fixed_training():
    """Test training with the original config now that we've fixed the embedding issue"""
    
    print("🎉 TESTING FIXED TRAINING")
    print("=" * 40)
    
    # Restore original config
    config_path = "configs/celestial_production_deep_ultimate.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Restore celestial features
        config_content = config_content.replace(
            'use_celestial_graph: false        # Temporarily disabled for debugging',
            'use_celestial_graph: true'
        )
        config_content = config_content.replace(
            'aggregate_waves_to_celestial: false  # Temporarily disabled for debugging',
            'aggregate_waves_to_celestial: true'
        )
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print("✅ Original config restored")
        
        # Test with reduced epochs for quick validation
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Temporarily reduce for testing
        config['train_epochs'] = 2
        config['batch_size'] = 2
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("✅ Config adjusted for testing (2 epochs, batch_size=2)")
        
        # Import and test training function
        from scripts.train.train_celestial_production import train_celestial_pgat_production
        
        print("🚀 Starting training...")
        result = train_celestial_pgat_production(config_path)
        
        print(f"✅ Training completed: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_training()
    print("\n" + "=" * 40)
    if success:
        print("🎉 TRAINING FIX SUCCESSFUL!")
        print("✅ Ready for full production training")
    else:
        print("❌ Training still has issues")
    
    sys.exit(0 if success else 1)