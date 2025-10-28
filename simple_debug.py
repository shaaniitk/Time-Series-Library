#!/usr/bin/env python3
"""
Simple debug script to catch the tuple error
"""
import traceback
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def debug_simple():
    """Simple debug approach"""
    print("🔍 SIMPLE DEBUG - TUPLE ERROR")
    print("=" * 50)
    
    try:
        # Import the production training function
        from scripts.train.train_celestial_production import train_celestial_pgat_production
        
        # Create minimal config
        config = {
            'seq_len': 24,
            'pred_len': 6,
            'enc_in': 113,  # Match the detected features
            'c_out': 4,
            'd_model': 128,  # Divisible by 13
            'n_heads': 8,
            'dropout': 0.1,
            'batch_size': 2,
            'train_epochs': 1,
            'learning_rate': 0.001,
            'patience': 1,
            'max_train_batches': 1,  # Just 1 batch
            
            # All components disabled
            'use_multi_scale_patching': False,
            'use_hierarchical_mapper': False,
            'use_stochastic_learner': False,
            'use_gated_graph_combiner': False,
            'use_mixture_decoder': False,
        }
        
        print("✅ Config created")
        print(f"📊 Features: {config['enc_in']}")
        print(f"📊 d_model: {config['d_model']}")
        
        # Save config to file
        import yaml
        config_path = "debug_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Config saved to {config_path}")
        
        # Try to run training
        print("🚀 Starting training...")
        result = train_celestial_pgat_production(config_path)
        
        print(f"✅ Training completed! Result: {result}")
        
    except Exception as e:
        print(f"💥 ERROR CAUGHT: {type(e).__name__}: {e}")
        print("\n🔍 FULL TRACEBACK:")
        print("-" * 50)
        traceback.print_exc()
        print("-" * 50)

if __name__ == "__main__":
    debug_simple()