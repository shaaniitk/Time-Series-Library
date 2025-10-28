#!/usr/bin/env python3
"""
Quick component test with comprehensive logging
Tests one configuration at a time with detailed error capture
"""

import sys
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

# Create log file
log_file = f"quick_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Redirect all output to log file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8', errors='replace')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

logger = Logger(log_file)
sys.stdout = logger
sys.stderr = logger

print("🔥 Quick Component Test with Logging")
print("=" * 50)
print(f"📝 Log file: {log_file}")
print(f"🕐 Start: {datetime.now()}")
print("=" * 50)

try:
    # Test just the baseline configuration first
    import yaml
    
    # Create a minimal working config based on your working production config
    config = {
        'model': 'Celestial_Enhanced_PGAT',
        'data': 'custom',
        'root_path': './data',
        'data_path': 'prepared_financial_data.csv',
        'features': 'MS',
        'target': 'log_Open,log_High,log_Low,log_Close',
        'embed': 'timeF',
        'freq': 'd',
        'use_gpu': True,
        
        # Minimal but sufficient for data splits
        'seq_len': 96,      # Sufficient for data splits
        'label_len': 48,
        'pred_len': 8,      # Short but reasonable prediction
        'validation_length': 200,  # Ensure enough validation data
        'test_length': 200,        # Ensure enough test data
        
        # Model - minimal but compatible
        'd_model': 52,      # 52 ÷ 13 = 4 (celestial bodies)
        'n_heads': 4,       # 52 ÷ 4 = 13 (attention)
        'e_layers': 1,      # Minimal layers
        'd_layers': 1,
        'd_ff': 64,
        'dropout': 0.1,
        
        # Dimensions
        'enc_in': 118,
        'dec_in': 118,
        'c_out': 4,
        
        # Training - ultra fast
        'train_epochs': 1,
        'batch_size': 8,    # Larger batch for fewer iterations
        'learning_rate': 0.001,
        'patience': 1,
        'lradj': 'type1',   # Simple LR schedule
        'weight_decay': 0.0001,
        
        # Celestial system - minimal
        'use_celestial_graph': True,
        'aggregate_waves_to_celestial': True,
        'celestial_fusion_layers': 1,
        'use_petri_net_combiner': True,
        'num_message_passing_steps': 1,
        'edge_feature_dim': 4,
        'use_temporal_attention': True,
        'use_spatial_attention': True,
        'bypass_spatiotemporal_with_petri': True,
        'num_input_waves': 113,
        'target_wave_indices': [0, 1, 2, 3],
        
        # Components - start with NONE
        'use_mixture_decoder': False,
        'use_stochastic_learner': False,
        'use_hierarchical_mapping': False,
        'use_hierarchical_fusion': True,
        'use_efficient_covariate_interaction': True,
        
        # Required fields
        'use_target_autocorrelation': True,
        'target_autocorr_layers': 1,
        'use_celestial_target_attention': True,
        'celestial_target_use_gated_fusion': True,
        'use_c2t_edge_bias': True,
        'c2t_edge_bias_weight': 0.1,
        'use_calendar_effects': True,
        'calendar_embedding_dim': 8,
        'reg_loss_weight': 0.0001,
        
        # Logging
        'log_interval': 1,
        'save_checkpoints': False,
        'enable_memory_diagnostics': False,
        'collect_diagnostics': False,
    }
    
    # Save config
    config_path = "quick_test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created minimal config: {config_path}")
    print("🔧 Configuration:")
    print(f"   • seq_len: {config['seq_len']}")
    print(f"   • pred_len: {config['pred_len']}")
    print(f"   • d_model: {config['d_model']} (÷13 = {config['d_model']//13})")
    print(f"   • n_heads: {config['n_heads']} (÷4 = {config['d_model']//config['n_heads']})")
    print(f"   • batch_size: {config['batch_size']}")
    print(f"   • train_epochs: {config['train_epochs']}")
    
    # Test the configuration
    print(f"\n🚀 Testing minimal configuration...")
    
    # Add project paths
    sys.path.append(str(Path(__file__).parent))
    
    from scripts.train.train_celestial_production import train_celestial_pgat_production
    
    start_time = time.time()
    success = train_celestial_pgat_production(config_path)
    end_time = time.time()
    
    print(f"\n📊 RESULT: {'SUCCESS' if success else 'FAILED'}")
    print(f"⏱️  Runtime: {end_time - start_time:.1f}s")
    
    if success:
        print("🎉 Minimal configuration works!")
        print("✅ Ready to test component variations")
    else:
        print("❌ Even minimal configuration fails")
        print("🔍 Need to debug further")

except Exception as e:
    print(f"\n💥 EXCEPTION: {e}")
    print(f"🔍 Exception type: {type(e).__name__}")
    print("\n📋 Full traceback:")
    traceback.print_exc()

finally:
    print(f"\n🕐 End: {datetime.now()}")
    print(f"📝 Complete log: {log_file}")
    
    try:
        logger.log.close()
    except:
        pass